"""分层对话记忆管理:按客户身份维护"实时上下文 + 滚动摘要"两层记忆,并持久化到磁盘。

分层设计:
- 实时上下文层(短期):`messages` 保留最近若干轮逐字对话,供 LLM 精确理解"它/那"等指代;
  超过 `_MAX_TURNS` 触发压缩。
- 摘要压缩层(长期):`summary` 由 LLM 把溢出的旧消息合并进既有摘要,滚动更新;
  LLM 组 prompt 时以【历史对话摘要】注入,实现超长对话不失忆。
- 持久化:整份记忆写入 `memory_store.json`,服务重启不丢失;LLM 不可用时压缩跳过,
  工作区按 FIFO 硬上限截断(行为退化为旧版)。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_MAX_TURNS = 20    # 工作区上限(user+assistant 合计),超出触发摘要压缩
_KEEP_RECENT = 12  # 压缩后保留的近期逐字消息数
_HARD_CAP = 40     # LLM 不可用时的 FIFO 硬上限,防止无界增长
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store.json")

_SUMMARY_PROMPT = (
    "你是对话摘要器。请把【已有摘要】与【新对话片段】合并成一份要点式摘要,"
    "供后续问答时作为对话背景使用。要求:\n"
    "1. 保留客户问过的问题、得到的关键事实(如有文档编号一并保留)、尚未解决的疑问;\n"
    "2. 去掉寒暄与重复;总长不超过 300 字;直接输出摘要本身,不要任何解释。\n\n"
    "【已有摘要】\n{prev}\n\n【新对话片段】\n{transcript}"
)


@dataclass
class Conversation:
    client_key: str
    messages: List[dict] = field(default_factory=list)  # 实时上下文层
    summary: str = ""  # 摘要压缩层

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def history_for_llm(self) -> List[dict]:
        return list(self.messages)

    def maybe_compress(self, invoke_text: Callable[[List[dict]], str]) -> bool:
        """工作区溢出时,把最旧消息经 LLM 压缩合并进长期摘要。成功返回 True。

        任何失败(LLMError/空结果)都不丢数据:溢出消息原样放回,下次再试。
        """
        if len(self.messages) <= _MAX_TURNS:
            return False
        split = len(self.messages) - _KEEP_RECENT
        overflow, self.messages = self.messages[:split], self.messages[split:]
        transcript = "\n".join(
            f"{'客户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in overflow
        )
        prompt = _SUMMARY_PROMPT.format(prev=self.summary or "无", transcript=transcript)
        try:
            new_summary = (invoke_text([{"role": "user", "content": prompt}]) or "").strip()
        except Exception:
            new_summary = ""
        if not new_summary:
            self.messages = overflow + self.messages  # 压缩失败,恢复现场
            return False
        self.summary = new_summary
        return True

    def _trim_hard(self) -> None:
        """无 LLM 时的兜底:硬上限 FIFO,行为等价于旧版。"""
        if len(self.messages) > _HARD_CAP:
            self.messages = self.messages[-_HARD_CAP:]

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def to_dict(self) -> dict:
        return {"client_key": self.client_key, "summary": self.summary, "messages": self.messages}

    @staticmethod
    def from_dict(d: dict) -> "Conversation":
        return Conversation(
            client_key=d["client_key"],
            messages=list(d.get("messages") or []),
            summary=d.get("summary") or "",
        )


class ConversationManager:
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        self._lock = threading.Lock()
        self._load()

    def get(self, client_key: str) -> Conversation:
        with self._lock:
            if client_key not in self._conversations:
                self._conversations[client_key] = Conversation(client_key=client_key)
            return self._conversations[client_key]

    def clear(self, client_key: str) -> None:
        with self._lock:
            if client_key in self._conversations:
                self._conversations[client_key].clear()
        self.save()

    def reset_all(self) -> None:
        with self._lock:
            self._conversations.clear()
        self.save()

    def save(self) -> None:
        """持久化整份记忆;失败静默(记忆是可选增强,不能影响主流程)。"""
        try:
            data = [c.to_dict() for c in self._conversations.values()]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if not os.path.exists(MEMORY_FILE):
                return
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                conv = Conversation.from_dict(d)
                conv._trim_hard()
                self._conversations[conv.client_key] = conv
        except Exception:
            pass

    def compress_and_save(self, conv: Conversation, invoke_text: Callable[[List[dict]], str]) -> bool:
        """压缩工作区并持久化;供请求结束后调用。"""
        with self._lock:
            changed = conv.maybe_compress(invoke_text)
            if changed or len(conv.messages) > _HARD_CAP:
                conv._trim_hard()
            self.save()
            return changed


_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager
