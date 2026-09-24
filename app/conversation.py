"""对话记忆管理:按客户身份维护多轮对话历史,支持依赖上文的追问。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

_MAX_TURNS = 20  # 单客户保留的最大消息数(user+assistant 合计)


@dataclass
class Conversation:
    client_key: str
    messages: List[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def history_for_llm(self) -> List[dict]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        if len(self.messages) > _MAX_TURNS:
            self.messages = self.messages[-_MAX_TURNS:]


class ConversationManager:
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    def get(self, client_key: str) -> Conversation:
        if client_key not in self._conversations:
            self._conversations[client_key] = Conversation(client_key=client_key)
        return self._conversations[client_key]

    def clear(self, client_key: str) -> None:
        if client_key in self._conversations:
            self._conversations[client_key].clear()

    def reset_all(self) -> None:
        self._conversations.clear()


_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    return _manager
