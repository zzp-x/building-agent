"""LLM 调用模块(LangChain 模型类版)。

能力:
- 自动发现可用的 LLM API(探测逻辑与旧版一致):
  1. .env / 环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (OpenAI 兼容)
  2. 环境变量 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL
  3. 从 TRAE 用户 settings.json 中自动发现已配置的 key
- 按 kind 构造 langchain-openai.ChatOpenAI 或 langchain-anthropic.ChatAnthropic,
  由 LangGraph 节点通过 invoke_text / stream_text 调用
- 所有异常在模块边界统一转为 LLMError,由上层(状态图条件边)决定是否降级

设计原则:文档内容一律视为数据,通过 system prompt 明确告知模型不得执行文档中的"指令"。
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _load_dotenv() -> None:
    """极简 .env 加载器:读取当前文件同目录下的 .env,设置到环境变量(不覆盖已存在的)。

    .env 不入库,用于本地存放 API Key,避免硬编码进源码。
    形如: LLM_API_KEY=sk-xxx
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()


class LLMError(Exception):
    pass


def _find_trae_settings_key() -> Optional[dict]:
    """从 TRAE 用户 settings.json 中提取已配置的 LLM 相关环境变量。"""
    candidates = [
        os.path.join(
            os.environ.get("APPDATA", ""), "Trae CN", "User", "settings.json"
        ),
        os.path.join(
            os.environ.get("APPDATA", ""), "Trae", "User", "settings.json"
        ),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            continue
        env_list = settings.get("claudeCode.environmentVariables") or settings.get(
            "chat.agent.environmentVariables"
        )
        if not env_list:
            continue
        env_map = {item["name"]: item["value"] for item in env_list if item.get("name")}
        return env_map
    return None


def _detect_provider() -> Optional[dict]:
    """探测可用的 LLM 配置,返回 {base_url, api_key, model, kind} 或 None。"""
    # 1. 显式 OpenAI 兼容环境变量
    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if base and key:
        return {
            "base_url": base.rstrip("/"),
            "api_key": key,
            "model": model or "gpt-4o-mini",
            "kind": "openai",
        }

    # 2. Anthropic 兼容环境变量(如 MiniMax 的 /anthropic 端点)
    a_base = os.environ.get("ANTHROPIC_BASE_URL")
    a_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    a_model = os.environ.get("ANTHROPIC_MODEL")
    if a_base and a_key:
        return {
            "base_url": a_base.rstrip("/"),
            "api_key": a_key,
            "model": a_model or "claude-3-5-sonnet",
            "kind": "anthropic",
        }

    # 3. 从 TRAE settings 自动发现
    trae_env = _find_trae_settings_key()
    if trae_env:
        ab = trae_env.get("ANTHROPIC_BASE_URL")
        ak = trae_env.get("ANTHROPIC_AUTH_TOKEN") or trae_env.get("ANTHROPIC_API_KEY")
        am = trae_env.get("ANTHROPIC_MODEL")
        if ab and ak:
            return {
                "base_url": ab.rstrip("/"),
                "api_key": ak,
                "model": am or "MiniMax-M2.7",
                "kind": "anthropic",
            }
    return None


class LLMClient:
    """LangChain 模型类的薄封装:统一入口 + 统一异常边界(LLMError)。"""

    def __init__(self):
        self.config = _detect_provider()
        try:
            self._model = self._build_model()
            self.load_error: Optional[str] = None
        except Exception as e:  # 构造失败(依赖损坏等)按"不可用"降级,不让引擎崩溃
            self._model = None
            self.load_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _ensure_aiohttp_importable() -> None:
        """Windows 兼容性处理:证书库含损坏条目时,aiohttp 导入期的
        ssl.create_default_context() 会抛 SSLError,导致 anthropic SDK 不可用。
        导入 aiohttp 前临时放行逐条证书加载错误,导入完成后立即恢复原行为。"""
        import ssl

        if not hasattr(ssl.SSLContext, "load_verify_locations"):
            return
        orig = ssl.SSLContext.load_verify_locations

        def tolerant(ctx, *args, **kwargs):
            try:
                return orig(ctx, *args, **kwargs)
            except ssl.SSLError:
                return False  # 跳过无法解析的单条证书

        ssl.SSLContext.load_verify_locations = tolerant
        try:
            import aiohttp  # noqa: F401  anthropic SDK 的传递依赖
        except ImportError:
            pass
        finally:
            ssl.SSLContext.load_verify_locations = orig

    def _build_model(self):
        if not self.config:
            return None
        c = self.config
        if c["kind"] == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                base_url=c["base_url"],
                api_key=c["api_key"],
                model=c["model"],
                temperature=0.2,
                max_tokens=900,
                timeout=60,
                max_retries=0,  # 失败立刻抛错走降级,不做 SDK 层重试
            )
        self._ensure_aiohttp_importable()
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            base_url=c["base_url"],
            api_key=c["api_key"],
            model_name=c["model"],
            temperature=0.2,
            max_tokens=900,
            default_request_timeout=60,
            max_retries=0,
        )

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def provider_info(self) -> str:
        if self._model is None:
            info = "未检测到可用的 LLM API"
            if self.load_error:
                info += f"(加载失败: {self.load_error})"
            return info
        c = self.config
        return f"{c['kind']} | {c['base_url']} | model={c['model']}"

    @staticmethod
    def _to_lc_messages(messages: List[dict]):
        lc = []
        for m in messages:
            role = m["role"]
            if role == "system":
                lc.append(SystemMessage(content=m["content"]))
            elif role == "assistant":
                lc.append(AIMessage(content=m["content"]))
            else:
                lc.append(HumanMessage(content=m["content"]))
        return lc

    def invoke_text(self, messages: List[dict]) -> str:
        """非流式调用,返回完整文本。任何失败抛 LLMError。"""
        if not self._model:
            raise LLMError("未检测到可用的 LLM API,请配置 LLM_BASE_URL/LLM_API_KEY")
        try:
            resp = self._model.invoke(self._to_lc_messages(messages))
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}")
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                blk.get("text", "") for blk in content if isinstance(blk, dict)
            )
        return (content or "").strip()

    def stream_text(self, messages: List[dict]) -> Iterator[str]:
        """流式调用,yield 文本片段。任何失败(含迭代中途)抛 LLMError。"""
        if not self._model:
            raise LLMError("未检测到可用的 LLM API")
        try:
            for chunk in self._model.stream(self._to_lc_messages(messages)):
                text = chunk.content
                if isinstance(text, list):
                    text = "".join(
                        blk.get("text", "") for blk in text if isinstance(blk, dict)
                    )
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}")


_llm: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
