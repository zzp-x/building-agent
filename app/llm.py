"""LLM 调用模块。

能力:
- 自动发现可用的 LLM API:
  1. 环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (OpenAI 兼容)
  2. 环境变量 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL
  3. 从 TRAE 用户 settings.json 中自动发现已配置的 MiniMax 等 key
- 优先使用 OpenAI 兼容端点(MiniMax、OpenAI、DeepSeek 等)
- 调用失败时抛出 LLMError,由上层决定是否降级为检索模板回答

设计原则:文档内容一律视为数据,通过 system prompt 明确告知模型不得执行文档中的"指令"。
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import requests


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
    def __init__(self):
        self.config = _detect_provider()

    @property
    def available(self) -> bool:
        return self.config is not None

    @property
    def provider_info(self) -> str:
        if not self.config:
            return "未检测到可用的 LLM API"
        c = self.config
        return f"{c['kind']} | {c['base_url']} | model={c['model']}"

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: int = 60,
    ) -> str:
        if not self.config:
            raise LLMError("未检测到可用的 LLM API,请配置 LLM_BASE_URL/LLM_API_KEY")
        c = self.config
        try:
            if c["kind"] == "openai":
                return self._call_openai(c, messages, temperature, max_tokens, timeout)
            else:
                return self._call_anthropic(
                    c, messages, temperature, max_tokens, timeout
                )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}")

    def _call_openai(self, c, messages, temperature, max_tokens, timeout) -> str:
        url = f"{c['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {c['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            "model": c["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, c, messages, temperature, max_tokens, timeout) -> str:
        # Anthropic 兼容:system 消息需单独传,role 只能是 user/assistant
        system = ""
        conv = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        url = f"{c['base_url']}/v1/messages"
        headers = {
            "x-api-key": c["api_key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": c["model"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv,
        }
        if system.strip():
            body["system"] = system.strip()
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        # content 可能是 list of blocks
        content = data.get("content", [])
        if isinstance(content, list):
            return "".join(blk.get("text", "") for blk in content).strip()
        return str(content).strip()

    # ---- 流式调用(SSE) ----

    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: int = 90,
    ):
        """流式调用 LLM,yield 文本片段。调用方负责捕获 LLMError。"""
        if not self.config:
            raise LLMError("未检测到可用的 LLM API")
        c = self.config
        if c["kind"] == "openai":
            yield from self._stream_openai(c, messages, temperature, max_tokens, timeout)
        else:
            yield from self._stream_anthropic(
                c, messages, temperature, max_tokens, timeout
            )

    def _stream_openai(self, c, messages, temperature, max_tokens, timeout):
        import json as _json

        url = f"{c['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {c['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            "model": c["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with requests.post(
            url, headers=headers, json=body, timeout=timeout, stream=True
        ) as r:
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except _json.JSONDecodeError:
                        continue

    def _stream_anthropic(self, c, messages, temperature, max_tokens, timeout):
        import json as _json

        system = ""
        conv = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        url = f"{c['base_url']}/v1/messages"
        headers = {
            "x-api-key": c["api_key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": c["model"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv,
            "stream": True,
        }
        if system.strip():
            body["system"] = system.strip()
        with requests.post(
            url, headers=headers, json=body, timeout=timeout, stream=True
        ) as r:
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    evt = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text


_llm: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
