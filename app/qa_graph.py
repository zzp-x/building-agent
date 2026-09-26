"""LangGraph 状态图:QA 编排层(自我修正 RAG)。

图结构:
    START → retrieve →(条件路由)
      ├─ 检索到资料            → build_prompt → generate →(条件路由)
      │                                           ├─ LLM 成功 → postprocess → END
      │                                           └─ LLM 失败 → fallback    → END
      ├─ 无资料 且未改写过 且 LLM 可用 → rewrite_query → retrieve(只循环一次)
      └─ 无资料                        → insufficient → END

- retrieve / build_prompt / postprocess / fallback 复用 qa_engine 的既有实现;
- rewrite_query 用 LLM 把"上一问 + 当前问"改写成独立检索式,实现自我修正环;
- 无 LLM 环境下改写环自动短路(不尝试改写,直接按原路径降级)。
"""
from __future__ import annotations

import re
from typing import List, TypedDict
from langgraph.graph import END, START, StateGraph

from doc_loader import DocumentStore
from llm import LLMClient, LLMError
from qa_engine import (
    SYSTEM_PROMPT,
    QAContext,
    _build_context_block,
    _fallback_answer,
    _post_process,
    retrieve_chunks,
)
from retriever import Retriever


class QAState(TypedDict, total=False):
    query: str            # 用户原始问题
    effective_query: str  # 实际用于检索的问题(可能被改写过)
    client_key: str
    history: List[dict]
    summary: str  # 长期层:更早轮次的对话摘要(可为空)
    greeting: str  # 寒暄/能力咨询命中时的直接回复(空串=正常业务问题)
    chunks: List          # List[Chunk]
    messages: List[dict]
    answer: str
    sources: List[str]
    inferences: List[str]
    used_llm: bool
    llm_failed: bool
    rewrite_used: int     # 0/1:改写环只允许走一次


# 寒暄/能力咨询:短消息命中即直接回复,不检索、不调 LLM(降级模式同样可用)
_GREET_PATTERNS = [
    (
        re.compile(r"^(你好|您好|hi|hello|嗨|哈喽|在吗|在么|早|早上好|上午好|中午好|下午好|晚上好)[\s!！。.~～]*$", re.I),
        (
            "您好!我是云溪花园项目的资料助手,可以基于项目的施工日志、规范摘录、会议纪要等资料回答问题,"
            "并为每个结论标注来源文档。\n"
            "您可以这样问我:\n"
            "- 3#楼5层顶板哪天浇筑的混凝土?\n"
            "- 5层顶板什么时候可以拆模?\n"
            "- 临边作业有哪些安全要求?\n"
            "请问需要查询什么?"
        ),
    ),
    (
        re.compile(r"^(谢谢|感谢|多谢|辛苦了|thx|thanks|thank you)[\s!！。.~～]*$", re.I),
        "不客气!如还需要查询施工进度、规范要求或其他项目资料,随时告诉我。",
    ),
    (
        re.compile(r"^(再见|拜拜|bye|goodbye)[\s!！。.~～]*$", re.I),
        "再见!后续需要查询项目资料时随时回来找我。",
    ),
    (
        re.compile(r"(你是谁|你叫什么|你是做什么的|你能做什么|你会什么|有什么功能|怎么用|帮助|help)", re.I),
        (
            "我是云溪花园项目的资料问答助手,功能包括:\n"
            "1. 基于项目资料(施工日志、规范摘录、合同报价、会议纪要等)回答问题,并标注来源;\n"
            "2. 区分资料事实与模型推断,资料不足时明确告知;\n"
            "3. 按客户身份隔离资料,只回答您权限范围内的内容;\n"
            "4. 支持多轮追问,例如先问'5层顶板哪天浇筑的?'再问'那什么时候可以拆模?'。\n"
            "您可以直接输入要查询的问题。"
        ),
    ),
]


def _match_greeting(query: str) -> str:
    """识别寒暄/能力咨询;命中返回回复文本,否则返回空串。仅对短消息生效,避免误伤业务问题。"""
    q = (query or "").strip()
    if not q or len(q) > 20:
        return ""
    for pat, reply in _GREET_PATTERNS:
        if pat.search(q):
            return reply
    return ""


def build_graph(store: DocumentStore, retriever: Retriever, llm: LLMClient):
    """构建并编译 QA 状态图。依赖通过闭包注入,节点函数返回状态增量。"""

    def greet_check(state: QAState) -> dict:
        return {"greeting": _match_greeting(state["query"])}

    def route_entry(state: QAState) -> str:
        return "greet" if state.get("greeting") else "retrieve"

    def greet(state: QAState) -> dict:
        return {
            "answer": state["greeting"],
            "sources": [],
            "inferences": [],
            "used_llm": False,
        }

    def retrieve(state: QAState) -> dict:
        ctx = QAContext(
            query=state["query"],
            client_key=state["client_key"],
            history=state["history"],
        )
        chunks = retrieve_chunks(store, retriever, ctx, query_text=state["effective_query"])
        return {"chunks": chunks}

    def route_after_retrieve(state: QAState) -> str:
        if state.get("chunks"):
            return "build_prompt"
        if not state.get("rewrite_used") and llm.available:
            return "rewrite_query"
        return "insufficient"

    def rewrite_query(state: QAState) -> dict:
        last_user = ""
        for h in reversed(state["history"]):
            if h["role"] == "user":
                last_user = h["content"]
                break
        prompt = (
            "你是检索查询改写器。请结合对话上一问,把当前问题改写成一个不依赖上下文、"
            "适合关键词检索的独立问题。只输出改写后的问题本身,不要任何解释。\n\n"
            f"对话上一问:{last_user or '无'}\n当前问题:{state['query']}"
        )
        rewritten = ""
        try:
            rewritten = llm.invoke_text([{"role": "user", "content": prompt}])
        except LLMError:
            rewritten = ""
        rewritten = rewritten.strip().splitlines()[0].strip().strip('"“”') if rewritten else ""
        if not rewritten:
            rewritten = state["query"]  # 改写失败:保留原问题,二检为空自然走资料不足
        return {"effective_query": rewritten, "rewrite_used": 1}

    def build_prompt(state: QAState) -> dict:
        context_block = _build_context_block(state["chunks"], store)
        client = store.get_client(state["client_key"])
        client_note = (
            f"当前提问客户:{client.name if client else state['client_key']}。"
            "你只能使用该客户可见的资料,绝不能提及或暗示存在其他不可见文档。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + client_note}
        ]
        # 长期记忆层:更早轮次的滚动摘要(逐字历史放不下的部分)
        if state.get("summary"):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "【历史对话摘要】以下是本客户更早轮次对话的要点,"
                        "供理解上下文使用;涉及项目事实时以本次【项目资料】为准。\n"
                        + state["summary"]
                    ),
                }
            )
        for h in state["history"][-8:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append(
            {"role": "user", "content": f"{context_block}\n\n【用户问题】{state['query']}"}
        )
        return {"messages": messages}

    def generate(state: QAState) -> dict:
        # 用 stream_text 保证 messages 模式下 token 逐段流出;节点内累积完整文本
        parts = []
        try:
            for token in llm.stream_text(state["messages"]):
                parts.append(token)
        except LLMError:
            return {"llm_failed": True, "used_llm": False}
        return {"answer": "".join(parts).strip(), "llm_failed": False, "used_llm": True}

    def route_after_generate(state: QAState) -> str:
        return "fallback" if state.get("llm_failed") else "postprocess"

    def postprocess(state: QAState) -> dict:
        result = _post_process(state["answer"], state["chunks"], store)
        # LLM 没标注来源时,尝试从回答中匹配文档 id 补标(仅在客户可见集合内)
        if not result["sources"] and "资料不足" not in result["answer"]:
            ids = re.findall(r"\b(\d{2})-", result["answer"])
            visible_ids = {d.id for d in store.visible_documents(state["client_key"])}
            for did in dict.fromkeys(ids):
                doc = store.get_document(did)
                if doc and did in visible_ids:
                    result["sources"].append(doc.display_name())
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "inferences": result["inferences"],
            "used_llm": True,
        }

    def fallback(state: QAState) -> dict:
        fb = _fallback_answer(state["query"], state["chunks"], store)
        return {
            "answer": fb["answer"],
            "sources": fb["sources"],
            "inferences": fb["inferences"],
            "used_llm": False,
        }

    def insufficient(state: QAState) -> dict:
        return {
            "answer": "资料不足:现有资料中未检索到与该问题相关的内容。",
            "sources": [],
            "inferences": [],
            "used_llm": False,
        }

    g = StateGraph(QAState)
    g.add_node("greet_check", greet_check)
    g.add_node("greet", greet)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("build_prompt", build_prompt)
    g.add_node("generate", generate)
    g.add_node("postprocess", postprocess)
    g.add_node("fallback", fallback)
    g.add_node("insufficient", insufficient)

    g.add_edge(START, "greet_check")
    g.add_conditional_edges(
        "greet_check",
        route_entry,
        {"greet": "greet", "retrieve": "retrieve"},
    )
    g.add_edge("greet", END)
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "build_prompt": "build_prompt",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("build_prompt", "generate")
    g.add_conditional_edges(
        "generate",
        route_after_generate,
        {"fallback": "fallback", "postprocess": "postprocess"},
    )
    g.add_edge("postprocess", END)
    g.add_edge("fallback", END)
    g.add_edge("insufficient", END)
    return g.compile()
