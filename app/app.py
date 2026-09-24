"""Flask 后端:提供对话 API 与静态页面。

接口:
- GET  /                 -> 网页
- GET  /api/clients      -> 客户列表
- POST /api/chat         -> {client_key, message} -> {answer, sources, used_llm, inferences}
- POST /api/clear        -> {client_key} 清空该客户对话
- GET  /api/debug/retrieve?client_key=&q= -> 调试用:返回检索结果
"""
from __future__ import annotations

import json
import os

from flask import Flask, Response, jsonify, request, send_from_directory

from conversation import get_conversation_manager
from doc_loader import get_store
from qa_engine import QAContext, get_engine
from retriever import get_retriever

app = Flask(__name__, static_folder=None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/clients")
def clients():
    store = get_store()
    return jsonify(
        [
            {"key": c.key, "name": c.name, "role": c.role, "description": c.description}
            for c in store.all_clients()
        ]
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    client_key = data.get("client_key")
    message = (data.get("message") or "").strip()

    store = get_store()
    if not store.get_client(client_key):
        return jsonify({"error": "未知客户身份"}), 400
    if not message:
        return jsonify({"error": "消息不能为空"}), 400

    mgr = get_conversation_manager()
    conv = mgr.get(client_key)
    history = conv.history_for_llm()

    engine = get_engine()
    result = engine.answer(QAContext(query=message, client_key=client_key, history=history))

    # 记录对话历史(只记录问答文本,不记录内部检索细节)
    conv.add_user(message)
    conv.add_assistant(result["answer"])

    return jsonify(result)


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """SSE 流式对话端点。

    前端用 fetch POST + ReadableStream 消费。
    每个事件: data: {json}\n\n
    """
    data = request.get_json(force=True)
    client_key = data.get("client_key")
    message = (data.get("message") or "").strip()

    store = get_store()
    if not store.get_client(client_key):
        return jsonify({"error": "未知客户身份"}), 400
    if not message:
        return jsonify({"error": "消息不能为空"}), 400

    mgr = get_conversation_manager()
    conv = mgr.get(client_key)
    history = conv.history_for_llm()
    engine = get_engine()

    def generate():
        full_answer = ""
        for event in engine.answer_stream(
            QAContext(query=message, client_key=client_key, history=history)
        ):
            if event["type"] == "delta":
                full_answer += event["text"]
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        # 记录对话历史
        conv.add_user(message)
        conv.add_assistant(full_answer)

    return Response(
        generate(),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/clear", methods=["POST"])
def clear():
    data = request.get_json(force=True)
    client_key = data.get("client_key")
    mgr = get_conversation_manager()
    mgr.clear(client_key)
    return jsonify({"ok": True})


@app.route("/api/debug/retrieve")
def debug_retrieve():
    client_key = request.args.get("client_key", "")
    q = request.args.get("q", "")
    retriever = get_retriever()
    results = retriever.search_with_scores(q, client_key, top_k=8)
    return jsonify(
        [
            {
                "score": round(s, 4),
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "doc_type": c.doc_type,
                "doc_date": c.doc_date,
                "superseded_by": c.superseded_by,
                "text": c.text[:300],
            }
            for s, c in results
        ]
    )


@app.route("/api/status")
def status():
    from llm import get_llm

    llm = get_llm()
    return jsonify(
        {
            "llm_available": llm.available,
            "llm_provider": llm.provider_info,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
