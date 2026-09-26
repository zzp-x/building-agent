# LangGraph 重构计划：QA 编排层迁移

## Context

当前项目是纯手写编排（Flask + requests 调 LLM + 手写检索流程）。用户要求迁移到 LangGraph 编排，且已确认两个决策：
1. **LLM 调用层同步迁移**到 LangChain 模型类（ChatOpenAI / ChatAnthropic），保留现有 4 级探测逻辑（.env LLM_* → ANTHROPIC_* → TRAE settings.json）
2. **增强版图形态**：带"资料不足 → 查询改写 → 重检索一次"的自我修正循环（RAG Self-corrective）

硬约束：
- 公共 API 不变：`QAContext` / `QAEngine.answer` / `QAEngine.answer_stream` 签名不变，SSE 事件协议（sources→meta→delta→done）不变 → **app.py、前端、conversation.py、retriever.py、doc_loader.py、tests 均不改**
- 无 LLM 环境（降级模式）下 `tests/test_core.py` 12 个测试必须全 PASS，降级行为与现在等价
- 顺带修复 [qa_engine.py:280](e:\Py-Code\面试文件\building-agent\app\qa_engine.py) 的 `visible_ids` NameError（answer() 来源补标分支）
- README 后续单独更新（不在本次范围，末尾列出待改点）

## 文件改动清单

### 1. 新增 `app/qa_graph.py`（核心）
- `QAState(TypedDict)`：`query, effective_query, client_key, history, chunks, messages, answer, sources, inferences, used_llm, llm_failed, rewrite_used`
- `build_graph(store, retriever, llm) -> CompiledStateGraph`：节点用闭包持有依赖，节点函数 `(state) -> dict` 返回增量

**图拓扑：**
```
START → retrieve
retrieve →(条件 route_after_retrieve):
    chunks 非空              → build_prompt
    chunks 空且 rewrite_used=0 且 llm.available → rewrite_query
    否则                     → insufficient(END)
rewrite_query → retrieve（置 rewrite_used=1，只循环一次；LLM 改写失败时保留原 query，二检仍空自然走 insufficient）
build_prompt → generate
generate →(条件): llm_failed → fallback(END)；否则 → postprocess(END)
```

节点复用 qa_engine.py 现有函数（qa_graph 顶层 import）：
- `retrieve`：调用从 qa_engine 抽出的模块级 `retrieve_chunks(store, retriever, ctx)`（原 `_retrieve_chunks` 主体，改为接受 `effective_query`；追问扩展仍用原始 query+history）
- `rewrite_query`：LLM 把当前问题+上一问改写为独立检索式；`try/except LLMError` → 失败置 rewrite_used=1 并保留原 query
- `build_prompt`：`_build_context_block` + SYSTEM_PROMPT + client_note + `history[-8:]` → state.messages
- `generate`：`llm.invoke_text(messages)`；**只捕 LLMError** → `{"llm_failed": True, "used_llm": False}`（真实 bug 照常抛出）
- `postprocess`：`_post_process` + 来源补标（节点内重新算 `visible_ids = {d.id for d in store.visible_documents(client_key)}`，修复 NameError）
- `fallback`：复用 `_fallback_answer`
- `insufficient`：固定"资料不足"回答

### 2. 改 `app/llm.py`
- `_detect_provider()` 探测逻辑**原样保留**，`LLMClient.__init__` 按 kind 构造模型：
  - openai → `ChatOpenAI(base_url=..., api_key=..., model=..., temperature=0.2, max_tokens=900, timeout=60, max_retries=0)`
  - anthropic → `ChatAnthropic(base_url=..., api_key=..., model_name=...)`
  - `max_retries=0`：避免 SDK 默认重试拖慢降级
- 新增 `invoke_text(messages: List[dict]) -> str` 和 `stream_text(messages) -> Iterator[str]`：内部显式转 SystemMessage/HumanMessage/AIMessage；**边界统一 `except Exception → raise LLMError`**（含流式迭代中途的网络异常），对外契约与现在完全一致
- 保留 `available` / `provider_info`（/api/status 用）；`get_llm()` 不变

### 3. 改 `app/qa_engine.py`（facade 化）
- `_retrieve_chunks` 主体抽为模块级 `retrieve_chunks(store, retriever, ctx)`；`QAEngine._retrieve_chunks` 保留薄包装
- `QAEngine.__init__` 内**延迟 import** `build_graph`（qa_graph 顶层 import qa_engine 的 SYSTEM_PROMPT/_build_context_block/_fallback_answer/_post_process，避免循环 import）
- `answer()`：`graph.invoke(state)` → 映射为 `{answer, sources, used_llm, inferences}`（键不变）
- `answer_stream()`：`graph.stream(state, stream_mode=["messages", "updates"], config={"recursion_limit": 12})`，映射伪代码：
  - `mode=="messages"`：`(chunk, meta)`，仅当 `meta["langgraph_node"]=="generate"` 且 chunk.content 非空时：首次先 yield `{"type":"meta","used_llm":True}`，再 yield `{"type":"delta","text":chunk.content}`（rewrite_query 节点的 token 被 langgraph_node 过滤掉）
  - `mode=="updates"`：payload 形如 `{节点名: {变更键: 值}}`
    - `build_prompt` → yield `{"type":"sources","sources":[...]}`（取 chunks 的 display_name 去重）
    - `postprocess` → yield `{"type":"done","sources":...,"used_llm":True}`
    - `fallback` → yield meta(False) → 逐行 delta（模拟流式）→ done(False)（与现有 344-351 行双 meta 行为一致）
    - `insufficient` → sources([]) → meta(False) → delta → done（与现有 297-305 行一致）

### 4. 改 `requirements.txt`
```
langgraph>=0.2.60
langchain-core>=0.3
langchain-openai>=0.2
langchain-anthropic>=0.2
```
安装后按实测通过版本补上界（如 `<0.7` / `<1.0`）。注意：langgraph/langchain-core 要求 **Python ≥3.9**（README 的 3.8+ 需后续改为 3.9+）。

## 验证步骤

1. **降级等价**：临时清空 `.env` 中 key（或 unset LLM_* ANTHROPIC_*）→ `python -m pytest app/tests/test_core.py`（在 app/ 下）→ 12 个全 PASS
2. **有 key 非流式**：启动 `python app/app.py`，curl POST `/api/chat` 问"5层顶板哪天浇筑的"→ 回答含日期 + 来源；问"精装修造价"→ 资料不足
3. **有 key 流式**：curl -N POST `/api/chat/stream` → 事件顺序 sources→meta(True)→delta...→done；把 key 改错重启 → 中途 meta(False) + fallback 逐行输出
4. **增强环**：问一个首查为空的问题（有 key 时）→ 服务端日志确认 rewrite_query → 二次 retrieve 路径只走一轮
5. 恢复 `.env` 原 key

## 风险点
- `stream_mode="messages"` 的 token 传播依赖 langgraph 版本实现，若丢 token 需在节点内显式传 `config` 给模型调用（实现时实测）
- SSE 中途 fallback 时前端已收到部分 delta——与现有行为一致，不算回归
- langgraph 0.6+/1.0 可能有 API 变动，requirements 上界必须锁

## README 待更新点（本次不改，仅记录）
依赖清单、Python 3.9+、架构章节（手写编排 → LangGraph StateGraph）、自我修正 RAG 流程说明、降级行为、FAQ（pydantic v2 冲突）
