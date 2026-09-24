"""核心测试:覆盖两条硬要求——有据可查、客户隔离。

运行: python -m pytest tests/  或  python tests/test_core.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_loader import get_store
from retriever import get_retriever
from qa_engine import QAContext, get_engine


# ---------- 客户隔离 ----------

def test_fenbao_cannot_see_restricted_docs():
    """分包不可见的文档(15报价/24机电/28合同/32付款)绝不能出现在其可见集合中。"""
    store = get_store()
    fenbao_visible = {d.id for d in store.visible_documents("fenbao_B")}
    restricted = {"15", "24", "28", "32"}
    assert restricted.isdisjoint(fenbao_visible), (
        f"分包不应看到受限文档,但看到了: {restricted & fenbao_visible}"
    )


def test_zongbao_sees_all():
    """总包可见全部 32 份文档。"""
    store = get_store()
    assert len(store.visible_documents("zongbao_A")) == len(store.documents) == 32


def test_retriever_respects_isolation():
    """检索层对分包返回的 chunk 不得包含受限文档。"""
    retriever = get_retriever()
    chunks = retriever.search("钢筋 报价 单价 合同", "fenbao_B", top_k=20)
    restricted = {"15", "24", "28", "32"}
    hit = {c.doc_id for c in chunks} & restricted
    assert not hit, f"检索结果泄露了受限文档: {hit}"


def test_quotation_not_leaked_to_fenbao():
    """分包询问钢筋报价时,回答中不得出现真实报价数字(4180/4150/4120)。"""
    engine = get_engine()
    r = engine.answer(QAContext(query="钢筋报价单的单价是多少?", client_key="fenbao_B"))
    answer = r["answer"]
    for price in ["4180", "4150", "4120"]:
        assert price not in answer, f"分包回答泄露了报价 {price}"
    # 也不应出现 doc28 的合同单价 268 / 36
    for price in ["268", "36 元"]:
        assert price not in answer, f"分包回答泄露了合同单价 {price}"


def test_unknown_client_gets_nothing():
    """未知客户身份不应返回任何文档。"""
    store = get_store()
    assert store.visible_documents("nonexistent") == []


# ---------- 有据可查 ----------

def test_zongbao_quotation_has_price():
    """总包询问钢筋报价时,应能从 doc15 得到价格(降级模式下也应检索到)。"""
    engine = get_engine()
    r = engine.answer(QAContext(query="钢筋报价单的单价是多少?", client_key="zongbao_A"))
    # 降级或 LLM 模式下,来源中应包含 doc15
    src_joined = " ".join(r["sources"])
    assert "15" in src_joined or "供应商报价单" in src_joined, (
        f"总包问报价应检索到 doc15,实际来源: {r['sources']}"
    )


def test_pour_date_answer_cites_source():
    """询问浇筑日期,回答应包含日期 2026-05-12 且标注来源。"""
    engine = get_engine()
    r = engine.answer(
        QAContext(query="3#楼5层顶板哪天浇筑的混凝土?", client_key="zongbao_A")
    )
    assert "2026-05-12" in r["answer"] or "5-12" in r["answer"], (
        f"回答应包含浇筑日期 2026-05-12,实际: {r['answer'][:200]}"
    )
    assert r["sources"], "回答应标注来源"


def test_insufficient_info_for_unknown():
    """询问资料中完全没有的内容,应回答'资料不足'。"""
    engine = get_engine()
    r = engine.answer(
        QAContext(query="项目的精装修造价是多少?", client_key="zongbao_A")
    )
    assert "资料不足" in r["answer"], (
        f"资料中无精装修信息,应回答资料不足,实际: {r['answer'][:200]}"
    )


def test_supersession_detected():
    """进度计划 v1(02)应被标记为由 v2(03)替代。"""
    store = get_store()
    v1 = store.get_document("02")
    assert v1.superseded_by == "03", "v1 进度计划应被 v2 替代"


def test_superseded_doc_ranked_lower():
    """已作废的 v1 进度计划在检索中排序应低于 v2。"""
    retriever = get_retriever()
    results = retriever.search_with_scores(
        "主体结构进度计划 封顶日期", "zongbao_A", top_k=10
    )
    scores = {c.doc_id: s for s, c in results}
    if "02" in scores and "03" in scores:
        assert scores["03"] > scores["02"], "v2 应排在作废的 v1 之前"


# ---------- 对话与追问 ----------

def test_followup_retrieves_removal_spec():
    """追问'什么时候可以拆模'应检索到拆模规范(doc29)。"""
    engine = get_engine()
    # 先问浇筑日期,建立上下文
    engine.answer(QAContext(query="5层顶板哪天浇筑的?", client_key="zongbao_A"))
    r = engine.answer(
        QAContext(
            query="那什么时候可以拆模?",
            client_key="zongbao_A",
            history=[{"role": "user", "content": "5层顶板哪天浇筑的?"}],
        )
    )
    src = " ".join(r["sources"])
    assert "29" in src or "拆模" in src, (
        f"追问拆模应检索到 doc29,实际来源: {r['sources']}"
    )


if __name__ == "__main__":
    import traceback

    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__} -> {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
