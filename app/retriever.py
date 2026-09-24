"""检索模块:文档分块 + 基于关键词的 TF-IDF 检索。

设计取舍(见 DECISIONS.md):
- 不用向量数据库/embedding API,避免额外依赖与网络;采用 jieba 分词 + TF-IDF,
  对中文短文档效果足够,且可离线运行。
- 分块策略:按 Markdown 标题与段落切分,每个 chunk 保留所属文档 id 与标题,
  便于回答时精确标注来源。
- 对检索结果做去重与时效标记(被替代的旧版计划降权)。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import jieba

from doc_loader import Document, get_store

# 停用词(常见无意义词)
_STOPWORDS = set(
    "的 了 和 是 在 有 与 及 或 等 为 对 将 由 按 本 该 此 其 之 于 以 并 也 都 就 还 要 可 不 没 无 中 内 外 上 下 前 后 时 日 月 年".split()
)

# 领域同义词扩展:把查询中的关键词扩展为文档中可能出现的同义表述
_SYNONYMS = {
    "拆模": ["拆除", "底模", "拆模", "模板拆除", "支模拆除", "试块", "强度", "养护", "抗压"],
    "浇筑": ["浇注", "混凝土浇筑", "浇捣"],
    "顶板": ["楼板", "屋面板"],
    "强度": ["抗压强度", "mpa", "兆帕"],
    "隐蔽": ["隐蔽工程", "隐蔽验收"],
    "试块": ["试件", "同条件", "标准养护"],
    "钢筋": ["hrb400", "纵筋", "箍筋"],
    "临边": ["临边防护", "洞口防护"],
    "付款": ["进度款", "工程款", "付款"],
    "报价": ["单价", "价格", "报价单"],
}


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    doc_type: str
    doc_date: str
    chunk_index: int
    text: str
    superseded_by: Optional[str] = None


def _tokenize(text: str) -> List[str]:
    tokens = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w or w in _STOPWORDS:
            continue
        # 过滤纯标点/单字符英文,保留中文与数字
        if re.fullmatch(r"[\W_]+", w):
            continue
        tokens.append(w.lower())
    return tokens


def _split_into_chunks(doc: Document) -> List[Chunk]:
    """按 Markdown 标题与段落切分,短文档整体作为一个 chunk。"""
    lines = doc.content.splitlines()
    chunks: List[Chunk] = []
    current: List[str] = []
    current_heading = ""

    def flush():
        text = "\n".join(current).strip()
        if text:
            chunks.append(
                Chunk(
                    doc_id=doc.id,
                    doc_title=doc.title,
                    doc_type=doc.type,
                    doc_date=doc.date,
                    chunk_index=len(chunks),
                    text=(current_heading + "\n" + text) if current_heading else text,
                    superseded_by=doc.superseded_by,
                )
            )
        current.clear()

    for line in lines:
        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip()
        else:
            current.append(line)
    flush()

    # 若切分后没有 chunk(空文档),至少保留一个
    if not chunks:
        chunks.append(
            Chunk(
                doc_id=doc.id,
                doc_title=doc.title,
                doc_type=doc.type,
                doc_date=doc.date,
                chunk_index=0,
                text=doc.content,
                superseded_by=doc.superseded_by,
            )
        )
    return chunks


def _expand_with_synonyms(tokens: List[str]) -> List[str]:
    """用领域同义词扩展查询 token。"""
    expanded = list(tokens)
    for t in tokens:
        for syn in _SYNONYMS.get(t, []):
            expanded.append(syn)
    return expanded


class Retriever:
    def __init__(self):
        self.store = get_store()
        self.chunks: List[Chunk] = []
        self.doc_freq: Counter = Counter()
        self.num_docs = 0
        self._avg_dl = 0.0
        self._build_index()

    def _build_index(self) -> None:
        self.chunks = []
        for doc in self.store.documents.values():
            self.chunks.extend(_split_into_chunks(doc))
        # 以 chunk 为文档单位计算 IDF
        self.num_docs = len(self.chunks)
        total_len = 0
        for ch in self.chunks:
            toks = _tokenize(ch.text)
            total_len += len(toks)
            for t in set(toks):
                self.doc_freq[t] += 1
        self._avg_dl = total_len / max(self.num_docs, 1)

    def _bm25_score(self, query_tokens: List[str], chunk: Chunk) -> float:
        """BM25 打分,对短查询更稳健;并对精确短语匹配加权。"""
        chunk_tokens = _tokenize(chunk.text)
        n = len(chunk_tokens)
        if n == 0:
            return 0.0
        tf = Counter(chunk_tokens)
        k1, b = 1.5, 0.75
        avgdl = self._avg_dl or 1.0
        score = 0.0
        q_set = set(query_tokens)
        for q in q_set:
            if q in tf:
                df = self.doc_freq.get(q, 0)
                idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                f = tf[q]
                denom = f + k1 * (1 - b + b * n / avgdl)
                score += idf * (f * (k1 + 1)) / denom
        # 精确短语加权:把 query 中连续 2 个 token 在 chunk 中出现的次数作为加分
        bigrams = {
            query_tokens[i] + query_tokens[i + 1]
            for i in range(len(query_tokens) - 1)
        }
        chunk_bigrams = [
            chunk_tokens[i] + chunk_tokens[i + 1] for i in range(n - 1)
        ]
        phrase_hits = sum(1 for bg in chunk_bigrams if bg in bigrams)
        score += phrase_hits * 0.8
        # 时效处理:被替代的旧版文档降权
        if chunk.superseded_by:
            score *= 0.3
        return score

    def search(self, query: str, client_key: str, top_k: int = 6) -> List[Chunk]:
        """按客户权限过滤后检索相关 chunk。

        注意:检索前先按 visible_to 过滤 chunk —— 这是客户隔离的第二道防线,
        确保无权文档的内容绝不会进入 LLM 上下文。
        """
        visible_ids = {d.id for d in self.store.visible_documents(client_key)}
        candidate_chunks = [c for c in self.chunks if c.doc_id in visible_ids]

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidate_chunks[:top_k]
        query_tokens = _expand_with_synonyms(query_tokens)

        scored = []
        for ch in candidate_chunks:
            s = self._bm25_score(query_tokens, ch)
            if s > 0:
                scored.append((s, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:top_k]]

    def search_with_scores(
        self, query: str, client_key: str, top_k: int = 6
    ) -> List[tuple]:
        """返回 (score, chunk) 列表,供调试与展示。"""
        visible_ids = {d.id for d in self.store.visible_documents(client_key)}
        candidate_chunks = [c for c in self.chunks if c.doc_id in visible_ids]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [(0.0, c) for c in candidate_chunks[:top_k]]
        query_tokens = _expand_with_synonyms(query_tokens)
        scored = []
        for ch in candidate_chunks:
            s = self._bm25_score(query_tokens, ch)
            if s > 0:
                scored.append((s, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
