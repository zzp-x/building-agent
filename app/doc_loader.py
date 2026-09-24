"""文档加载与客户权限隔离模块。

职责:
- 读取 access.json,解析客户身份与每份文档的 visible_to
- 读取 docs/ 下所有 md/txt/csv 文档的原文
- 按客户身份过滤可见文档(硬要求:绝不能让某客户看到其无权查看的文档)
- 解析进度计划的版本作废关系(用于时效处理)
"""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Q1-语料")
ACCESS_FILE = os.path.join(CORPUS_DIR, "access.json")
DOCS_DIR = os.path.join(CORPUS_DIR, "docs")


@dataclass
class Document:
    id: str
    file: str
    type: str
    date: str
    visible_to: List[str]
    title: str
    content: str
    # 时效处理:若该文档被更新版本替代,记录替代它的文档 id
    superseded_by: Optional[str] = None

    def display_name(self) -> str:
        return f"{self.id}-{self.title}"


@dataclass
class Client:
    key: str
    name: str
    role: str
    description: str


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_csv_as_text(path: str) -> str:
    """把 CSV 读成可读的文本表格,便于检索与展示。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    lines = []
    for row in rows:
        # 清理单元格空白
        cells = [c.strip() for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_title(filename: str) -> str:
    """从文件名去掉序号前缀和扩展名,得到标题。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    # 形如 "01-项目概况" -> "项目概况"
    if "-" in base and base[:base.index("-")].isdigit():
        return base[base.index("-") + 1 :]
    return base


def _detect_supersession(docs: Dict[str, Document]) -> None:
    """识别进度计划的版本替代关系。

    规则:若文档正文中出现"本版替代 ... 第 N 版"或"第 N 版作废",
    且同类型文档中存在更早版本,则标记旧版被新版替代。
    此处采用稳健策略:进度计划文档按日期排序,较晚的版本替代较早的版本,
    且需要正文里显式声明作废(避免误判)。
    """
    plan_docs = [d for d in docs.values() if d.type == "进度计划"]
    plan_docs.sort(key=lambda d: d.date)
    for i, cur in enumerate(plan_docs):
        # 检查正文是否声明替代之前版本
        declares_supersede = any(
            kw in cur.content
            for kw in ["替代", "作废", "第 1 版", "第1版"]
        )
        if declares_supersede:
            # 替代所有更早的进度计划
            for prev in plan_docs[:i]:
                if prev.superseded_by is None:
                    prev.superseded_by = cur.id


class DocumentStore:
    def __init__(self, corpus_dir: str = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.access_file = os.path.join(corpus_dir, "access.json")
        self.docs_dir = os.path.join(corpus_dir, "docs")
        self.clients: Dict[str, Client] = {}
        self.documents: Dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        with open(self.access_file, "r", encoding="utf-8") as f:
            access = json.load(f)

        for key, info in access["clients"].items():
            self.clients[key] = Client(
                key=key,
                name=info["name"],
                role=info["role"],
                description=info.get("description", ""),
            )

        for meta in access["documents"]:
            doc_id = meta["id"]
            rel_path = meta["file"]
            abs_path = os.path.join(self.corpus_dir, rel_path)
            ext = os.path.splitext(abs_path)[1].lower()
            if ext == ".csv":
                content = _read_csv_as_text(abs_path)
            else:
                content = _read_text(abs_path)
            self.documents[doc_id] = Document(
                id=doc_id,
                file=rel_path,
                type=meta["type"],
                date=meta["date"],
                visible_to=list(meta["visible_to"]),
                title=_extract_title(rel_path),
                content=content,
            )

        _detect_supersession(self.documents)

    def visible_documents(self, client_key: str) -> List[Document]:
        """返回某客户可见的全部文档(权限过滤的唯一入口)。"""
        if client_key not in self.clients:
            return []
        return [d for d in self.documents.values() if client_key in d.visible_to]

    def get_client(self, client_key: str) -> Optional[Client]:
        return self.clients.get(client_key)

    def all_clients(self) -> List[Client]:
        return list(self.clients.values())

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)


# 全局单例
_store: Optional[DocumentStore] = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
