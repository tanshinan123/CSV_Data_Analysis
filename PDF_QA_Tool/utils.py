from __future__ import annotations

import os
import tempfile
from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _is_dashscope_compatible_base(base_url: str) -> bool:
    return bool(base_url and "dashscope.aliyuncs.com" in base_url)


def _sanitize_docs_for_embedding(docs: List[Document]) -> List[Document]:
    """百炼等接口要求嵌入输入为合法 str，且不宜传空串。"""
    out: List[Document] = []
    for d in docs:
        raw = d.page_content
        text = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
        text = text.strip()
        if not text:
            continue
        out.append(Document(page_content=text, metadata=d.metadata))
    return out


def _make_embeddings(
    *,
    api_key: str,
    base_url: str,
    embedding_model: str,
    embedding_dimensions: int | None,
):
    """百炼兼容 OpenAI 的 /embeddings 不接受 tiktoken 传来的 token 整型列表，需走 DashScope 原生嵌入。"""
    if _is_dashscope_compatible_base(base_url):
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as e:
            raise ImportError(
                "使用阿里云百炼向量模型需要安装 dashscope：pip install dashscope"
            ) from e
        return DashScopeEmbeddings(
            model=embedding_model,
            dashscope_api_key=api_key,
        )

    emb_kw: dict = {
        "model": embedding_model,
        "openai_api_key": api_key,
        "base_url": base_url,
    }
    if embedding_dimensions is not None:
        emb_kw["dimensions"] = embedding_dimensions
    return OpenAIEmbeddings(**emb_kw)


class _MergedDocumentsRetriever(BaseRetriever):
    """无向量 API 时，将分块合并为单段上下文（超长则截断）。"""

    documents: List[Document] = Field(default_factory=list)
    max_chars: int = 120000

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        merged = "\n\n".join(d.page_content for d in self.documents)
        if len(merged) > self.max_chars:
            merged = (
                merged[: self.max_chars]
                + "\n\n[内容已截断；超长 PDF 建议使用「阿里云百炼」以启用向量检索。]"
            )
        return [Document(page_content=merged)]


def _build_retriever(
    texts: List[Document],
    *,
    api_key: str,
    base_url: str,
    embedding_model: str | None,
    embedding_dimensions: int | None,
):
    if embedding_model:
        texts = _sanitize_docs_for_embedding(texts)
        if not texts:
            raise ValueError("分段后无有效文本可嵌入，请检查 PDF 是否含可复制文字。")
        embeddings_model = _make_embeddings(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
        vector_database = FAISS.from_documents(texts, embeddings_model)
        return vector_database.as_retriever()

    try:
        from langchain_community.retrievers import BM25Retriever

        k = min(8, max(1, len(texts)))
        return BM25Retriever.from_documents(texts, k=k)
    except ImportError:
        return _MergedDocumentsRetriever(documents=texts)


# 定义智能体函数，参数含大模型，记忆体，知识库（上传的文件）及用户会话
def qa_agent(
    openai_api_key,
    memory,
    uploaded_file,
    question,
    *,
    base_url: str,
    chat_model: str,
    embedding_model: str | None,
    embedding_dimensions: int | None,
):
    model = ChatOpenAI(
        model=chat_model,
        openai_api_key=openai_api_key,
        base_url=base_url,
    )

    uploaded_file.seek(0)
    file_content = uploaded_file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "、", "！", "？", ""],
    )
    texts = text_splitter.split_documents(docs)
    if not texts:
        raise ValueError("未能从 PDF 中解析出文本，请确认文件可读且非纯扫描件（或需 OCR）。")

    retriever = _build_retriever(
        texts,
        api_key=openai_api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )

    talk = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
    )

    response = talk.invoke({"chat_history": memory, "question": question})
    return response
