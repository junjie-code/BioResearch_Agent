# tools/knowledge_qa.py
"""
知识库问答工具（RAG）

基于 LangChain + Chroma 的检索增强生成，
从本地知识库中检索相关文档片段回答问题。
"""
import os
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import KNOWLEDGE_BASE_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL

# 全局变量，缓存向量数据库实例（避免重复加载）
_vectorstore = None


def _get_vectorstore():
    """获取或初始化向量数据库"""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 如果已有持久化的数据库，直接加载
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("[知识库] 加载已有向量数据库...")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
        return _vectorstore

    # 否则，从文档创建新的向量数据库
    print("[知识库] 首次运行，正在构建向量数据库...")

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)

    # 加载文档
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        print("[知识库] 警告：knowledge_base/docs/ 目录下没有找到文档！")
        return None

    # 文档分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " "],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[知识库] 共处理 {len(documents)} 个文档，分成 {len(chunks)} 个文本块")

    # 创建向量数据库
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print("[知识库] 向量数据库构建完成！")

    return _vectorstore


@tool
def knowledge_qa(query: str) -> str:
    """
    从内部知识库中检索与问题相关的信息。

    当用户询问实验室操作规程(SOP)、设备使用说明、
    内部制度文件等信息时使用此工具。

    Args:
        query: 用户的问题，例如 "PCR实验的退火温度怎么设置"

    Returns:
        从知识库中检索到的相关信息
    """
    vectorstore = _get_vectorstore()

    if vectorstore is None:
        return "知识库为空。请先在 knowledge_base/docs/ 目录中添加文档。"

    try:
        # 相似度搜索，返回最相关的 3 个文本块
        docs = vectorstore.similarity_search_with_score(query, k=3)

        if not docs:
            return f"未在知识库中找到与 '{query}' 相关的信息。"

        results = []
        for i, (doc, score) in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知来源"))
            results.append(
                f"[来源 {i}] {source} (相关度: {1 - score:.2f})\n"
                f"{doc.page_content}\n"
            )

        return f"从知识库中找到 {len(results)} 条相关信息：\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"知识库检索出错: {str(e)}"