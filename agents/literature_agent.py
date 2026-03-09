from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from agent.model_factory import ModelFactory
from agent.context_manager import ContextManager

from tools.pubmed_search import search_pubmed  


SYSTEM_PROMPT = """你是生物医学文献检索专家。根据用户的研究问题，
使用PubMed检索相关文献，并整理关键发现。

要求：
1. 提取用户问题中的关键词，构造合适的检索策略
2. 对检索结果按相关性排序
3. 提取每篇文献的核心发现
4. 如果有前序Agent的结果，结合这些信息优化检索"""


async def literature_agent_node(state: dict, *, store: BaseStore) -> dict:
    llm = ModelFactory.get_llm("lit_agent")
    ctx = ContextManager()

    # === 读取私有记忆 ===
    namespace = ("memory", "lit_agent")
    try:
        search_history = await store.aget(namespace, "search_history")
    except Exception:
        search_history = None

    # === 构建Agent专属上下文（不含其他Agent的历史） ===
    agent_context = ctx.build_agent_context(state, "lit_agent")

    # === 如果有前序Agent结果，注入上下文 ===
    prior_info = ""
    if state.get("agent_results"):
        prior_info = ctx.build_prior_results_summary(state["agent_results"])

    # === 让LLM提取检索关键词 ===
    user_text = agent_context[-1] if agent_context else ""
    keyword_prompt = f"""从以下科研问题中提取PubMed检索关键词（英文，用空格分隔）：
问题：{user_text}
{f'参考信息：{prior_info}' if prior_info else ''}
只返回关键词，不要其他内容。"""

    keyword_response = await llm.ainvoke([HumanMessage(content=keyword_prompt)])
    query = keyword_response.content.strip()

    # === 调用 pubmed_search 工具 ===
    try:
        papers = search_pubmed(query)  
    except Exception as e:
        return {"agent_results": {"literature": f"文献检索出错：{str(e)}"}}

    # === LLM整理结果 ===
    summary_prompt = f"""{SYSTEM_PROMPT}

用户问题：{user_text}
检索关键词：{query}
检索结果：{papers}

请整理为结构化的文献摘要。"""

    response = await llm.ainvoke([HumanMessage(content=summary_prompt)])

    # === 写入私有记忆 ===
    from datetime import datetime
    record = {"query": query, "time": datetime.now().isoformat(), "count": len(papers) if isinstance(papers, list) else 0}
    updated_history = (search_history or []) + [record]
    try:
        await store.aput(namespace, "search_history", updated_history[-20:])
    except Exception:
        pass

    return {"agent_results": {"literature": response.content}}