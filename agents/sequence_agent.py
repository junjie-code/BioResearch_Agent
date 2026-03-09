from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore
from agent.model_factory import ModelFactory
from agent.context_manager import ContextManager

from tools.sequence_analysis import analyze_sequence  # 复用你原有的函数


SYSTEM_PROMPT = """你是生物信息学序列分析专家。根据用户需求分析DNA/蛋白质序列。
你可以进行：GC含量计算、碱基组成分析、序列翻译、基本统计等。
用专业但清晰的语言呈现分析结果。"""


async def sequence_agent_node(state: dict, *, store: BaseStore) -> dict:
    llm = ModelFactory.get_llm("seq_agent")
    ctx = ContextManager()

    # 读取私有记忆
    namespace = ("memory", "seq_agent")
    try:
        cache = await store.aget(namespace, "seq_cache")
    except Exception:
        cache = None

    agent_context = ctx.build_agent_context(state, "seq_agent")
    user_text = agent_context[-1] if agent_context else ""

    # 让LLM提取序列或基因名
    extract_prompt = f"""从用户输入中提取需要分析的DNA/蛋白质序列。
用户输入：{user_text}
对用户输入序列做分析统计。"""

    extract_response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
    sequence_input = extract_response.content.strip()

    # 检查缓存，避免重复分析
    if cache and sequence_input in cache:
        return {"agent_results": {"sequence": cache[sequence_input]}}

    # 调用你原有的序列分析工具
    try:
        analysis_result = analyze_sequence(sequence_input)
    except Exception as e:
        return {"agent_results": {"sequence": f"序列分析出错：{str(e)}"}}

    # LLM解读结果
    interpret_prompt = f"""{SYSTEM_PROMPT}

用户问题：{user_text}
分析结果：{analysis_result}

请用专业语言解读这些分析结果，给出科研建议。"""

    response = await llm.ainvoke([HumanMessage(content=interpret_prompt)])
    result_text = response.content

    # 写入私有记忆
    updated_cache = (cache or {})
    updated_cache[sequence_input] = result_text
    try:
        await store.aput(namespace, "seq_cache", updated_cache)
    except Exception:
        pass

    return {"agent_results": {"sequence": result_text}}