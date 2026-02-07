# agent/graph.py
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from config.settings import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    MAX_AGENT_STEPS
)


def create_agent(tools: list, enable_memory: bool = False):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = state["messages"]
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        full_messages = [system_message] + messages
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    def tool_node_with_count(state: AgentState):
        """工具节点：执行工具并增加计数"""
        tool_executor = ToolNode(tools)
        result = tool_executor.invoke(state)
        # 更新计数
        current_count = state.get("tool_call_count", 0)
        result["tool_call_count"] = current_count + 1
        return result

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        current_count = state.get("tool_call_count", 0)

        # 超过最大步数，强制结束
        if current_count >= MAX_AGENT_STEPS:
            return END

        # LLM 想调用工具 → 继续
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # 否则结束
        return END

    # 构建图
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_with_count)  # 用带计数的版本
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    if enable_memory:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()

    return app