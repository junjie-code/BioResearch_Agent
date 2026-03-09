from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.supervisor import supervisor_node, route_by_intent
from agents.literature_agent import literature_agent_node
from agents.sequence_agent import sequence_agent_node
from agents.image_agent import image_agent_node
from agents.kb_agent import kb_agent_node


# ===== 全局状态定义 =====
class GlobalState(TypedDict):
    """
    三层记忆架构 - 第1层：全局共享状态
    所有Agent可读 agent_results，各Agent只写自己的 key
    Supervisor 管控 user_intent / task_plan / current_step
    """
    task_id: str                                    # 任务唯一ID
    user_intent: str                                # Supervisor 写入的意图描述
    task_plan: list[str]                            # Agent执行计划 ["lit_agent", "seq_agent"]
    current_step: int                               # 当前执行到第几步
    current_agent: str                              # 当前路由目标
    messages: list                                  # 用户消息列表（含图片base64）
    agent_results: Annotated[dict, operator.ior]    # 各Agent合并写入结果
    final_answer: str                               # 最终汇总回答


def build_graph():
    """构建 Supervisor 多Agent协作图"""
    graph = StateGraph(GlobalState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("lit_agent", literature_agent_node)
    graph.add_node("seq_agent", sequence_agent_node)
    graph.add_node("image_agent", image_agent_node)
    graph.add_node("kb_agent", kb_agent_node)

    # Supervisor 作为入口
    graph.set_entry_point("supervisor")

    # Supervisor → 条件路由
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "lit_agent": "lit_agent",
            "seq_agent": "seq_agent",
            "image_agent": "image_agent",
            "kb_agent": "kb_agent",
            "done": END,
        }
    )

    # 各Agent → 回到Supervisor
    for agent in ["lit_agent", "seq_agent", "image_agent", "kb_agent"]:
        graph.add_edge(agent, "supervisor")

    return graph


def create_app():
    """编译图，注入记忆组件"""
    graph = build_graph()

    # 第2层：Agent私有记忆（Scoped Store）
    store = InMemoryStore()

    # 第3层：跨会话持久化
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

    app = graph.compile(
        checkpointer=checkpointer,
        store=store,
    )
    return app


# 供 app.py 导入
app = create_app()