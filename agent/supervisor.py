"""
Supervisor：任务编排中枢
职责：意图分类 → 任务拆解 → Agent路由 → 结果汇总
"""
import json
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage
from agent.model_factory import ModelFactory
from agent.prompt_guard import PromptGuard
from agent.context_manager import ContextManager


# ===== 意图分类 Prompt =====
INTENT_PROMPT = """你是一个生物科研任务调度器。分析用户输入，判断需要调用哪些Agent。

## 可用Agent
| Agent名 | 能力 | 适用场景 |
|---------|------|---------|
| lit_agent | PubMed文献检索 | 用户问"最新研究""有哪些文献""综述""paper" |
| seq_agent | DNA/蛋白质序列分析 | 用户问"序列分析""GC含量""碱基组成""翻译""FASTA" |
| image_agent | 细胞核图像检测与分析 | 用户上传了图片（细胞图/显微镜图) |


## 规则
1. 选择1~3个最相关的Agent
2. 如果用户没上传图片，不要选 image_agent
3. 有先后依赖的按顺序排列（如：先查序列再查文献）
4. 如果只是闲聊或无法判断，拒绝回答。

## 严格输出JSON（不要输出任何其他内容）
{{"intent": "一句话描述用户意图", "plan": ["agent1", "agent2"]}}

## 用户输入
{user_input}

## 是否上传了图片
{has_image}"""


async def supervisor_node(state: dict) -> dict:
    """
    Supervisor 核心逻辑，会被 LangGraph 多次调用：
    - 第1次：意图分类 + 生成执行计划
    - 中间：按计划派发下一个Agent
    - 最后：汇总所有结果
    """
    llm = ModelFactory.get_llm("supervisor")

    # ====== 阶段3：所有Agent完成 → 汇总 ======
    if state.get("task_plan") and state.get("current_step", 0) >= len(state["task_plan"]):
        return await _summarize_results(state, llm)

    # ====== 阶段1：首次进入 → 意图分类 + 拆解 ======
    if not state.get("task_plan"):
        return await _classify_and_plan(state, llm)

    # ====== 阶段2：中间步骤 → 派发下一个Agent ======
    step = state["current_step"]
    next_agent = state["task_plan"][step]
    return {
        "current_agent": next_agent,
        "current_step": step + 1,
    }


async def _classify_and_plan(state: dict, llm) -> dict:
    """意图分类 + 任务拆解"""
    user_msg = _get_last_user_text(state["messages"])

    # 提示词注入检查
    cleaned, is_injection = PromptGuard.sanitize(user_msg)
    if is_injection:
        return {
            "final_answer": "检测到异常输入，请重新描述您的科研问题。",
            "current_agent": "done",
        }

    has_image = "是" if _has_image(state["messages"]) else "否"

    prompt = INTENT_PROMPT.format(user_input=cleaned, has_image=has_image)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    result = _robust_json_parse(response.content)
    plan = result.get("plan", ["kb_agent"])

    # 校验plan合法性
    valid_agents = {"lit_agent", "seq_agent", "image_agent", "kb_agent"}
    plan = [a for a in plan if a in valid_agents]
    if not plan:
        plan = ["kb_agent"]

    return {
        "task_id": str(uuid.uuid4())[:8],
        "user_intent": result.get("intent", ""),
        "task_plan": plan,
        "current_step": 0,
        "current_agent": plan[0],
    }


async def _summarize_results(state: dict, llm) -> dict:
    """汇总所有Agent结果"""
    ctx = ContextManager()
    results_summary = ctx.build_supervisor_summary(state.get("agent_results", {}))

    prompt = f"""请根据各专业Agent的分析结果，为用户生成完整回答。

用户问题：{_get_last_user_text(state['messages'])}

各Agent分析结果：
{results_summary}

要求：整合所有信息，结构清晰，专业准确。如有多个Agent结果，请指出它们之间的关联。"""

    summarizer = ModelFactory.get_llm("summarizer")
    response = await summarizer.ainvoke([HumanMessage(content=prompt)])

    return {
        "final_answer": response.content,
        "current_agent": "done",
    }


def route_by_intent(state: dict) -> str:
    """LangGraph 条件路由函数"""
    return state.get("current_agent", "done")


# ===== 工具函数 =====
def _get_last_user_text(messages: list) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    return str(last)


def _has_image(messages: list) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content", [])
        if isinstance(content, list):
            return any(c.get("type") in ("image_url", "image") for c in content)
    return False


def _robust_json_parse(text: str) -> dict:
    """多层兜底JSON解析"""
    import re
    # 第1层：直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # 第2层：去markdown标记
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, TypeError):
        pass
    # 第3层：正则提取
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    # 兜底
    return {"intent": "unknown", "plan": ["kb_agent"]}