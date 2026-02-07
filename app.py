# app.py
"""
BioResearch Agent - Streamlit Web 界面

启动方式: streamlit run app.py
"""
import streamlit as st
import uuid

# 页面配置（必须是第一个 Streamlit 命令）
st.set_page_config(
    page_title="🧬 BioResearch Agent",
    page_icon="🧬",
    layout="wide",
)

from agent.graph import create_agent
from tools import ALL_TOOLS


# === 初始化 ===
@st.cache_resource
def init_agent():
    """初始化 Agent（只执行一次，缓存结果）"""
    return create_agent(ALL_TOOLS, enable_memory=True)


agent_app = init_agent()

# === 侧边栏 ===
with st.sidebar:
    st.title("🧬 BioResearch Agent")
    st.markdown("### 生物科研智能助手")
    st.markdown("---")

    st.markdown("#### 💡 你可以尝试：")
    example_queries = [
        "搜索 CRISPR 基因编辑最新研究",
        "分析DNA序列 ATCGATCGAATTCCGG",
        "查询PCR实验操作步骤",
        "搜索CAR-T疗法文献并生成报告",
    ]
    for q in example_queries:
        if st.button(q, key=f"example_{q[:10]}"):
            st.session_state["example_input"] = q

    st.markdown("---")
    st.markdown("#### 🛠️ 可用工具")
    st.markdown("""
    - 📚 PubMed 文献检索
    - 🧬 DNA/蛋白质序列分析
    - 🔬 细胞核图像分析
    - 📖 知识库问答
    - 📝 报告生成
    """)

    st.markdown("---")
    if st.button("🗑️ 清除对话历史"):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown(
        "Made by [junjie](https://github.com/junjie-code) | "
        "Powered by DeepSeek + LangGraph"
    )

# === 初始化会话状态 ===
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# === 主界面 ===
st.title("🧬 BioResearch Agent")
st.caption("基于 DeepSeek + LangGraph 的生物科研智能助手")

# 显示对话历史
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === 处理示例输入 ===
if "example_input" in st.session_state:
    user_input = st.session_state.pop("example_input")
else:
    user_input = None

# === 聊天输入 ===
prompt = st.chat_input("请输入你的问题...")

# 合并输入（来自示例按钮或手动输入）
if user_input:
    prompt = user_input

if prompt:
    # 显示用户消息
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent 处理
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考..."):
            # 创建一个状态容器显示中间步骤
            status_container = st.empty()
            step_log = []

            try:
                config = {
                    "configurable": {"thread_id": st.session_state["thread_id"]},
                    "recursion_limit": 20,
                }

                # 使用 stream 模式获取中间步骤
                final_response = ""
                for event in agent_app.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config,
                    stream_mode="values",
                ):
                    last_msg = event["messages"][-1]

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            step_log.append(f"🔧 调用工具: **{tc['name']}**")
                            status_container.markdown(
                                "**执行步骤：**\n" + "\n".join(f"- {s}" for s in step_log)
                            )

                    elif hasattr(last_msg, "type"):
                        if last_msg.type == "tool":
                            step_log.append(f"✅ 工具返回结果")
                            status_container.markdown(
                                "**执行步骤：**\n" + "\n".join(f"- {s}" for s in step_log)
                            )
                        elif last_msg.type == "ai" and last_msg.content and not (
                            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                        ):
                            final_response = last_msg.content

                # 清除状态显示，展示最终结果
                status_container.empty()

                if step_log:
                    with st.expander("📋 查看 Agent 执行步骤", expanded=False):
                        for s in step_log:
                            st.markdown(f"- {s}")

                st.markdown(final_response)

            except Exception as e:
                final_response = f"❌ 处理出错: {str(e)}\n\n请尝试重新提问或简化问题。"
                st.error(final_response)

    # 保存 Agent 回复
    st.session_state["messages"].append({"role": "assistant", "content": final_response})