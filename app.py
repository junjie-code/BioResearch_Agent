"""
Streamlit 主界面
改动点：导入graph，消息格式适配多模态
"""
import streamlit as st
import asyncio
import base64
from agent.graph import app  # 导入改造后的Multi-Agent图

st.set_page_config(page_title="BioResearch Multi-Agent", layout="wide")
st.title(" BioResearch Multi-Agent 生物科研多智能体系统")

# 会话管理
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 图片上传
uploaded_image = st.sidebar.file_uploader(
    "上传科研图像（细胞图/WB/电泳图）",
    type=["png", "jpg", "jpeg", "tif"]
)

# 用户输入
user_input = st.chat_input("请输入您的科研问题...")

if user_input:
    # 构造消息（支持多模态）
    if uploaded_image:
        image_b64 = base64.b64encode(uploaded_image.read()).decode()
        message = {
            "content": [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }
        display_text = f"{user_input}\n\n[已上传图片: {uploaded_image.name}]"
    else:
        message = {"content": user_input}
        display_text = user_input

    # 显示用户消息
    st.session_state.chat_history.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    # 调用 Multi-Agent 图
    with st.chat_message("assistant"):
        with st.spinner("多Agent协作分析中..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            input_state = {
                "messages": [message],
                "agent_results": {},
                "task_plan": [],
                "current_step": 0,
            }

            # 异步调用
            result = asyncio.run(app.ainvoke(input_state, config=config))

            answer = result.get("final_answer", "抱歉，处理过程中出现了问题。")

            # 显示执行过程（可选）
            if result.get("task_plan"):
                plan_str = " → ".join(result["task_plan"])
                st.caption(f"📋 执行路径：{plan_str}")

            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})