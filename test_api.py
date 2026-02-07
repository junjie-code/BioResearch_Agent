# test_api.py（临时测试文件，测完可删）
from langchain_openai import ChatOpenAI
from config.settings import *

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# 测试基本对话
response = llm.invoke("你是谁？请简单介绍自己。")
print("=== 基本对话测试 ===")
print(response.content)

# 测试 function calling（关键！）
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b

llm_with_tools = llm.bind_tools([add])
response = llm_with_tools.invoke("请计算 3 + 5")
print("\n=== Function Calling 测试 ===")
print(f"Tool calls: {response.tool_calls}")