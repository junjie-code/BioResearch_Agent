"""
模型工厂：不同Agent使用不同模型，统一管理
"""
import os
from langchain_openai import ChatOpenAI


class ModelFactory:
    # 集中配置，切换模型只改这里
    # image_agent 不调LLM（纯U-Net推理），不需要配置
    CONFIG = {
        "supervisor":  {"model": "deepseek-reasoner",  "provider": "deepseek", "temperature": 0},
        "lit_agent":   {"model": "deepseek-chat",  "provider": "deepseek", "temperature": 0},
        "seq_agent":   {"model": "deepseek-chat",  "provider": "deepseek", "temperature": 0},
        "summarizer":  {"model": "deepseek-reasoner",  "provider": "deepseek", "temperature": 0.3},
    }

    @classmethod
    def get_llm(cls, agent_name: str) -> ChatOpenAI:
        cfg = cls.CONFIG.get(agent_name, cls.CONFIG["supervisor"])

        if cfg["provider"] == "deepseek":
            return ChatOpenAI(
                model=cfg["model"],
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v3",
                temperature=cfg["temperature"],
            )
        elif cfg["provider"] == "openai":
            return ChatOpenAI(
                model=cfg["model"],
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=cfg["temperature"],
            )