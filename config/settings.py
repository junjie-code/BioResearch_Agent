# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# === LLM 配置 ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = "deepseek-chat"      # DeepSeek-V3，性价比最高
LLM_TEMPERATURE = 0.3            # Agent 场景建议低温度，减少幻觉
LLM_MAX_TOKENS = 4096

# === Embedding 配置 ===
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# === 路径配置 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "docs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "chroma_db")
UNET_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "unet_nuclei_epoch_20.pth")

# === Agent 配置 ===
MAX_AGENT_STEPS = 1        # Agent 最大执行步数，防止死循环