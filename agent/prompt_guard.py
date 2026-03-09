import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PromptGuard:
    """
    基于微调 Qwen2-7B 的二分类模型，判断用户输入是否为恶意注入。
    模型训练数据：正常科研问题（正样本） + 各类提示词注入变体（负样本）
    输出：0=正常，1=恶意注入
    """

    _model = None
    _tokenizer = None
    _device = None

    # 模型路径（微调后的checkpoint）
    MODEL_PATH = os.getenv("PROMPT_GUARD_MODEL_PATH", "models/prompt_guard_qwen2_7b")

    # 置信度阈值：高于此值判定为恶意
    THRESHOLD = 0.85

    @classmethod
    def _load_model(cls):
        """懒加载：首次调用时加载模型，后续复用"""
        if cls._model is not None:
            return

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cls._tokenizer = AutoTokenizer.from_pretrained(
            cls.MODEL_PATH,
            trust_remote_code=True,
        )
        cls._model = AutoModelForSequenceClassification.from_pretrained(
            cls.MODEL_PATH,
            num_labels=2,              # 二分类：正常 / 恶意
            trust_remote_code=True,
        ).to(cls._device)

        cls._model.eval()

    @classmethod
    def sanitize(cls, text: str) -> tuple[str, bool]:
        """
        检测输入是否为恶意注入
        返回：(原始文本, 是否恶意)
        - 不再做文本替换，交给 Supervisor 决定如何处理
        - 恶意判断基于模型置信度是否超过阈值
        """
        cls._load_model()

        inputs = cls._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(cls._device)

        with torch.no_grad():
            logits = cls._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # probs[0][0] = 正常概率, probs[0][1] = 恶意概率
            malicious_prob = probs[0][1].item()

        is_malicious = malicious_prob > cls.THRESHOLD

        return text, is_malicious