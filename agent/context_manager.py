"""
上下文管理器：为每个Agent构建精简的、相关的上下文
避免所有历史消息无差别塞给每个Agent
"""


class ContextManager:
    # 每个Agent关心的关键词
    AGENT_KEYWORDS = {
        "lit_agent":   ["文献", "论文", "研究", "paper", "pubmed", "综述", "发现"],
        "seq_agent":   ["序列", "基因", "蛋白", "DNA", "GC含量", "碱基", "翻译", "FASTA"],
        "image_agent": ["图像", "图片", "细胞",  "显微", "电泳"],
    }

    def build_agent_context(self, state: dict, agent_name: str) -> list:
        """为特定Agent构建精简上下文"""
        messages = state.get("messages", [])
        if not messages:
            return []

        context = []

        # 1. 始终包含当前用户输入
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            content = last_msg.get("content", "")
            if isinstance(content, str):
                context.append(content)
            elif isinstance(content, list):
                context.append(" ".join(c.get("text", "") for c in content if c.get("type") == "text"))
        else:
            context.append(str(last_msg))

        # 2. 历史消息中只保留与当前Agent相关的
        keywords = self.AGENT_KEYWORDS.get(agent_name, [])
        for msg in messages[:-1]:
            msg_text = str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
            if any(kw in msg_text for kw in keywords):
                context.append(msg_text)

        # 3. 限制总长度（粗略估计，按字符数）
        total = ""
        trimmed = []
        for c in context:
            if len(total) + len(c) > 8000:  # 约4000 tokens
                break
            total += c
            trimmed.append(c)

        return trimmed

    def build_prior_results_summary(self, agent_results: dict, max_chars: int = 600) -> str:
        """将前序Agent的结果摘要压缩，供下游Agent参考"""
        parts = []
        per_agent_max = max_chars // max(len(agent_results), 1)
        for name, result in agent_results.items():
            text = result if isinstance(result, str) else str(result)
            truncated = text[:per_agent_max] + ("..." if len(text) > per_agent_max else "")
            parts.append(f"[{name}] {truncated}")
        return "\n".join(parts)

    def build_supervisor_summary(self, agent_results: dict) -> str:
        """Supervisor汇总用，每个Agent结果最多1500字符"""
        parts = []
        for name, result in agent_results.items():
            text = result if isinstance(result, str) else str(result)
            truncated = text[:1500] + ("..." if len(text) > 1500 else "")
            parts.append(f"### {name} 的分析结果：\n{truncated}")
        return "\n\n".join(parts)