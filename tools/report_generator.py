# tools/report_generator.py
"""
报告生成工具

将 Agent 收集到的信息整合成结构化的 Markdown 报告。
利用 LLM 进行内容组织和润色。
"""
import os
from datetime import datetime
from langchain_core.tools import tool


@tool
def report_generator(
    topic: str,
    content: str,
    report_type: str = "综述"
) -> str:
    """
    将收集到的信息整合成结构化报告。

    当用户需要将多个来源的信息（如文献检索结果、知识库内容、
    实验数据）汇总成一份报告时使用此工具。

    Args:
        topic: 报告的主题，例如 "CRISPR在肿瘤治疗中的应用"
        content: 需要整合的原始内容（来自其他工具的输出）
        report_type: 报告类型，可选 "综述"、"实验报告"、"分析报告"

    Returns:
        格式化的 Markdown 报告文本
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# {report_type}：{topic}

**生成时间：** {now}
**生成方式：** BioResearch Agent 自动生成

---

## 摘要

本{report_type}基于 Agent 自动检索和分析的结果，围绕「{topic}」主题进行整理。

## 详细内容

{content}

---

## 说明

本报告由 BioResearch Agent 自动生成，内容来源包括 PubMed 文献数据库、
内部知识库等。建议结合原始文献进行进一步验证。
"""

    # 保存报告到文件
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return f"报告已生成并保存到: {filepath}\n\n{report}"