# tools/__init__.py
"""工具注册中心"""
from tools.pubmed_search import pubmed_search
from tools.sequence_analysis import sequence_analysis
from tools.cell_image_analysis import cell_image_analysis
from tools.knowledge_qa import knowledge_qa
from tools.report_generator import report_generator

ALL_TOOLS = [
    pubmed_search,
    sequence_analysis,
    cell_image_analysis,
    knowledge_qa,
    report_generator,
]