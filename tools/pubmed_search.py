# tools/pubmed_search.py
"""
PubMed 文献检索工具

使用 Biopython 的 Entrez 模块访问 NCBI PubMed 数据库，
搜索生物医学文献并返回标题、作者、摘要等信息。
"""
from langchain_core.tools import tool
from Bio import Entrez

# 设置你的邮箱（NCBI 要求提供邮箱用于追踪请求）
Entrez.email = "297225545@qq.com"  # ← 改成你的邮箱


@tool
def pubmed_search(query: str, max_results: int = 3) -> str:
    """
    搜索 PubMed 生物医学文献数据库。

    当用户需要查找生物医学领域的学术文献、研究论文、
    临床试验报告时使用此工具。

    Args:
        query: 英文搜索关键词，例如 "CRISPR cancer therapy"
        max_results: 返回结果数量，默认3篇

    Returns:
        包含文献标题、作者、发表日期和摘要的格式化文本
    """
    try:
        # Step 1: 搜索获取文献 ID 列表
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return f"未找到与 '{query}' 相关的文献。请尝试调整搜索关键词。"

        # Step 2: 获取文献详细信息
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="xml",
            retmode="xml"
        )
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()

        # Step 3: 解析并格式化结果
        results = []
        for i, article in enumerate(articles["PubmedArticle"], 1):
            medline = article["MedlineCitation"]
            article_data = medline["Article"]

            title = str(article_data.get("ArticleTitle", "无标题"))

            # 提取作者
            authors = article_data.get("AuthorList", [])
            if authors:
                author_names = []
                for author in authors[:3]:  # 最多取前3位作者
                    last = author.get("LastName", "")
                    first = author.get("ForeName", "")
                    if last:
                        author_names.append(f"{last} {first}")
                author_str = ", ".join(author_names)
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = "作者信息不可用"

            # 提取摘要
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(part) for part in abstract_parts) if abstract_parts else "摘要不可用"
            # 截断过长的摘要
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            # 提取 PMID
            pmid = str(medline.get("PMID", "N/A"))

            # 提取发表日期
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "N/A")
            month = pub_date.get("Month", "")
            date_str = f"{year} {month}".strip()

            results.append(
                f"[文献 {i}]\n"
                f"  标题: {title}\n"
                f"  作者: {author_str}\n"
                f"  日期: {date_str}\n"
                f"  PMID: {pmid}\n"
                f"  链接: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
                f"  摘要: {abstract}\n"
            )

        return f"共找到 {len(results)} 篇相关文献：\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"PubMed 搜索出错: {str(e)}。请检查网络连接或稍后重试。"