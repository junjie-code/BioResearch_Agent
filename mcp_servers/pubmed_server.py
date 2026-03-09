import json
from mcp.server import Server
from mcp.types import TextContent

from tools.pubmed_search import search_pubmed

server = Server("pubmed-mcp-server")


@server.tool()
async def pubmed_search(query: str, max_results: int = 5) -> list[TextContent]:
    """检索PubMed生物医学文献数据库"""
    try:
        results = search_pubmed(query, max_results=max_results)
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@server.tool()
async def pubmed_search_by_gene(gene_name: str, year_from: int = 2023) -> list[TextContent]:
    """按基因名检索最新文献"""
    query = f"{gene_name} AND {year_from}[PDAT]:3000[PDAT]"
    try:
        results = search_pubmed(query, max_results=5)
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())