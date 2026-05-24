import logging
import sys
import mcp_rag_opensearch

from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("retrieve-server")

try:
    mcp = FastMCP(
        name = "rag-opensearch"
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# RAG
######################################
@mcp.tool()
def retrieve(keyword: str) -> str:
    """
    Query the keyword using RAG.
    keyword: the keyword to query
    return: the result of query
    """
    logger.info(f"search --> keyword: {keyword}")

    return mcp_rag_opensearch.retrieve(keyword)

if __name__ =="__main__":
    mcp.run(transport="stdio")


