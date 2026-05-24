import json
import logging
import sys

import boto3
import utils
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_core.documents import Document
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("rag-opensearch")

config = utils.load_config()

opensearch_url = config.get("managed_opensearch_url")
if opensearch_url is None:
    raise Exception("No OpenSearch URL")

projectName = config.get("projectName", "langgraph-nova")
region = config.get("region", "us-west-2")

enableParentDocumentRetrival = "Enable"
enableHybridSearch = "Enable"

index_name = projectName
number_of_results = 5

session = boto3.Session(region_name=region)
credentials = session.get_credentials()

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token,
)

os_client = OpenSearch(
    hosts=[{"host": opensearch_url.replace("https://", ""), "port": 443}],
    http_compress=True,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
)

def lexical_search(query, top_k):
    min_match = 0

    search_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f"{min_match}%",
                                "operator": "or",
                            }
                        }
                    },
                ],
                "filter": [],
            }
        }
    }

    response = os_client.search(body=search_query, index=index_name)

    docs = []
    for i, document in enumerate(response["hits"]["hits"]):
        if i >= top_k:
            break

        excerpt = document["_source"]["text"]
        metadata = document["_source"].get("metadata") or {}
        name = metadata.get("name", "") or ""

        page = metadata.get("page", "") or ""

        url = metadata.get("url", "") or ""

        docs.append(
            Document(
                page_content=excerpt,
                metadata={
                    "name": name,
                    "url": url,
                    "page": page,
                    "from": "lexical",
                },
            )
        )

    for i, doc in enumerate(docs):
        text = doc.page_content[:100] if len(doc.page_content) >= 100 else doc.page_content
        logger.info(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")

    return docs


def get_parent_content(parent_doc_id):
    response = os_client.get(index=index_name, id=parent_doc_id)

    source = response["_source"]
    metadata = source.get("metadata") or {}

    name = metadata.get("name", "") or ""
    url = metadata.get("url", "") or ""

    return source["text"], name, url


def get_embedding():
    LLM_embedding = [
        {
            "bedrock_region": "us-west-2",
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0",
        },
        {
            "bedrock_region": "us-east-1",
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0",
        },
        {
            "bedrock_region": "us-east-2",
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0",
        },
    ]

    selected_embedding = 0
    embedding_profile = LLM_embedding[selected_embedding]
    bedrock_region = embedding_profile["bedrock_region"]
    model_id = embedding_profile["model_id"]
    logger.info(
        f"selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}"
    )

    boto3_bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=bedrock_region,
        config=Config(retries={"max_attempts": 30}),
    )

    return BedrockEmbeddings(
        client=boto3_bedrock,
        region_name=bedrock_region,
        model_id=model_id,
    )


def retrieve_documents_from_opensearch(query, top_k):
    logger.info("###### retrieve_documents_from_opensearch ######")

    bedrock_embedding = get_embedding()

    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name=index_name,
        is_aoss=False,
        embedding_function=bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=awsauth,
        connection_class=RequestsHttpConnection,
    )

    relevant_docs = []
    if enableParentDocumentRetrival == "Enable":
        result = vectorstore_opensearch.similarity_search_with_score(
            query=query,
            k=top_k * 2,
            search_type="script_scoring",
            pre_filter={"term": {"metadata.doc_level": "child"}},
        )
        logger.info(f"result: {result}")

        relevant_documents = []
        docList = []
        for re in result:
            if "parent_doc_id" in re[0].metadata:
                parent_doc_id = re[0].metadata["parent_doc_id"]
                doc_level = re[0].metadata["doc_level"]
                logger.info(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")

                if doc_level == "child":
                    if parent_doc_id in docList:
                        logger.info("duplicated")
                    else:
                        relevant_documents.append(re)
                        docList.append(parent_doc_id)
                        if len(relevant_documents) >= top_k:
                            break

        for i, doc in enumerate(relevant_documents):
            text = doc[0].page_content[:100] if len(doc[0].page_content) >= 100 else doc[0].page_content
            logger.info(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")

        for i, document in enumerate(relevant_documents):
            logger.info(f"## Document(opensearch-vector) {i+1}: {document}")

            parent_doc_id = document[0].metadata["parent_doc_id"]
            doc_level = document[0].metadata["doc_level"]

            content, name, url = get_parent_content(parent_doc_id)

            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "name": name,
                        "url": url,
                        "doc_level": doc_level,
                        "from": "vector",
                    },
                )
            )
    else:
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query=query,
            k=top_k,
        )

        for i, document in enumerate(relevant_documents):
            logger.info(f"## Document(opensearch-vector) {i+1}: {document}")
            name = document[0].metadata.get("name", "") or ""
            url = document[0].metadata.get("url", "") or ""
            content = document[0].page_content

            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "name": name,
                        "url": url,
                        "from": "vector",
                    },
                )
            )

    if enableHybridSearch == "Enable":
        relevant_docs += lexical_search(query, top_k)

    return relevant_docs


def retrieve(query: str) -> str:
    """Query managed OpenSearch (vector + hybrid lexical) and return MCP JSON."""
    logger.info(f"retrieve --> query: {query}")

    relevant_docs = retrieve_documents_from_opensearch(query, number_of_results)

    json_docs = []
    for doc in relevant_docs:
        name = doc.metadata.get("name", "") or ""
        title = name.split("/")[-1] if name else name
        json_docs.append(
            {
                "contents": doc.page_content,
                "reference": {
                    "url": doc.metadata.get("url", ""),
                    "title": title,
                    "from": doc.metadata.get("from", "RAG"),
                },
            }
        )

    logger.info(f"json_docs: {len(json_docs)} document(s)")
    return json.dumps(json_docs, ensure_ascii=False)
