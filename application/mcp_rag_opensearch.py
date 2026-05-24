import json
import logging
import sys
from multiprocessing import Pipe, Process

import boto3
import info
import utils
from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from opensearchpy import OpenSearch, RequestsHttpConnection
from pydantic.v1 import BaseModel, Field
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

enableHybridSearch = "Enable"
enableGrading = "Enable"
multi_region = "Disable"
grading_model_name = "Claude 4.5 Sonnet"

index_name = projectName
number_of_results = 5
selected_chat = 0

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

        reference_content = reference_content_from_text(excerpt)

        docs.append(
            Document(
                page_content=reference_content,
                metadata={
                    "name": name,
                    "url": url,
                    "page": page,
                    "from": "lexical",
                },
            )
        )

    for i, doc in enumerate(docs):
        text = doc.page_content[:200] if len(doc.page_content) >= 200 else doc.page_content
        logger.info(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")

    return docs


def get_parent_content(parent_doc_id):
    response = os_client.get(index=index_name, id=parent_doc_id)

    source = response["_source"]
    metadata = source.get("metadata") or {}

    name = metadata.get("name", "") or ""
    url = metadata.get("url", "") or ""

    return source["text"], name, url


_CONTEXTUAL_PREFIX_MARKERS = (
    "이 청크",
    "This chunk",
    "이 문서",
    "This document",
    "The chunk",
)
_BODY_START_MARKERS = ("---", "#", "|", "<page>")


def content_after_first_separator(content: str, separator: str = "---") -> str:
    """Return text after the first '---' separator (skip prefix before the body)."""
    if not content:
        return content
    idx = content.find(separator)
    if idx == -1:
        return content
    return content[idx + len(separator) :].lstrip("\n")


def strip_contextual_prefix(content: str) -> str:
    """Remove contextual embedding prefix when present (context + \\n\\n + body)."""
    if not content:
        return content

    stripped = content.lstrip("\n")
    if "\n\n" not in stripped:
        return content

    prefix, body = stripped.split("\n\n", 1)
    body = body.lstrip("\n")

    if body.startswith(_BODY_START_MARKERS):
        return body
    if prefix.startswith(_CONTEXTUAL_PREFIX_MARKERS):
        return body

    return content


def reference_content_from_text(text: str) -> str:
    """Prepare text for RAG reference (strip contextual prefix when indexed with it)."""
    return strip_contextual_prefix(text)


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

    result = vectorstore_opensearch.similarity_search_with_score(
        query=query,
        k=top_k * 2,
        search_type="script_scoring",
        pre_filter={"term": {"metadata.doc_level": "child"}},
    )
    # logger.info(f"result: {result}")

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

    relevant_docs = []    
    for i, (child_doc, score) in enumerate(relevant_documents):
        metadata = child_doc.metadata
        parent_doc_id = metadata["parent_doc_id"]
        doc_level = metadata["doc_level"]
        page = metadata.get("page", "") or ""

        logger.info(
            f"## Document(opensearch-vector) {i+1}: parent_doc_id={parent_doc_id}, page={page}, score={score}"
        )

        content, name, url = get_parent_content(parent_doc_id)
        logger.info(f"content: {content}")

        body = content_after_first_separator(content)
        reference_content = reference_content_from_text(body)
        logger.info(f"reference_content: {reference_content}")

        relevant_docs.append(
            Document(
                page_content=reference_content,
                metadata={
                    "name": name,
                    "url": url,
                    "doc_level": doc_level,
                    "page": page,
                    "from": "vector",
                },
            )
        )
        logger.info(f"reference_content: {reference_content}")
    
    if enableHybridSearch == "Enable":
        relevant_docs += lexical_search(query, top_k/2)

    return relevant_docs


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def _get_grading_chat(models, selected):
    profile = models[selected]
    bedrock_region = profile["bedrock_region"]
    model_id = profile["model_id"]
    model_type = profile["model_type"]
    max_output_tokens = 4096
    logger.info(
        "grading LLM: selected=%s, region=%s, model_id=%s, model_type=%s",
        selected,
        bedrock_region,
        model_id,
        model_type,
    )

    if model_type == "nova":
        stop_sequence = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == "claude":
        stop_sequence = "\n\nHuman:"
    else:
        stop_sequence = "\n\nHuman:"

    boto3_bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=bedrock_region,
        config=Config(retries={"max_attempts": 30}),
    )
    # Claude 4.x: temperature and top_p cannot both be set (use chat.py style).
    if model_type == "openai":
        parameters = {
            "max_tokens": max_output_tokens,
            "temperature": 0.1,
        }
    elif model_type == "nova":
        parameters = {
            "max_tokens": max_output_tokens,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": [stop_sequence],
        }
    else:
        parameters = {
            "max_tokens": max_output_tokens,
            "stop_sequences": [stop_sequence],
        }
    return ChatBedrock(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=parameters,
        region_name=bedrock_region,
    )


def get_retrieval_grader(chat):
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    return grade_prompt | structured_llm_grader


def _grade_document_based_on_relevance(conn, question, doc, models, selected):
    chat = _get_grading_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke(
        {"question": question, "document": doc.page_content}
    )
    if score.binary_score.lower() == "yes":
        logger.info("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:
        logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    conn.close()


def _grade_documents_using_parallel_processing(models, question, documents):
    global selected_chat

    number_of_models = len(models)
    filtered_docs = []
    processes = []
    parent_connections = []

    for doc in documents:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        process = Process(
            target=_grade_document_based_on_relevance,
            args=(child_conn, question, doc, models, selected_chat),
        )
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0

    for process in processes:
        process.start()

    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()
        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()

    return filtered_docs


def grade_documents(question, documents):
    """Filter retrieved documents by LLM relevance grading (see lambda_function.py)."""
    logger.info("###### grade_documents ######")

    if not documents:
        return []

    models = info.get_model_info(grading_model_name)
    if multi_region == "Enable":
        return _grade_documents_using_parallel_processing(models, question, documents)

    llm = _get_grading_chat(models, 0)
    retrieval_grader = get_retrieval_grader(llm)
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score.lower() == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
    return filtered_docs


def retrieve(query: str) -> str:
    """Query managed OpenSearch (vector + hybrid lexical) and return MCP JSON."""
    logger.info(f"retrieve --> query: {query}")

    relevant_docs = retrieve_documents_from_opensearch(query, number_of_results)

    if enableGrading == "Enable":
        logger.info("grading enabled for %s document(s)", len(relevant_docs))
        relevant_docs = grade_documents(query, relevant_docs)
        logger.info("%s document(s) after grading", len(relevant_docs))

    json_docs = []
    seen_contents = set()
    for i, doc in enumerate(relevant_docs):
        logger.info(f"doc[{i}]: {doc}")

        if doc.page_content in seen_contents:
            continue
        seen_contents.add(doc.page_content)

        name = doc.metadata.get("name", "") or ""
        title = name.split("/")[-1] if name else name
        page = doc.metadata.get("page", "")
        json_docs.append(
            {
                "contents": doc.page_content,
                "reference": {
                    "url": doc.metadata.get("url", ""),
                    "title": title,
                    "page": page,
                    "from": doc.metadata.get("from", "RAG"),
                },
            }
        )

    logger.info(f"json_docs: {len(json_docs)} document(s)")
    return json.dumps(json_docs, ensure_ascii=False)
