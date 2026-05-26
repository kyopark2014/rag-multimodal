import logging
import os
import re
import time
import json
import sys
import tempfile
import traceback
import boto3
import mcp_server_text_extraction as tex
import utils
import chat

from urllib import parse
from urllib.parse import urlparse
from typing import List, Literal, Optional
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("multimodal")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")

config = utils.load_config()
s3_bucket = config.get("s3_bucket")
region = config.get("region", "us-west-2")
sharing_url = (config.get("sharing_url") or "").rstrip("/")
s3_prefix = "docs"
markdown_s3_prefix = "markdown"
meta_prefix = "metadata/"

LLM_PROMPT = (
    "페이지 내용을 Markdown 형식으로 변환합니다. 평문이 아니라 제목(#·##)·목록·강조·코드 블록 등 "
    "Markdown 문법을 적절히 써서 구조화해 주세요. 문장 단위로 읽기 쉽게 구분합니다. "
    "상단의 header와 하단의 footer는 출력에서 제외합니다. 상단 header는 주로 현재 페이지 제목이고, "
    "footer에는 페이지 번호 등이 있는데, 변환 결과에는 포함하지 않습니다.\n\n"
    "페이지에 그림·도표·사진·스크린샷·다이어그램·캡처 등 시각적 요소가 있으면, 그 이미지가 무엇을 보여주는지·"
    "본문과 어떤 관계인지·어떤 정보를 전달하는지를 빠짐없이 상세히 풀어서 서술합니다."
)


def _extract_text_from_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = tex._prepare_image_base64(raw)
    raw_text = tex._extract_text_with_llm(b64, LLM_PROMPT)
    return tex._parse_result(raw_text).strip()


def _ensure_fitz():
    try:
        import fitz
    except ImportError:
        logger.info("PyMuPDF is not installed. Installing now …")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pymupdf"],
            stdout=subprocess.DEVNULL,
        )
        import fitz
    return fitz


def _s3_key_from_url(file_url: str) -> Optional[str]:
    """Derive S3 object key (or path-like key) from *file_url*."""
    if file_url.startswith("s3://"):
        without_scheme = file_url[5:]
        _, _, key = without_scheme.partition("/")
        return key or None
    if sharing_url and file_url.startswith(sharing_url):
        return parse.unquote(file_url[len(sharing_url) :].lstrip("/"))
    if file_url.startswith(("http://", "https://")):
        return parse.unquote(urlparse(file_url).path.lstrip("/"))
    if not file_url.startswith("~"):
        return file_url.lstrip("/")
    return None


def _artifact_stem(file_url: str) -> str:
    """Folder/file stem from the original PDF name in *file_url*, not the temp download path."""
    local = os.path.expanduser(file_url)
    if os.path.isfile(local):
        return os.path.splitext(os.path.basename(local))[0]

    s3_key = _s3_key_from_url(file_url)
    if s3_key:
        name = os.path.basename(s3_key)
        stem, _ = os.path.splitext(name)
        if stem:
            return stem

    raise FileNotFoundError(f"Cannot determine PDF name from: {file_url}")


def _resolve_pdf_path(file_url: str) -> tuple[str, bool]:
    """Resolve *file_url* to a local PDF path.

    Returns:
        (local_path, is_temp) — *is_temp* is True when the caller should delete the file.
    """
    local = os.path.expanduser(file_url)
    if os.path.isfile(local):
        return local, False

    s3_key = _s3_key_from_url(file_url)
    bucket = s3_bucket
    logger.info(f"bucket: {bucket}, s3_key: {s3_key}")

    if file_url.startswith("s3://"):
        without_scheme = file_url[5:]
        bucket, _, _ = without_scheme.partition("/")

    if s3_key and bucket:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        s3 = boto3.client("s3", region_name=region)
        logger.info(f"Downloading s3://{bucket}/{s3_key} → {tmp_path}")
        s3.download_file(bucket, s3_key, tmp_path)
        return tmp_path, True

    raise FileNotFoundError(f"PDF not found: {file_url}")

def get_embedding():
    global selected_embedding
    bedrock_region =  region
    model_id = "amazon.titan-embed-text-v2:0"
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )      
    return bedrock_embedding

def get_contextual_docs_from_chunks(whole_doc, splitted_docs): # per chunk
    contextual_template = (
        "<document>"
        "{WHOLE_DOCUMENT}"
        "</document>"
        "Here is the chunk we want to situate within the whole document."
        "<chunk>"
        "{CHUNK_CONTENT}"
        "</chunk>"
        "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk."
        "Answer only with the succinct context and nothing else."
        "Put it in <result> tags."
    )          
    
    contextual_prompt = ChatPromptTemplate([
        ('human', contextual_template)
    ])

    contexualized_docs = []
    contexualized_chunks = []
    for i, doc in enumerate(splitted_docs):
        # chat = get_contexual_retrieval_chat()
        llm = chat.get_chat()
        
        contexual_chain = contextual_prompt | llm
            
        response = contexual_chain.invoke(
            {
                "WHOLE_DOCUMENT": whole_doc.page_content,
                "CHUNK_CONTENT": doc.page_content
            }
        )
        # print('--> contexual chunk: ', response)
        output = response.content
        contextualized_chunk = output[output.find('<result>')+8:output.find('</result>')]
        contextualized_chunk.replace('\n', '')
        contexualized_chunks.append(contextualized_chunk)
        
        print(f"--> {i}: original_chunk: {doc.page_content}")
        print(f"--> {i}: contexualized_chunk: {contextualized_chunk}")
        
        contexualized_docs.append(
            Document(
                page_content="\n"+contextualized_chunk+"\n\n"+doc.page_content,
                metadata=doc.metadata
            )
        )
    return contexualized_docs, contexualized_chunks

def create_metadata(bucket, key, meta_prefix, url, category, documentId, ids, files):
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_url": url,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,      
        "ids": ids,
        "files": files
    }
    print('metadata: ', metadata)

    if markdown_s3_prefix in key:
        rel_key = key[key.find(markdown_s3_prefix) + len(markdown_s3_prefix) + 1 :]
    elif s3_prefix in key:
        rel_key = key[key.find(s3_prefix) + len(s3_prefix) + 1 :]
    else:
        rel_key = key
    objectName = os.path.basename(rel_key)
    print('objectName: ', objectName)

    client = boto3.client('s3')
    try: 
        metadata_key = meta_prefix + objectName + ".metadata.json"
        client.put_object(
            Body=json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
            Bucket=bucket,
            Key=metadata_key,
            ContentType="application/json",
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

def get_documentId(key, category):
    documentId = category + "-" + key
    documentId = documentId.replace(' ', '_') # remove spaces  
    documentId = documentId.replace(',', '_') # remove commas # not allowed: [ " * \\ < | , > / ? ]
    documentId = documentId.replace('/', '_') # remove slash
    documentId = documentId.lower() # change to lowercase
                
    return documentId

def add_to_opensearch(body, name: str = "", url: str = ""):
    index_name = config.get("projectName")
    doc_metadata = {}
    if name:
        doc_metadata["name"] = name
    if url:
        doc_metadata["url"] = url
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    bedrock_embeddings = get_embedding()
    opensearch_url = config.get("managed_opensearch_url")
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token,
    )

    vectorstore = OpenSearchVectorSearch(
        index_name=index_name,  
        is_aoss = False,
        #engine="faiss",  # default: nmslib
        embedding_function=bedrock_embeddings,
        opensearch_url=opensearch_url,
        http_auth=awsauth,
        connection_class=RequestsHttpConnection
    )  

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        # separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    )
    parent_chunks = parent_splitter.split_text(body)
    splitted_docs = [
        Document(page_content=chunk, metadata=dict(doc_metadata))
        for chunk in parent_chunks
    ]
    logger.info(f"len(splitted_docs): {len(splitted_docs)}")
    if splitted_docs:
        logger.info(f"splitted_docs[0]: {splitted_docs[0].page_content}")

    parent_docs = []
    if chat.contextual_embedding == 'Enable':
        whole_doc = Document(page_content=body, metadata=dict(doc_metadata))
        parent_docs, contexualized_chunks = get_contextual_docs_from_chunks(whole_doc, splitted_docs)

        logger.info(f"parent contextual chunk[0]: {parent_docs[0].page_content}")
        logger.info(f"contexualized_chunks[0]: {contexualized_chunks[0]}")
    else:
        parent_docs = splitted_docs

    ids = []
    if len(parent_docs):
        lastest_page = 0
        for i, doc in enumerate(parent_docs):
            text = doc.page_content

            pages = []
            open_tag = "<page>"
            close_tag = "</page>"
            start = 0
            while True:
                page_tag = text.find(open_tag, start)
                if page_tag == -1:
                    break
                content_start = page_tag + len(open_tag)
                end_tag = text.find(close_tag, content_start)
                if end_tag == -1:
                    break
                page_num = text[content_start:end_tag].strip()
                if page_num.isdigit():
                    pages.append(int(page_num))
                start = end_tag + len(close_tag)
            logger.info(f"related pages: {pages}")
            
            if pages:
                doc.metadata["page"] = int(pages[0]) # first page number
                lastest_page = int(pages[-1])
            else:
                doc.metadata["page"] = int(lastest_page)
                logger.info(f"lastest_page: {lastest_page}") # use latest page number if no pages tag

            doc.metadata["doc_level"] = "parent"

            # remove <page> tags from content
            doc.page_content = re.sub(r'\n<page>\d+</page>', '', doc.page_content)

        logger.info(f"parent_docs[0]: {parent_docs[0].page_content}")

        try:
            parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
            logger.info(f"parent_doc_ids: {parent_doc_ids}")
            ids = parent_doc_ids

            for i, doc in enumerate(parent_docs):
                _id = parent_doc_ids[i]
                child_docs = child_splitter.split_documents([doc])
                for _doc in child_docs:
                    _doc.metadata["parent_doc_id"] = _id
                    _doc.metadata["doc_level"] = "child"

                if chat.contextual_embedding == 'Enable':
                    contexualized_child_docs = [] # contexualized child doc
                    for _doc in child_docs:
                        # remove <page> tags from content
                        page_content = re.sub(r'\n<page>\d+</page>\n', '', _doc.page_content)
                        
                        contexualized_child_docs.append(
                            Document(
                                page_content=contexualized_chunks[i]+"\n\n"+page_content   ,
                                metadata=_doc.metadata
                            )
                        )
                    child_docs = contexualized_child_docs

                logger.info(f"child_docs[0]: {child_docs[0].page_content}")

                child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
                logger.info(f"child_doc_ids: {child_doc_ids}")
                logger.info(f"len(child_doc_ids): {len(child_doc_ids)}")
                    
                ids += child_doc_ids
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                
            #raise Exception ("Not able to add docs in opensearch")          

    return ids

def upload_to_s3(file_bytes, key):
    """
    Upload a file to S3 and return the URL
    """

    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
        )

        content_type = utils.get_contents_type(key)       
        # logger.info(f"content_type: {content_type}") 

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=key,
            ContentType=content_type,
            Body=file_bytes
        )
        logger.info(f"Uploaded s3://{s3_bucket}/{key}")

        return key
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.error(err_msg)
        return None

def pdf_to_images(file_url: str, dpi: Optional[int] = None) -> list[str]:
    """Convert every page of the PDF at *file_url* to PNG images.

    *file_url* may be a local path, S3 URI, sharing URL, or S3 object key.
    Images are saved under ``artifacts/<pdf_stem>/page_001.png``, …

    Args:
        dpi: Resolution for rendered images. Defaults to 150 when omitted.

    Returns:
        List of absolute paths to the saved image files.
    """
    fitz = _ensure_fitz()
    if dpi is None:
        dpi = 150

    pdf_path, is_temp = _resolve_pdf_path(file_url)
    stem = _artifact_stem(file_url)
    try:
        output_dir = os.path.join(ARTIFACTS_DIR, stem)
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        total = len(doc)
        saved = []
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            filename = f"page_{i:03d}.png"
            out_path = os.path.join(output_dir, filename)
            pix.save(out_path)
            saved.append(out_path)
            logger.info(f"  [{i}/{total}] Saved → {out_path}")

            # save to s3 (Body must be bytes, not a PyMuPDF Pixmap)
            upload_to_s3(pix.tobytes("png"), f"images/{stem}/{filename}")

        doc.close()
        return saved
    finally:
        if is_temp and os.path.isfile(pdf_path):
            os.remove(pdf_path)

def img2text(images: list[str], filename: Optional[str] = None) -> list[str]:
    """Convert images to per-page markdown files (e.g. page_001.png → page_001.md).

    Markdown files are always written under ``artifacts/<filename>/``.

    Args:
        images: List of absolute paths to the image files.
        filename: Artifacts subfolder name (defaults to parent dir of the first image).

    Returns:
        List of absolute paths to the generated markdown files.
    """
    if filename is None:
        filename = os.path.basename(os.path.dirname(images[0]))
    output_dir = os.path.join(ARTIFACTS_DIR, filename)
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []

    # Extract text from each image and save as a markdown file
    pages: list[str] = []
    for i, img_path in enumerate(images, start=1):
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        out_path = os.path.join(output_dir, f"{stem}.md")
        if os.path.isfile(out_path):
            logger.info(f"[{i}/{len(images)}] {stem}.md already exists, skipping")
            saved.append(out_path)
            with open(out_path, encoding="utf-8") as f:
                pages.append(f.read())
            continue

        logger.info(f"[{i}/{len(images)}] {img_name} → {stem}.md")
        try:
            body = _extract_text_from_image(img_path)
        except Exception as e:
            body = f"> (추출 오류: {e})"
        text = body.rstrip() + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        saved.append(out_path)
        pages.append(body)
        logger.info(f"Wrote markdown → {out_path}")
    
    # Upload concatenated page text as a single markdown file to S3
    extracted_text = '\n'.join(pages)
    s3_client = boto3.client("s3", region_name=region) if s3_bucket else None
    s3_key = f"{markdown_s3_prefix}/{filename}.md"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        ContentType="text/markdown",
        Body=extracted_text.encode("utf-8"),
    )
    logger.info(f"Uploaded s3://{s3_bucket}/{s3_key}")

    # Log the CloudFront sharing URL for the uploaded markdown
    config = utils.load_config()
    markdown_url = config['sharing_url'] + f"/{markdown_s3_prefix}/{filename}.md"
    logger.info(f"markdown_url: {markdown_url}")

    # Wrap each page with <page> tags for RAG page metadata
    rag_body = ""    
    md_path = os.path.join(output_dir, f"{filename}.md")    
    for i, page in enumerate(pages):
        tag = f'\n<page>{i+1}</page>\n'
        rag_body += f"{page}{tag}"    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(rag_body)

    # add to opensearch
    path = (config.get("sharing_url") or sharing_url or "").rstrip("/")    
    doc_url = f"{path}/{s3_prefix}/{filename}.pdf" if path else ""

    ids = add_to_opensearch(rag_body, name=filename, url=doc_url)

    # metadata for the document
    category = "upload"
    documentId = get_documentId(s3_key, category)
    create_metadata(bucket=s3_bucket, key=s3_key, meta_prefix=meta_prefix, url=doc_url or path + parse.quote(s3_key), category=category, documentId=documentId, ids=ids, files=saved)

    return extracted_text        

def sync_data_source(file_url: str) -> Optional[list[str]]:
    """PDF → images → LLM Markdown → S3, then trigger knowledge-base ingestion."""
    stem = _artifact_stem(file_url)
    images = pdf_to_images(file_url)
    if not images:
        logger.warning("No images generated from PDF")
        return None

    extracted_body = img2text(images, filename=stem)
    return extracted_body if extracted_body else None
