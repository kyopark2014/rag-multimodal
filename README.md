# RAG Multimodal

여기에서는 Multimodal LLM을 이용하여 RAG를 구현합니다. 지식저장소로 managed OpenSearch를 사용합니다. 파일업로드시 pdf의 경우에 각 page 단위로 이미지를 추출한 후에 OCR을 수행합니다. 이후 하나의 markdown 파일을 생성한 후에 chunking과 embedding후에 OpenSearch에 document를 추가합니다. Amazon S3에 저장된 파일이 삭제될 경우에는 해당 파일의 meta를 확인하여 OpenSearch에서 관련된 Document를 삭제합니다.

Streamlit에서 OpenSearch로 검색을 수행하면 hybrid search를 이용해 vector와 lexical로 검색한 결과를 추출합니다. 이를 grading을 통해 관련성을 검토하여 관련된 문서를 이용해 답변을 생성합니다.


## Advanced RAG 기법

RAG의 성능을 높이기 위해 이 프로젝트에 적용한 advanced RAG 기법을 정리합니다. 구현은 주로 [`application/multimodal.py`](./application/multimodal.py)(문서 인덱싱), [`application/mcp_rag_opensearch.py`](./application/mcp_rag_opensearch.py)(검색·하이브리드), [`application/mcp_server_text_extraction.py`](./application/mcp_server_text_extraction.py)(이미지→텍스트)에 있습니다.

### OCR

PDF를 페이지별 PNG로 렌더링한 뒤 멀티모달 LLM으로 Markdown을 추출하고, OpenSearch에 parent/child 청크로 적재합니다. 업로드 시 [`application/app.py`](./application/app.py)에서 `multimodal.sync_data_source()`를 호출하며, 내부 흐름은 `pdf_to_images` → `img2text` → `add_to_opensearch` 입니다.

Contextual embedding은 `chat.contextual_embedding`이 `'Enable'`일 때 parent/child 청크에 문서 전체 맥락을 붙입니다. 인덱스·검색용 OpenSearch 클라이언트와 벡터스토어는 아래와 같습니다.

[`application/mcp_rag_opensearch.py`](./application/mcp_rag_opensearch.py) — 검색·parent 조회·lexical 하이브리드:

```python
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
```

[`application/multimodal.py`](./application/multimodal.py) — 문서 추가·삭제 시 벡터스토어:

```python
vectorstore = OpenSearchVectorSearch(
    index_name=index_name,
    is_aoss=False,
    embedding_function=bedrock_embeddings,
    opensearch_url=opensearch_url,
    http_auth=awsauth,
    connection_class=RequestsHttpConnection,
)
```

각 parent 청크가 전체 문서에서 어떤 위치·의미를 갖는지 설명하는 contextual text는 `get_contextual_docs_from_chunks`로 생성합니다 (`chat.get_chat()` 사용).

```python
def get_contextual_docs_from_chunks(whole_doc, splitted_docs):
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
    contextual_prompt = ChatPromptTemplate([("human", contextual_template)])
    # ...
    contextualized_chunk = output[output.find("<result>") + 8 : output.find("</result>")]
    contexualized_docs.append(
        Document(
            page_content="\n" + contextualized_chunk + "\n\n" + doc.page_content,
            metadata=doc.metadata,
        )
    )
```

페이지 이미지에서 텍스트를 뽑을 때는 크기 제한(약 200만 픽셀, base64 5MB 이하)을 맞춘 뒤 Bedrock 멀티모달로 Markdown을 추출합니다. [`application/mcp_server_text_extraction.py`](./application/mcp_server_text_extraction.py)의 `_prepare_image_base64` / `_extract_text_with_llm`을 [`multimodal.py`](./application/multimodal.py)의 `_extract_text_from_image`가 호출합니다.

```python
def _extract_text_from_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = tex._prepare_image_base64(raw)
    raw_text = tex._extract_text_with_llm(b64, LLM_PROMPT)
    return tex._parse_result(raw_text).strip()
```

페이지별 Markdown을 `<page>N</page>` 태그로 이어 `rag_body`를 만든 뒤 `add_to_opensearch(rag_body, ...)`로 인덱싱합니다. S3에는 `markdown/{문서이름}.md`와 `metadata/{문서이름}.metadata.json`(벡터 `ids` 포함)이 저장됩니다.

대화형 **이미지 분석** 모드에서는 [`application/chat.py`](./application/chat.py)의 `extract_text` / `summary_image`로 텍스트 추출과 요약을 수행합니다 (`summary_image(img_base64, instruction)`).

```python
def summary_image(img_base64, instruction):
    query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    if instruction:
        query = f"{instruction}. <result> tag를 붙여주세요. 한국어로 답변하세요."
    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": query},
            ]
        )
    ]
    result = llm.invoke(messages)
    return result.content
```

### Parent Child Chunking

검색은 작은 **child** 청크로 하고, 답변에 쓰는 본문은 **parent** 청크에서 가져옵니다. [`application/multimodal.py`](./application/multimodal.py)의 `add_to_opensearch`에서 `RecursiveCharacterTextSplitter`로 parent/child를 나눕니다.

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
)
```

parent를 먼저 OpenSearch에 넣고, child metadata에 `parent_doc_id`와 `doc_level`을 넣습니다. contextual embedding이 켜져 있으면 parent에서 얻은 contextual text를 child `page_content` 앞에 붙입니다. 반환된 `ids`는 metadata JSON에 저장되어, [`lambda-s3-event-manager`](./lambda-s3-event-manager/lambda_function.py)가 PDF 삭제 시 OpenSearch 벡터를 지울 때 사용합니다.

```python
parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size=10000)
ids = parent_doc_ids

for i, doc in enumerate(parent_docs):
    _id = parent_doc_ids[i]
    child_docs = child_splitter.split_documents([doc])
    for _doc in child_docs:
        _doc.metadata["parent_doc_id"] = _id
        _doc.metadata["doc_level"] = "child"

    if chat.contextual_embedding == "Enable":
        contexualized_child_docs = []
        for _doc in child_docs:
            page_content = re.sub(r"\n<page>\d+</page>\n", "", _doc.page_content)
            contexualized_child_docs.append(
                Document(
                    page_content=contexualized_chunks[i] + "\n\n" + page_content,
                    metadata=_doc.metadata,
                )
            )
        child_docs = contexualized_child_docs

    child_doc_ids = vectorstore.add_documents(child_docs, bulk_size=10000)
    ids += child_doc_ids
```

검색 시 [`application/mcp_rag_opensearch.py`](./application/mcp_rag_opensearch.py)는 `metadata.doc_level: child`로 벡터 검색한 뒤, `parent_doc_id`로 parent 본문을 `os_client.get`으로 읽어 reference에 사용합니다. 하이브리드 검색이 켜져 있으면 lexical 검색 결과를 함께 합칩니다.

## 설치

### 사전 요구 사항

- Python 3.x
- AWS CLI 자격 증명이 구성된 상태 (`aws configure` 또는 환경 변수)
- `pip install -r requirements.txt`

### 인프라 배포

프로젝트 루트에서 installer를 실행합니다.

```bash
python3 installer.py
```

installer는 다음 리소스를 생성·구성합니다.

- S3 버킷 (`docs/` 프리픽스)
- Amazon OpenSearch Service 관리형 도메인 (`rag-multimodal`)
- CloudFront 배포
- **lambda-s3-event-manager**: S3 `docs/` PDF **삭제** 시 `metadata/*.metadata.json`의 `ids`로 OpenSearch 벡터 삭제 (IAM 역할 포함)

설치가 끝나면 `application/config.json`이 갱신됩니다.

### OpenSearch Dashboards (브라우저 접속)

installer는 OpenSearch **Fine-grained access control(FGAC)** 을 활성화하여 브라우저에서 Dashboards에 로그인할 수 있게 합니다.

| 항목 | 값 |
|------|-----|
| 사용자명 | `admin` (고정) |
| 비밀번호 | 설치 시 터미널에서 직접 입력 (두 번 확인), `application/config.json`에 저장 |

비밀번호 규칙 (AWS OpenSearch FGAC): 8~128자, 대문자·소문자·숫자 각 1자 이상.

FGAC가 이미 켜진 도메인을 재설치할 때는 비밀번호 입력을 건너뛰며, `config.json`에 기존 `managed_opensearch_dashboards_password`가 있으면 그대로 유지합니다.

installer는 FGAC 활성화 후 **도메인 액세스 정책**을 갱신합니다(IAM root + Dashboards용 요청 허용). FGAC 마이그레이션 모드(`AnonymousAuthEnabled`)가 켜져 있으면 먼저 끈 뒤 정책을 적용합니다. 실제 권한은 FGAC가 검사합니다.

#### 설치 후 OpenSearch 접속

`config.json`의 `managed_opensearch_dashboards_url`로 접속합니다 (예: `https://<domain-endpoint>/_dashboards`).

- **Username:** `managed_opensearch_dashboards_user` (`admin`)
- **Password:** `managed_opensearch_dashboards_password`

브라우저에서 URL만 열면 Dashboards 로그인 화면(HTTP 302)으로 이동합니다. `admin` / 설치 시 비밀번호로 로그인하세요. IAM SigV4 RAG API 호출은 계정 root Principal 정책으로 계속 동작합니다.



### 인프라 삭제

```bash
python3 uninstaller.py
```
