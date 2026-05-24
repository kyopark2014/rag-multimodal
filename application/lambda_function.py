import json
import boto3
import os
import info

from botocore.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_aws import ChatBedrock
from pydantic.v1 import BaseModel, Field
from multiprocessing import Process, Pipe
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings

bedrock_region = os.environ.get('bedrock_region')
projectName = os.environ.get('projectName')
path = os.environ.get('sharing_url')

enableHybridSearch = "Enable"
enableParentDocumentRetrival = "Enable"

opensearch_url = os.environ.get('opensearch_url')
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

index_name = projectName
session = boto3.Session(region_name=bedrock_region)
credentials = session.get_credentials()

# AWS4Auth settings (for AWS managed OpenSearch)
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    bedrock_region,
    'es',  # OpenSearch service uses 'es'
    session_token=credentials.token
)

os_client = OpenSearch(
    hosts=[{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress=True,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

def lexical_search(query, top_k):
    # lexical search (keyword)
    min_match = 0
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f'{min_match}%',
                                "operator":  "or",
                            }
                        }
                    },
                ],
                "filter": [
                ]
            }
        }
    }

    response = os_client.search(
        body=query,
        index=index_name
    )
    # print('lexical query result: ', json.dumps(response))
        
    docs = []
    for i, document in enumerate(response['hits']['hits']):
        if i>=top_k: 
            break
                    
        excerpt = document['_source']['text']
        
        name = document['_source']['metadata']['name']
        # print('name: ', name)

        page = ""
        if "page" in document['_source']['metadata']:
            page = document['_source']['metadata']['page']
        
        url = ""
        if "url" in document['_source']['metadata']:
            url = document['_source']['metadata']['url']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'page': page,
                        'from': 'lexical'
                    },
                )
            )
    
    for i, doc in enumerate(docs):
        #print('doc: ', doc)
        #print('doc content: ', doc.page_content)
        
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        print(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")   
        
    return docs

selected_embedding = 0
def get_embedding():
    LLM_embedding = [
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        }
    ]
    
    global selected_embedding
    embedding_profile = LLM_embedding[selected_embedding]
    bedrock_region =  embedding_profile['bedrock_region']
    model_id = embedding_profile['model_id']
    print(f"selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}")
    
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
    
    if multi_region=='Enable':
        selected_embedding = selected_embedding + 1
        if selected_embedding == len(LLM_embedding):
            selected_embedding = 0
    else:
        selected_embedding = 0

    return bedrock_embedding

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('url: ', metadata['url'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    url = ""
    if "url" in metadata:
        url = metadata['url']
    
    return source['text'], metadata['name'], url

def retrieve_documents_from_opensearch(query, top_k):
    print(f"###### retrieve_documents_from_opensearch ######")

    # Vector Search
    bedrock_embedding = get_embedding()       

    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name=index_name,  
        is_aoss = False,
        #engine="faiss",  # default: nmslib
        embedding_function=bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=awsauth,
        connection_class=RequestsHttpConnection
    )  
    
    relevant_docs = []
    if enableParentDocumentRetrival == 'Enable':
        result = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k*2,  
            search_type="script_scoring",
            pre_filter={"term": {"metadata.doc_level": "child"}}
        )
        print(f"result: {result}")
                
        relevant_documents = []
        docList = []
        for re in result:
            if 'parent_doc_id' in re[0].metadata:
                parent_doc_id = re[0].metadata['parent_doc_id']
                doc_level = re[0].metadata['doc_level']
                print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                        
                if doc_level == 'child':
                    if parent_doc_id in docList:
                        print(f"duplicated")
                    else:
                        relevant_documents.append(re)
                        docList.append(parent_doc_id)                        
                        if len(relevant_documents)>=top_k:
                            break
                                    
        # print('relevant_documents: ', relevant_documents)    
        for i, doc in enumerate(relevant_documents):
            if len(doc[0].page_content)>=100:
                text = doc[0].page_content[:100]
            else:
                text = doc[0].page_content            
            print(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")

        for i, document in enumerate(relevant_documents):
            print(f"## Document(opensearch-vector) {i+1}: {document}")
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
            
            content, name, url = get_parent_content(parent_doc_id) # use pareant document
            #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {content}")
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k
        )
        
        for i, document in enumerate(relevant_documents):
            print(f"## Document(opensearch-vector) {i+1}: {document}")   
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    # print('the number of docs (vector search): ', len(relevant_docs))

    # Lexical Search
    if enableHybridSearch == 'Enable':
        relevant_docs += lexical_search(query, top_k)    

    return relevant_docs
   
selected_chat = 0
multi_region = 'Disable'
def get_chat(models, extended_thinking):
    global selected_chat

    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    number_of_models = len(models)

    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
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
    if extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        print(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,            
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    else:
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def get_parallel_processing_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
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
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    return chat
                
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print(f"---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print(f"--GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()

def grade_documents_using_parallel_processing(models, question, documents):
    global selected_chat

    number_of_models = len(models)    
    filtered_docs = []    
    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected_chat))
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
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(model_name, question, documents):
    print(f"###### grade_documents ######")
    print(f"start grading...")

    models = info.get_model_info(model_name)
    
    filtered_docs = []
    if multi_region == 'Enable':  # parallel processing        
        filtered_docs = grade_documents_using_parallel_processing(models, question, documents)

    else:
        # Score each doc    
        llm = get_chat(models, extended_thinking="Disable")
        retrieval_grader = get_retrieval_grader(llm)
        for i, doc in enumerate(documents):
            # print('doc: ', doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            # print("score: ", score)
            
            grade = score.binary_score
            # print("grade: ", grade)
            
            if grade.lower() == "yes": # Document relevant
                print(f"---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            
            else: # Document not relevant
                print(f"---GRADE: DOCUMENT NOT RELEVANT---")
                continue
    
    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updated_docs = len(updated_docs)   
    
    if length_original == length_updated_docs:
        print('no duplication')
    else:
        print('length of updated relevant_docs: ', length_updated_docs)
    
    return updated_docs

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")

def lambda_handler(event, context):
    print('event: ', event)
    
    function = event['function']
    print('function: ', function)

    keyword = event.get('keyword')
    print('keyword: ', keyword)

    top_k = event.get('top_k')
    print('top_k: ', top_k)
    
    grading = event.get('grading')
    print('grading: ', grading)

    model_name = event.get('model_name')
    print('model_name: ', model_name)

    global multi_region
    multi_region = event.get('multi_region')
    print('multi_region: ', multi_region)

    global contentList
    contentList = []

    docs = []
    if function == 'search_rag':
        print('keyword: ', keyword)        

        relevant_docs = retrieve_documents_from_opensearch(keyword, top_k)

        if grading == "Enable":            
            filtered_docs = grade_documents(model_name, keyword, relevant_docs)  # grade documents            
            filtered_docs = check_duplication(filtered_docs)  # check duplication
            docs = filtered_docs            
        else:
            docs = relevant_docs

        json_docs = []
        for doc in docs:
            print('doc: ', doc)

            json_docs.append({
                "contents": doc.page_content,              
                "reference": {
                    "url": doc.metadata["url"],                   
                    "title": doc.metadata["name"],
                    "from": doc.metadata["from"]
                }
            })

        print('json_docs: ', json_docs)
        
    return {
        'response': json.dumps(json_docs, ensure_ascii=False)
    }
