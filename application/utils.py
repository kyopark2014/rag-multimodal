import logging
import sys
import json
import traceback
import boto3
import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("utils")

workingDir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(workingDir, "config.json")
    
def load_config():
    config = None

    try: 
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}

        projectName = "rag-multimodal"
        session = boto3.Session()
        region = session.region_name
        config['region'] = region
        config['projectName'] = projectName
        
        sts = boto3.client("sts")
        response = sts.get_caller_identity()
        accountId = response["Account"]
        config['accountId'] = accountId
        config['s3_bucket'] = f'storage-for-{projectName}-{accountId}-{region}'
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)    
    return config

config = load_config()

accountId = config.get('accountId')
if not accountId:
    sts = boto3.client("sts")
    response = sts.get_caller_identity()
    accountId = response["Account"]
    config['accountId'] = accountId
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

bedrock_region = config.get('region', 'us-west-2')
logger.info(f"bedrock_region: {bedrock_region}")
projectName = config.get('projectName', 'mop')
logger.info(f"projectName: {projectName}")


def persist_config_updates(updates):
    """Merge values fetched from Secrets Manager into config and write config.json."""
    global config
    if not updates:
        return
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        s = value.strip() if isinstance(value, str) else str(value)
        if not s:
            continue
        if config.get(key) != s:
            config[key] = s
            changed = True
    if not changed:
        return
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved Secrets Manager values to config.json: %s",
            ", ".join(str(k) for k in updates if updates.get(k)),
        )
    except Exception as e:
        logger.warning("Failed to write config.json: %s", e)


def get_contents_type(file_name):
    if file_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif file_name.lower().endswith((".pdf")):
        content_type = "application/pdf"
    elif file_name.lower().endswith((".txt")):
        content_type = "text/plain"
    elif file_name.lower().endswith((".csv")):
        content_type = "text/csv"
    elif file_name.lower().endswith((".ppt", ".pptx")):
        content_type = "application/vnd.ms-powerpoint"
    elif file_name.lower().endswith((".doc", ".docx")):
        content_type = "application/msword"
    elif file_name.lower().endswith((".xls")):
        content_type = "application/vnd.ms-excel"
    elif file_name.lower().endswith((".py")):
        content_type = "text/x-python"
    elif file_name.lower().endswith((".js")):
        content_type = "application/javascript"
    elif file_name.lower().endswith((".md")):
        content_type = "text/markdown"
    elif file_name.lower().endswith((".json")):
        content_type = "application/json"
    elif file_name.lower().endswith((".png")):
        content_type = "image/png"
    else:
        content_type = "no info"    
    return content_type

def load_mcp_env():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "r", encoding="utf-8") as f:
        mcp_env = json.load(f)
    return mcp_env

def save_mcp_env(mcp_env):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "w", encoding="utf-8") as f:
        json.dump(mcp_env, f)

# api key to get information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)

# Tavily Search API key: prefer config.json, else Secrets Manager
tavily_api_wrapper = ""
tavily_key = (config.get("tavily_api_key") or "").strip()
if tavily_key:
    tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
    os.environ["TAVILY_API_KEY"] = tavily_key
else:
    try:
        get_tavily_api_secret = secretsmanager.get_secret_value(
            SecretId=f"tavilyapikey-{projectName}"
        )
        secret = json.loads(get_tavily_api_secret["SecretString"])

        if "tavily_api_key" in secret:
            tavily_key = (secret["tavily_api_key"] or "").strip()

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            os.environ["TAVILY_API_KEY"] = tavily_key
            persist_config_updates({"tavily_api_key": tavily_key})
        else:
            logger.info("tavily_key is required.")
    except Exception as e:
        logger.info(f"Tavily credential is required: {e}")
        pass

def sanitize_data_source_name(name):
    """
    Sanitize a name to comply with AWS Bedrock data source name pattern:
    ([0-9a-zA-Z][_-]?){1,100}
    - Pattern means: alphanumeric, optionally followed by underscore or hyphen, repeated 1-100 times
    - Cannot have consecutive underscores or hyphens
    - Must start with alphanumeric
    """
    import re
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    sanitized = re.sub(r'[^0-9a-zA-Z_-]', '', name)
    
    # Replace consecutive underscores/hyphens with single hyphen
    # This ensures the pattern [0-9a-zA-Z][_-]? is followed correctly
    sanitized = re.sub(r'[_-]{2,}', '-', sanitized)
    
    # Ensure it starts with alphanumeric character
    if sanitized and not sanitized[0].isalnum():
        sanitized = 'ds' + sanitized
    
    # Remove trailing hyphens/underscores (they must be followed by alphanumeric per pattern)
    sanitized = sanitized.rstrip('_-')
    
    # Ensure it's not empty and limit to 100 characters
    if not sanitized:
        sanitized = 'datasource'
    
    # Final validation: ensure it matches the pattern exactly
    pattern = re.compile(r'^([0-9a-zA-Z][_-]?){1,100}$')
    if not pattern.match(sanitized):
        # If still doesn't match, create a safe default name
        # Use project name or create a simple alphanumeric name
        safe_name = re.sub(r'[^0-9a-zA-Z]', '', name.lower())
        if not safe_name:
            safe_name = 'datasource'
        sanitized = safe_name[:100]
    
    return sanitized[:100]

knowledge_base_id = config.get('knowledge_base_id')
data_source_id = config.get('data_source_id')
region = config.get('region', 'us-west-2')
s3_bucket = config.get('s3_bucket', f'storage-for-rag-project-{accountId}-{region}')
sharing_url = config.get('sharing_url', '')

def update_sharing_url():
    """Look up CloudFront distribution domain for this project and save as sharing_url."""
    try:
        cf_client = boto3.client('cloudfront', region_name=region)
        paginator = cf_client.get_paginator('list_distributions')
        target_origin_id = f"s3-{projectName}"

        for page in paginator.paginate():
            dist_list = page.get('DistributionList', {})
            for dist in dist_list.get('Items', []):
                origins = dist.get('Origins', {}).get('Items', [])
                for origin in origins:
                    if origin['Id'] == target_origin_id:
                        domain = dist['DomainName']
                        url = f"https://{domain}"
                        logger.info(f"sharing_url found: {url}")
                        config['sharing_url'] = url
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=2)
                        return url
        logger.warning(f"CloudFront distribution with origin '{target_origin_id}' not found")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"Failed to look up sharing_url: {err_msg}")
    return ''

if not sharing_url:
    sharing_url = update_sharing_url()

def update_rag_info():
    knowledge_base_id = None
    data_source_id = None
    try: 
        client = boto3.client(
            service_name='bedrock-agent',
            region_name=region
        )

        response = client.list_knowledge_bases(
            maxResults=50
        )
        logger.info(f"(list_knowledge_bases) response: {response}")
        
        knowledge_base_name = projectName
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    logger.info(f"knowledge_base_id: {knowledge_base_id}")

        if not knowledge_base_id:
            logger.warning(f"Knowledge Base not found for project: {knowledge_base_name}")
            return knowledge_base_id, data_source_id

        if not s3_bucket:
            logger.warning(f"s3_bucket is not configured, skipping data source lookup")
            return knowledge_base_id, data_source_id

        response = client.list_data_sources(
            knowledgeBaseId=knowledge_base_id,
            maxResults=10
        )        
        logger.info(f"(list_data_sources) response: {response}")
        
        data_source_name = sanitize_data_source_name(s3_bucket)
        if 'dataSourceSummaries' in response:
            for data_source in response['dataSourceSummaries']:
                logger.info(f"data_source: {data_source}")
                if data_source['name'] == data_source_name:
                    data_source_id = data_source['dataSourceId']
                    logger.info(f"data_source_id: {data_source_id}")
                    break    
        
        # save config
        config['knowledge_base_id'] = knowledge_base_id
        config['data_source_id'] = data_source_id
        config['s3_bucket'] = s3_bucket
        config['region'] = region
        config['projectName'] = projectName
        config['accountId'] = accountId
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return knowledge_base_id, data_source_id

if not knowledge_base_id or not data_source_id:
    knowledge_base_id, data_source_id = update_rag_info()

def sync_data_source():
    if knowledge_base_id and data_source_id:
        try:
            bedrock_client = boto3.client(
                service_name='bedrock-agent',
                region_name=region
            )
                
            response = bedrock_client.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id
            )
            logger.info(f"(start_ingestion_job) response: {response}")
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
