import logging
import sys
import json
import traceback
import boto3
import os
from botocore.exceptions import ClientError
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

def _opensearch_domain_endpoint(domain_status):
    endpoint = domain_status.get("Endpoint")
    if endpoint:
        return f"https://{endpoint}"
    return None


def update_rag_info():
    """Discover managed OpenSearch domain endpoint and persist to config.json."""
    managed_opensearch_url = config.get("managed_opensearch_url")
    domain_name = projectName
    try:
        es_client = boto3.client("es", region_name=region)
        response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
        logger.info(f"(describe_elasticsearch_domain) domain: {domain_name}")

        domain_status = response.get("DomainStatus", {})
        endpoint_url = _opensearch_domain_endpoint(domain_status)
        if not endpoint_url:
            logger.warning(
                f"OpenSearch domain '{domain_name}' has no endpoint yet "
                f"(created={domain_status.get('Created')}, "
                f"processing={domain_status.get('Processing')})"
            )
            return managed_opensearch_url

        updates = {
            "managed_opensearch_url": endpoint_url,
            "s3_bucket": s3_bucket,
            "region": region,
            "projectName": projectName,
            "accountId": accountId,
        }
        arn = domain_status.get("ARN")
        if arn:
            updates["managed_opensearch_arn"] = arn

        if managed_opensearch_url != endpoint_url:
            logger.info(f"managed_opensearch_url: {endpoint_url}")
            config.update(updates)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            managed_opensearch_url = endpoint_url

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            logger.warning(f"OpenSearch domain not found for project: {domain_name}")
        else:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return managed_opensearch_url


managed_opensearch_url = config.get("managed_opensearch_url")
if not managed_opensearch_url:
    managed_opensearch_url = update_rag_info()
