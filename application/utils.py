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
favorite_tools_path = os.path.join(os.path.dirname(config_path), "favorite_tools.json")
SKILLS_DIR = os.path.join(workingDir, "skills")
SESSION_STORAGE_DIR = os.environ.get(
    "SESSION_STORAGE_DIR",
    os.path.join(workingDir, ".session_storage"),
)


def sanitize_user_path_segment(user_id: str | None) -> str | None:
    """Return a safe single path segment for per-user workspace folders, or None."""
    if not user_id:
        return None
    raw = str(user_id).strip()
    if raw.startswith("v1.") and raw.count(".") >= 2:
        logger.warning("Refusing signed session token as artifacts path segment")
        return None
    if len(raw) > 128:
        logger.warning("Refusing oversized user_id as artifacts path segment")
        return None
    segment = (
        raw
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )
    return segment or None


def get_user_artifacts_dir(user_id: str | None) -> str:
    segment = sanitize_user_path_segment(user_id) or "default"
    return os.path.join(SESSION_STORAGE_DIR, segment, "artifacts")


def ensure_user_artifacts_dir(user_id: str | None) -> str:
    segment = sanitize_user_path_segment(user_id)
    if not segment:
        raise ValueError(
            "Invalid user_id for artifacts path; expected a plain user id, "
            "not a signed session cookie"
        )
    artifacts_dir = os.path.join(SESSION_STORAGE_DIR, segment, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def get_user_skills_dir(user_id: str | None) -> str:
    segment = sanitize_user_path_segment(user_id) or "default"
    return os.path.join(SESSION_STORAGE_DIR, segment, "skills")


def ensure_user_skills_dir(user_id: str | None) -> str:
    segment = sanitize_user_path_segment(user_id)
    if not segment:
        raise ValueError(
            "Invalid user_id for skills path; expected a plain user id, "
            "not a signed session cookie"
        )
    skills_dir = os.path.join(SESSION_STORAGE_DIR, segment, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    return skills_dir


def get_user_skills_list_path(user_id: str | None) -> str:
    segment = sanitize_user_path_segment(user_id) or "default"
    return os.path.join(SESSION_STORAGE_DIR, segment, "skills.list")


def _list_skill_dir_names(skills_dir: str) -> list[str]:
    if not os.path.isdir(skills_dir):
        return []
    names: list[str] = []
    try:
        entries = sorted(os.listdir(skills_dir))
    except OSError as e:
        logger.warning("Failed to list skills directory %s: %s", skills_dir, e)
        return []
    for entry in entries:
        if os.path.isfile(os.path.join(skills_dir, entry, "SKILL.md")):
            names.append(entry)
    return names


def _load_skills_list_file(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("Failed to read skills.list %s: %s", path, e)
        return []


def _seed_skill_names(user_id: str | None) -> list[str]:
    default_path = os.path.join(workingDir, "skills.list")
    builtin = _load_skills_list_file(default_path)
    user_skills = _list_skill_dir_names(get_user_skills_dir(user_id))
    merged: list[str] = []
    seen: set[str] = set()
    for name in builtin + user_skills:
        if name not in seen:
            merged.append(name)
            seen.add(name)
    return merged


def write_user_skills_list(user_id: str | None, names: list[str] | None = None) -> str:
    ensure_user_skills_dir(user_id)
    path = get_user_skills_list_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged = names if names is not None else _seed_skill_names(user_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(merged) + ("\n" if merged else ""))
    return path


def _builtin_skill_exists(name: str) -> bool:
    return os.path.isfile(os.path.join(workingDir, "skills", name, "SKILL.md"))


def _user_skill_exists(user_id: str | None, name: str) -> bool:
    return os.path.isfile(
        os.path.join(get_user_skills_dir(user_id), name, "SKILL.md")
    )


def ensure_user_skills_list(user_id: str | None) -> str:
    ensure_user_skills_dir(user_id)
    path = get_user_skills_list_path(user_id)
    if not os.path.isfile(path):
        return write_user_skills_list(user_id)

    existing = _load_skills_list_file(path)
    kept = [
        name
        for name in existing
        if _builtin_skill_exists(name) or _user_skill_exists(user_id, name)
    ]
    seen = set(kept)
    default_path = os.path.join(workingDir, "skills.list")
    candidates = _load_skills_list_file(default_path) + _list_skill_dir_names(
        get_user_skills_dir(user_id)
    )
    appended = [name for name in candidates if name not in seen]
    updated = kept + appended
    if updated != existing:
        return write_user_skills_list(user_id, updated)
    return path


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



def load_favorite_tools() -> dict[str, list[str]]:
    """Load favorite tool defaults for initial selections."""
    fallback = {"MCP": [], "SKILL": []}
    try:
        with open(favorite_tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("favorite_tools.json not found: %s", favorite_tools_path)
        return fallback
    except Exception as e:
        logger.warning("Failed to load favorite_tools.json: %s", e)
        return fallback

    if not isinstance(data, dict):
        return fallback

    favorites: dict[str, list[str]] = {}
    for key in ("MCP", "SKILL"):
        values = data.get(key, [])
        if isinstance(values, list):
            favorites[key] = [v for v in values if isinstance(v, str) and v.strip()]
        else:
            favorites[key] = []
    return favorites


def save_favorite_tools(
    *, skills: list[str] | None = None, mcp_servers: list[str] | None = None
) -> dict[str, list[str]]:
    """Persist favorite tool defaults in favorite_tools.json."""
    favorites = load_favorite_tools()
    if skills is not None:
        favorites["SKILL"] = [v for v in skills if isinstance(v, str) and v.strip()]
    if mcp_servers is not None:
        favorites["MCP"] = [v for v in mcp_servers if isinstance(v, str) and v.strip()]

    with open(favorite_tools_path, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)
    return favorites


def get_initial_tool_defaults() -> tuple[list[str], list[str]]:
    """Return initial skill/MCP defaults from favorite_tools.json."""
    favorite_tools = load_favorite_tools()
    default_skills = favorite_tools.get("SKILL") or []
    default_mcp_servers = favorite_tools.get("MCP") or []
    return default_skills, default_mcp_servers

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


def docs_s3_prefix(project: str | None = None) -> str:
    """Return the S3 docs prefix used by this project (historically ``docs``)."""
    _ = project
    configured = (config.get("s3_docs_prefix") or "docs/").strip().strip("/")
    return configured or "docs"


def upload_to_s3(
    file_bytes: bytes,
    file_name: str,
    user_id: str | None = None,
) -> dict | None:
    """Upload a file to S3 under docs/ (or images/) and return metadata."""
    from urllib import parse

    if not s3_bucket:
        logger.error("s3_bucket is not configured")
        return None

    try:
        s3_client = boto3.client(service_name="s3", region_name=bedrock_region)
        content_type = get_contents_type(file_name)
        logger.info("content_type: %s", content_type)

        prefix = (
            "images"
            if isinstance(content_type, str) and content_type.startswith("image/")
            else docs_s3_prefix()
        )
        user_segment = sanitize_user_path_segment(user_id)
        if user_segment:
            s3_key = f"{prefix}/{user_segment}/{file_name}"
            relative_url_path = f"{prefix}/{parse.quote(user_segment)}/{parse.quote(file_name)}"
        else:
            s3_key = f"{prefix}/{file_name}"
            relative_url_path = f"{prefix}/{parse.quote(file_name)}"
        user_meta = {"content_type": content_type}

        put_params = {
            "Bucket": s3_bucket,
            "Key": s3_key,
            "Metadata": user_meta,
            "Body": file_bytes,
        }
        if content_type and content_type != "no info":
            put_params["ContentType"] = content_type
        if content_type == "application/pdf":
            put_params["ContentDisposition"] = "inline"

        response = s3_client.put_object(**put_params)
        logger.info("upload response: %s", response)

        url = None
        if sharing_url:
            url = f"{sharing_url.rstrip('/')}/{relative_url_path}"

        return {
            "file_name": file_name,
            "s3_key": s3_key,
            "content_type": content_type,
            "url": url,
        }
    except Exception:
        logger.error("Error uploading to S3: %s", traceback.format_exc())
        return None
