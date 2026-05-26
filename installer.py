#!/usr/bin/env python3
"""
AWS Infrastructure Installer using boto3
This script creates AWS infrastructure resources equivalent to the CDK stack.
"""

import boto3
import getpass
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from botocore.exceptions import ClientError
from typing import Dict, Iterable, List, Optional

import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Configuration
project_name = "rag-multimodal" # at least 3 characters
region = "us-west-2"
cloudfront_comment = "CloudFront-for-rag-project"

sts_client = boto3.client("sts", region_name=region)
account_id = sts_client.get_caller_identity()["Account"]

# Initialize boto3 clients
s3_client = boto3.client("s3", region_name=region)
secrets_client = boto3.client("secretsmanager", region_name=region)
cloudfront_client = boto3.client("cloudfront", region_name=region)
es_client = boto3.client("es", region_name=region)
opensearch_client = boto3.client("opensearch", region_name=region)
iam_client = boto3.client("iam", region_name=region)
lambda_client = boto3.client("lambda", region_name=region)

# OpenSearch managed domain (used by application/opensearch.py as managed_opensearch_url)
opensearch_domain_name = project_name
OPENSEARCH_MASTER_USERNAME = "admin"
# application/config.json holds the (optional) saved Dashboards master password
# so we can reuse it for FGAC role mappings without prompting on every run.
APPLICATION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "application", "config.json"
)
OPENSEARCH_CONFIG_PASSWORD_KEY = "managed_opensearch_dashboards_password"
# FGAC security role used for the installer + app + lambda SigV4 callers.
OPENSEARCH_BACKEND_ROLE_NAME = "all_access"

bucket_name = f"storage-for-rag-project-{account_id}-{region}"

# S3 docs prefix (application/chat.py, multimodal.py) and lambda-s3-event-manager
S3_DOCS_PREFIX = "docs/"
LAMBDA_S3_EVENT_FUNCTION_NAME = f"lambda-s3-event-manager-for-{project_name}"
LAMBDA_S3_EVENT_ROLE_NAME = f"role-lambda-s3-event-manager-for-{project_name}-{region}"
LAMBDA_S3_EVENT_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lambda-s3-event-manager",
)
S3_EVENT_NOTIFICATION_ID = f"{project_name}-docs-s3-event"
# ObjectCreated: upload/copy; ObjectRemoved: delete (required for index cleanup pipelines)
S3_LAMBDA_EVENTS = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f"installer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


def create_s3_bucket() -> str:
    """Create S3 bucket with CORS configuration."""
    logger.info(f"[1/3] Creating S3 bucket: {bucket_name}")
    
    try:
        # Create bucket
        logger.debug(f"Creating bucket in region: {region}")
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        logger.debug("Bucket created successfully")
        
        # Configure bucket
        logger.debug("Configuring public access block")
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True
            }
        )
        
        # Set CORS configuration
        logger.debug("Setting CORS configuration")
        cors_configuration = {
            "CORSRules": [
                {
                    "AllowedHeaders": ["*"],
                    "AllowedMethods": ["GET", "POST", "PUT"],
                    "AllowedOrigins": ["*"]
                }
            ]
        }
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )
        
        # Enable versioning (set to false means suspend)
        logger.debug("Configuring versioning")
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Suspended"}
        )
        
        # Create docs folder
        logger.debug("Creating docs folder")
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key="docs/",
                Body=b""
            )
            logger.debug("docs folder created successfully")
        except ClientError as e:
            logger.warning(f"Failed to create docs folder: {e}")
        
        logger.info(f"✓ S3 bucket created successfully: {bucket_name}")
        return bucket_name
    
    except ClientError as e:
        if e.response["Error"]["Code"] in ["BucketAlreadyExists", "BucketAlreadyOwnedByYou"]:
            logger.warning(f"S3 bucket already exists: {bucket_name}")
            # Create docs folder if bucket already exists
            logger.debug("Creating docs folder in existing bucket")
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key="docs/",
                    Body=b""
                )
                logger.debug("docs folder created successfully")
            except ClientError as folder_error:
                if folder_error.response["Error"]["Code"] != "NoSuchBucket":
                    logger.warning(f"Failed to create docs folder: {folder_error}")
            return bucket_name
        logger.error(f"Failed to create S3 bucket: {e}")
        raise


def _opensearch_domain_endpoint(domain_status: Dict) -> Optional[str]:
    endpoint = domain_status.get("Endpoint")
    if endpoint:
        return f"https://{endpoint}"
    return None


def _domain_engine_version(domain_status: Dict) -> Optional[str]:
    """OpenSearch engine version (es API: ElasticsearchVersion, opensearch API: EngineVersion)."""
    return domain_status.get("EngineVersion") or domain_status.get("ElasticsearchVersion")


def _is_opensearch_domain_usable(domain_status: Dict) -> bool:
    """
    Domain can serve API requests.

    Processing may stay True during package install or other config changes;
    in that case the endpoint is still usable and we should not block for hours.
    """
    if domain_status.get("Deleted"):
        return False
    if not domain_status.get("Created", False):
        return False
    return _opensearch_domain_endpoint(domain_status) is not None


def _is_opensearch_domain_provisioning(domain_status: Dict) -> bool:
    """Initial domain creation (no endpoint yet)."""
    return not _is_opensearch_domain_usable(domain_status)


def _wait_for_opensearch_domain(domain_name: str, max_wait: int = 3600, poll_interval: int = 30) -> Dict:
    """Wait until the managed OpenSearch domain has an endpoint (first-time create only)."""
    logger.info("  Waiting for OpenSearch domain to become available (this may take 20-40 minutes)...")
    waited = 0
    while waited < max_wait:
        response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
        status = response["DomainStatus"]
        endpoint_url = _opensearch_domain_endpoint(status)
        processing = status.get("Processing", True)
        created = status.get("Created", False)
        domain_status = status.get("DomainProcessingStatus") or status.get("ProcessingStatus")

        if waited % 120 == 0 and waited > 0:
            logger.info(
                f"  Domain status: created={created}, processing={processing}, "
                f"processing_status={domain_status}, waited={waited}s"
            )

        if _is_opensearch_domain_usable(status):
            if processing:
                logger.info(
                    f"  ✓ OpenSearch domain endpoint ready (config update in progress): "
                    f"{endpoint_url}"
                )
            else:
                logger.info(f"  ✓ OpenSearch domain is active: {endpoint_url}")
            return status

        time.sleep(poll_interval)
        waited += poll_interval

    raise TimeoutError(
        f"Timeout waiting for OpenSearch domain '{domain_name}' "
        f"after {max_wait} seconds"
    )


def _wait_for_opensearch_domain_config(
    domain_name: str, max_wait: int = 3600, poll_interval: int = 30
) -> Dict:
    """Wait until an OpenSearch domain configuration change finishes processing."""
    logger.info("  Waiting for OpenSearch domain configuration update...")
    waited = 0
    while waited < max_wait:
        response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
        status = response["DomainStatus"]
        if not status.get("Processing", False):
            logger.info("  ✓ OpenSearch domain configuration update complete")
            return status
        if waited % 120 == 0:
            proc = status.get("DomainProcessingStatus") or status.get("ProcessingStatus")
            logger.info(
                f"  Config update in progress (status={proc}), waited={waited}s"
            )
        time.sleep(poll_interval)
        waited += poll_interval
    raise TimeoutError(
        f"Timeout waiting for OpenSearch config update on '{domain_name}' "
        f"after {max_wait} seconds"
    )


def _validate_opensearch_master_password(password: str) -> Optional[str]:
    """Return an error message if the password does not meet OpenSearch FGAC rules."""
    if len(password) < 8 or len(password) > 128:
        return "Password must be 8–128 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    return None


def _opensearch_domain_fgac_status() -> Optional[bool]:
    """Return FGAC enabled state, or None if the domain does not exist yet."""
    try:
        response = es_client.describe_elasticsearch_domain(
            DomainName=opensearch_domain_name
        )
        return _is_fgac_enabled(response["DomainStatus"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return None
        raise


def _load_saved_opensearch_master_password() -> str:
    """Read the master password previously saved to application/config.json (if any)."""
    try:
        with open(APPLICATION_CONFIG_PATH, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    return data.get(OPENSEARCH_CONFIG_PASSWORD_KEY, "") or ""


def prompt_opensearch_master_password_if_needed() -> str:
    """Return the master password, reusing or re-prompting as needed.

    The password is required even on re-runs because FGAC rejects SigV4
    callers until we map them as backend roles via the OpenSearch security
    API (which only the admin user can call).
    """
    fgac_status = _opensearch_domain_fgac_status()
    if fgac_status is True:
        saved = _load_saved_opensearch_master_password()
        if saved:
            logger.info(
                f"OpenSearch FGAC already enabled; "
                f"reusing saved Dashboards password from "
                f"{os.path.relpath(APPLICATION_CONFIG_PATH)}"
            )
            return saved
        logger.info(
            f"OpenSearch FGAC already enabled but no saved password found in "
            f"{os.path.relpath(APPLICATION_CONFIG_PATH)}; "
            f"re-enter the existing '{OPENSEARCH_MASTER_USERNAME}' password "
            f"so the installer can map IAM principals as backend_roles."
        )
        return _prompt_opensearch_master_password_single()
    return prompt_opensearch_master_password()


def _prompt_opensearch_master_password_single() -> str:
    """Prompt once for the existing master password (no confirm step)."""
    logger.info("")
    logger.info(f"  Username: {OPENSEARCH_MASTER_USERNAME}")
    while True:
        password = getpass.getpass(
            f"Enter existing password for '{OPENSEARCH_MASTER_USERNAME}': "
        )
        if password:
            return password
        logger.warning("Password cannot be empty. Try again.")


def _opensearch_basic_auth_works(endpoint_url: str, password: str) -> bool:
    """Probe whether (admin, password) is accepted by the OpenSearch security plugin."""
    if not password:
        return False
    try:
        resp = requests.get(
            f"{endpoint_url.rstrip('/')}/_plugins/_security/api/account",
            auth=(OPENSEARCH_MASTER_USERNAME, password),
            timeout=15,
        )
    except requests.RequestException as exc:
        logger.warning(f"  OpenSearch reachability check failed: {exc}")
        return False
    if resp.status_code == 200:
        return True
    if resp.status_code in (401, 403):
        return False
    logger.warning(
        f"  Unexpected response while probing OpenSearch admin auth "
        f"({resp.status_code}): {resp.text[:200]}"
    )
    return False


def _reset_opensearch_master_password(domain_name: str) -> str:
    """Reset the OpenSearch FGAC master user password via AWS API (admin-only)."""
    new_password = prompt_opensearch_master_password()
    logger.info(
        f"  Resetting OpenSearch FGAC master password for "
        f"'{OPENSEARCH_MASTER_USERNAME}' via AWS API..."
    )
    es_client.update_elasticsearch_domain_config(
        DomainName=domain_name,
        AdvancedSecurityOptions={
            "Enabled": True,
            "InternalUserDatabaseEnabled": True,
            "MasterUserOptions": {
                "MasterUserName": OPENSEARCH_MASTER_USERNAME,
                "MasterUserPassword": new_password,
            },
        },
    )
    _wait_for_opensearch_domain_config(domain_name)
    logger.info(
        f"  ✓ Master password reset for '{OPENSEARCH_MASTER_USERNAME}'"
    )
    return new_password


def ensure_opensearch_master_password_works(
    endpoint_url: str,
    domain_name: str,
    candidate_password: str,
) -> str:
    """Validate the master password against the live cluster; reset on failure.

    Returns a working master password (possibly different from the input).
    """
    if _opensearch_basic_auth_works(endpoint_url, candidate_password):
        return candidate_password

    logger.warning(
        f"  '{OPENSEARCH_MASTER_USERNAME}' authentication rejected by "
        f"OpenSearch (likely wrong password)."
    )
    for attempt in range(2):
        retry = getpass.getpass(
            f"Re-enter password for '{OPENSEARCH_MASTER_USERNAME}' "
            f"(attempt {attempt + 1}/2, press Enter to reset via AWS API): "
        )
        if not retry:
            break
        if _opensearch_basic_auth_works(endpoint_url, retry):
            logger.info("  ✓ OpenSearch admin authentication succeeded")
            return retry
        logger.warning("  Still rejected by OpenSearch.")

    logger.info(
        f"  Falling back to AWS-side password reset for "
        f"'{OPENSEARCH_MASTER_USERNAME}' (set a new password below)."
    )
    new_password = _reset_opensearch_master_password(domain_name)
    if not _opensearch_basic_auth_works(endpoint_url, new_password):
        raise RuntimeError(
            "OpenSearch admin password reset succeeded but the cluster still "
            "rejects the new credentials — try again in a minute or check "
            "the domain status in the AWS console."
        )
    return new_password


def prompt_opensearch_master_password() -> str:
    """Prompt for the OpenSearch Dashboards master user (admin) password."""
    logger.info("")
    logger.info("OpenSearch Dashboards master user")
    logger.info(f"  Username: {OPENSEARCH_MASTER_USERNAME}")
    logger.info(
        "  Password: 8–128 chars, at least one uppercase, lowercase, and digit"
    )
    while True:
        password = getpass.getpass(
            f"Enter password for '{OPENSEARCH_MASTER_USERNAME}': "
        )
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            logger.warning("Passwords do not match. Try again.")
            continue
        error = _validate_opensearch_master_password(password)
        if error:
            logger.warning(error)
            continue
        return password


def _build_opensearch_access_policies() -> Dict:
    """
    Domain access policy for IAM (SigV4) and Dashboards (HTTP basic via FGAC).

    FGAC enforces authorization; the wildcard principal lets unsigned browser
    requests reach the cluster so Dashboards can prompt for admin credentials.
    """
    resource = (
        f"arn:aws:es:{region}:{account_id}:domain/{opensearch_domain_name}/*"
    )
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                "Action": "es:*",
                "Resource": resource,
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "es:*",
                "Resource": resource,
            },
        ],
    }


def _access_policy_allows_dashboards(access_policies: str) -> bool:
    try:
        policy = json.loads(access_policies) if access_policies else {}
    except json.JSONDecodeError:
        return False
    for statement in policy.get("Statement", []):
        principal = statement.get("Principal") or {}
        aws_principal = principal.get("AWS")
        if aws_principal in ("*", ["*"]):
            return True
    return False


def _disable_opensearch_anonymous_auth_migration(domain_name: str) -> None:
    """Turn off FGAC anonymous migration so the domain policy can allow Dashboards."""
    response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
    fgac = response["DomainStatus"].get("AdvancedSecurityOptions") or {}
    if not fgac.get("AnonymousAuthEnabled", False):
        return
    logger.info(
        "  Disabling AnonymousAuth migration mode (required before Dashboards "
        "access policy update)..."
    )
    es_client.update_elasticsearch_domain_config(
        DomainName=domain_name,
        AdvancedSecurityOptions={
            "Enabled": True,
            "InternalUserDatabaseEnabled": fgac.get(
                "InternalUserDatabaseEnabled", True
            ),
            "AnonymousAuthEnabled": False,
        },
    )
    _wait_for_opensearch_domain_config(domain_name)


def ensure_opensearch_access_policy(domain_name: str) -> None:
    """Apply access policy so browser Dashboards login can reach FGAC."""
    response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
    status = response["DomainStatus"]
    current_policy = status.get("AccessPolicies") or ""
    if _access_policy_allows_dashboards(current_policy):
        logger.info("  OpenSearch access policy already allows Dashboards")
        return

    fgac = status.get("AdvancedSecurityOptions") or {}
    if fgac.get("Enabled") and fgac.get("AnonymousAuthEnabled", False):
        _disable_opensearch_anonymous_auth_migration(domain_name)

    desired_policy = json.dumps(_build_opensearch_access_policies())
    logger.info("  Updating OpenSearch access policy for Dashboards...")
    es_client.update_elasticsearch_domain_config(
        DomainName=domain_name,
        AccessPolicies=desired_policy,
    )
    _wait_for_opensearch_domain_config(domain_name)
    logger.info("  ✓ OpenSearch access policy updated for Dashboards")


def finalize_opensearch_dashboards_access(domain_name: str) -> None:
    """Ensure FGAC migration is finished and the domain policy allows Dashboards."""
    response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
    fgac = response["DomainStatus"].get("AdvancedSecurityOptions") or {}
    if not fgac.get("Enabled", False):
        return
    _disable_opensearch_anonymous_auth_migration(domain_name)
    ensure_opensearch_access_policy(domain_name)


def _advanced_security_options(
    master_password: str, *, migrating_existing: bool = False
) -> Dict:
    """Fine-grained access control options for OpenSearch Dashboards login."""
    options = {
        "Enabled": True,
        "InternalUserDatabaseEnabled": True,
        "MasterUserOptions": {
            "MasterUserName": OPENSEARCH_MASTER_USERNAME,
            "MasterUserPassword": master_password,
        },
    }
    # AWS requires a short migration window when enabling FGAC on an existing domain.
    if migrating_existing:
        options["AnonymousAuthEnabled"] = True
    return options


def _is_fgac_enabled(domain_status: Dict) -> bool:
    opts = domain_status.get("AdvancedSecurityOptions") or {}
    return bool(opts.get("Enabled", False))


def _opensearch_dashboards_url(endpoint_url: str) -> str:
    return f"{endpoint_url.rstrip('/')}/_dashboards"


def ensure_opensearch_fine_grained_access(
    domain_name: str, master_password: str
) -> None:
    """Enable FGAC on an existing domain so Dashboards login works in the browser."""
    if not master_password:
        response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
        if _is_fgac_enabled(response["DomainStatus"]):
            return
        raise RuntimeError(
            "OpenSearch fine-grained access control is not enabled; "
            "re-run the installer and set the master user password."
        )

    response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
    status = response["DomainStatus"]
    if _is_fgac_enabled(status):
        logger.info(
            f"  Fine-grained access control already enabled "
            f"(Dashboards user: {OPENSEARCH_MASTER_USERNAME})"
        )
        if status.get("Processing", False):
            _wait_for_opensearch_domain_config(domain_name)
        return

    logger.info(
        "  Enabling fine-grained access control for OpenSearch Dashboards "
        "(existing domain migration)..."
    )
    es_client.update_elasticsearch_domain_config(
        DomainName=domain_name,
        AdvancedSecurityOptions=_advanced_security_options(
            master_password, migrating_existing=True
        ),
    )
    _wait_for_opensearch_domain_config(domain_name)
    logger.info(
        f"  ✓ Fine-grained access control enabled "
        f"(Dashboards user: {OPENSEARCH_MASTER_USERNAME})"
    )


def create_managed_opensearch_domain(master_password: str) -> Dict[str, str]:
    """
    Create Amazon OpenSearch Service domain for application/opensearch.py.

    Writes managed_opensearch_url to config.json; index name equals projectName.
    """
    logger.info(f"[2/3] Creating managed OpenSearch domain: {opensearch_domain_name}")

    try:
        response = es_client.describe_elasticsearch_domain(DomainName=opensearch_domain_name)
        status = response["DomainStatus"]
        endpoint_url = _opensearch_domain_endpoint(status)

        if status.get("Deleted"):
            raise RuntimeError(
                f"OpenSearch domain '{opensearch_domain_name}' is being deleted; "
                "wait for deletion to finish before re-running the installer"
            )

        if _is_opensearch_domain_usable(status):
            endpoint_url = _opensearch_domain_endpoint(status)
            if status.get("Processing", False):
                proc = status.get("DomainProcessingStatus") or status.get("ProcessingStatus")
                logger.warning(
                    f"  Skipping domain wait — reusing existing endpoint "
                    f"(processing={proc}): {endpoint_url}"
                )
            else:
                logger.warning(f"  Reusing existing OpenSearch domain: {endpoint_url}")
            ensure_opensearch_fine_grained_access(
                opensearch_domain_name, master_password
            )
            finalize_opensearch_dashboards_access(opensearch_domain_name)
            return {
                "arn": status["ARN"],
                "endpoint": endpoint_url,
                "dashboards_url": _opensearch_dashboards_url(endpoint_url),
                "domain_name": opensearch_domain_name,
                "engine_version": _domain_engine_version(status),
            }

        logger.info("  Existing domain found but still provisioning; waiting for endpoint...")
        status = _wait_for_opensearch_domain(opensearch_domain_name)
        endpoint_url = _opensearch_domain_endpoint(status)
        ensure_opensearch_fine_grained_access(
            opensearch_domain_name, master_password
        )
        finalize_opensearch_dashboards_access(opensearch_domain_name)
        return {
            "arn": status["ARN"],
            "endpoint": endpoint_url,
            "dashboards_url": _opensearch_dashboards_url(endpoint_url),
            "domain_name": opensearch_domain_name,
            "engine_version": _domain_engine_version(status),
        }
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            logger.error(f"Failed to describe OpenSearch domain: {e}")
            raise

    access_policies = _build_opensearch_access_policies()

    try:
        es_client.create_elasticsearch_domain(
            DomainName=opensearch_domain_name,
            ElasticsearchVersion="OpenSearch_3.5",
            ElasticsearchClusterConfig={
                "InstanceType": "r6g.large.elasticsearch",
                "InstanceCount": 1,
                "DedicatedMasterEnabled": False,
                "ZoneAwarenessEnabled": False,
            },
            EBSOptions={
                "EBSEnabled": True,
                "VolumeType": "gp3",
                "VolumeSize": 100,
            },
            NodeToNodeEncryptionOptions={"Enabled": True},
            EncryptionAtRestOptions={"Enabled": True},
            DomainEndpointOptions={"EnforceHTTPS": True},
            AccessPolicies=json.dumps(access_policies),
            AdvancedSecurityOptions=_advanced_security_options(master_password),
        )
        logger.info(f"  ✓ Initiated OpenSearch domain creation: {opensearch_domain_name}")
    except ClientError as create_error:
        if create_error.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            logger.error(f"Failed to create OpenSearch domain: {create_error}")
            raise
        logger.warning(f"  OpenSearch domain already exists: {opensearch_domain_name}")

    status = _wait_for_opensearch_domain(opensearch_domain_name)
    endpoint_url = _opensearch_domain_endpoint(status)
    if not endpoint_url:
        raise RuntimeError(
            f"OpenSearch domain '{opensearch_domain_name}' is active but has no endpoint"
        )

    ensure_opensearch_fine_grained_access(opensearch_domain_name, master_password)
    finalize_opensearch_dashboards_access(opensearch_domain_name)

    dashboards_url = _opensearch_dashboards_url(endpoint_url)
    logger.info(f"✓ Managed OpenSearch domain ready: {endpoint_url}")
    logger.info(f"  OpenSearch Dashboards: {dashboards_url}")
    logger.info(f"  Dashboards login user: {OPENSEARCH_MASTER_USERNAME}")
    return {
        "arn": status["ARN"],
        "endpoint": endpoint_url,
        "dashboards_url": dashboards_url,
        "domain_name": opensearch_domain_name,
        "engine_version": _domain_engine_version(status),
    }


def _get_domain_engine_version(domain_name: str) -> str:
    response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
    version = _domain_engine_version(response["DomainStatus"])
    if not version:
        raise RuntimeError(
            f"Could not determine engine version for OpenSearch domain '{domain_name}'"
        )
    return version


def find_analysis_nori_package_id(engine_version: str) -> Optional[str]:
    """
    Find the AWS-managed analysis-nori ZIP-PLUGIN for a domain engine version.

    Package name is still ``analysis-nori``; the installed plugin is
    ``opensearch-analysis-nori`` (e.g. G26022645 for OpenSearch_3.5).
    """
    response = opensearch_client.describe_packages(
        Filters=[{"Name": "PackageName", "Value": ["analysis-nori"]}]
    )
    for pkg in response.get("PackageDetailsList", []):
        if (
            pkg.get("EngineVersion") == engine_version
            and pkg.get("PackageType") == "ZIP-PLUGIN"
            and pkg.get("PackageStatus") == "AVAILABLE"
        ):
            return pkg["PackageID"]
    return None


def _find_domain_analysis_nori_package(domain_name: str) -> Optional[Dict]:
    """Return analysis-nori package details if already linked to the domain."""
    for pkg in opensearch_client.list_packages_for_domain(DomainName=domain_name).get(
        "DomainPackageDetailsList", []
    ):
        if pkg.get("PackageName") == "analysis-nori":
            return pkg
    return None


def _is_nori_analyzer_available(
    endpoint_url: str, master_password: str = ""
) -> bool:
    """True when the cluster can tokenize with the nori analyzer (plugin usable)."""
    try:
        os_client = _get_opensearch_client(endpoint_url, master_password)
        response = os_client.indices.analyze(body={"analyzer": "nori", "text": "test"})
        return bool(response.get("tokens"))
    except Exception:
        return False


def _wait_for_domain_package(
    domain_name: str,
    package_id: str,
    max_wait: int = 1800,
    poll_interval: int = 30,
) -> None:
    """Wait until an associated domain package reaches ACTIVE."""
    waited = 0
    while waited < max_wait:
        response = opensearch_client.list_packages_for_domain(DomainName=domain_name)
        for pkg in response.get("DomainPackageDetailsList", []):
            if pkg.get("PackageID") != package_id:
                continue
            status = pkg.get("DomainPackageStatus")
            if status == "ACTIVE":
                return
            if status == "ASSOCIATION_FAILED":
                raise RuntimeError(
                    f"analysis-nori association failed: {pkg.get('ErrorDetails')}"
                )
            if waited % 120 == 0:
                logger.info(f"  analysis-nori package status: {status} ({waited}s)")
        time.sleep(poll_interval)
        waited += poll_interval

    raise TimeoutError(
        f"Timeout waiting for analysis-nori package {package_id} on domain '{domain_name}'"
    )


def ensure_analysis_nori_plugin(
    domain_name: str,
    engine_version: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    master_password: str = "",
) -> bool:
    """
    Ensure analysis-nori is on the domain (associate only if missing).

    Returns True when a Nori-based index can be created.
    Skips associate/wait when the package is already linked and Nori is usable.
    """
    engine_version = engine_version or _get_domain_engine_version(domain_name)
    existing = _find_domain_analysis_nori_package(domain_name)
    if existing:
        status = existing.get("DomainPackageStatus")
        package_id = existing.get("PackageID")
        if status == "ACTIVE":
            logger.info(
                f"  Skipping analysis-nori — already ACTIVE (package {package_id})"
            )
            return True
        if status == "ASSOCIATION_FAILED":
            raise RuntimeError(
                f"analysis-nori association failed: {existing.get('ErrorDetails')}"
            )
        if endpoint_url and _is_nori_analyzer_available(
            endpoint_url, master_password
        ):
            logger.info(
                f"  Skipping analysis-nori wait — already linked ({status}, "
                f"package {package_id}), nori analyzer is usable"
            )
            return True
        logger.info(
            f"  Skipping analysis-nori associate — already linked ({status}, "
            f"package {package_id}); waiting for ACTIVE..."
        )
        _wait_for_domain_package(domain_name, package_id)
        logger.info(f"✓ analysis-nori plugin active (package {package_id})")
        return True

    package_id = find_analysis_nori_package_id(engine_version)
    if not package_id:
        logger.warning(
            f"No AWS analysis-nori package for {engine_version}. "
            "Install manually: OpenSearch console → Domain → Packages → "
            "associate 'analysis-nori' for your engine version."
        )
        return False

    logger.info(f"  Associating analysis-nori package {package_id} ({engine_version})...")
    opensearch_client.associate_package(PackageID=package_id, DomainName=domain_name)
    if endpoint_url and _is_nori_analyzer_available(
        endpoint_url, master_password
    ):
        logger.info(
            f"  Nori analyzer usable before package ACTIVE — continuing (package {package_id})"
        )
        return True
    _wait_for_domain_package(domain_name, package_id)
    logger.info(f"✓ analysis-nori plugin associated (package {package_id})")
    return True


def _get_opensearch_client(
    endpoint_url: str, master_password: str = ""
) -> OpenSearch:
    """OpenSearch client.

    Uses HTTP basic auth (admin/master_password) when available so installer
    operations work under FGAC even before IAM principals are mapped as
    backend roles. Falls back to SigV4 (same as application/mcp_rag_opensearch.py)
    when no master password is provided.
    """
    if master_password:
        http_auth: object = (OPENSEARCH_MASTER_USERNAME, master_password)
    else:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        http_auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",
            session_token=credentials.token,
        )
    return OpenSearch(
        hosts=[{"host": endpoint_url.replace("https://", ""), "port": 443}],
        http_compress=True,
        http_auth=http_auth,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )


def _normalize_iam_arn_for_backend_role(arn: str) -> str:
    """Convert STS assumed-role ARNs to their IAM role ARN form for FGAC.

    `sts.get_caller_identity` returns
    `arn:aws:sts::ACCOUNT:assumed-role/RoleName/SessionName` when called
    from an assumed role; OpenSearch backend_role matching expects the
    underlying IAM role ARN (`arn:aws:iam::ACCOUNT:role/RoleName`).
    """
    match = re.match(
        r"^arn:aws:sts::(?P<account>\d+):assumed-role/(?P<role>[^/]+)/.*$", arn
    )
    if match:
        return (
            f"arn:aws:iam::{match.group('account')}:role/{match.group('role')}"
        )
    return arn


def _get_caller_backend_role_arn() -> str:
    """Backend role ARN for the IAM identity currently running the installer."""
    return _normalize_iam_arn_for_backend_role(
        sts_client.get_caller_identity()["Arn"]
    )


def _opensearch_security_request(
    endpoint_url: str,
    method: str,
    path: str,
    master_password: str,
    body: Optional[Dict] = None,
) -> requests.Response:
    """Call OpenSearch security plugin with admin HTTP basic auth."""
    url = f"{endpoint_url.rstrip('/')}{path}"
    return requests.request(
        method,
        url,
        auth=(OPENSEARCH_MASTER_USERNAME, master_password),
        headers={"Content-Type": "application/json"},
        data=json.dumps(body) if body is not None else None,
        timeout=30,
    )


def ensure_opensearch_backend_role_mappings(
    endpoint_url: str,
    master_password: str,
    iam_arns: Iterable[str],
) -> None:
    """Map IAM principals onto `all_access` so SigV4 callers work under FGAC.

    For SigV4 requests, the OpenSearch security plugin treats the caller's
    IAM ARN as the OpenSearch *username* (e.g. logs show
    `name=arn:aws:iam::...:user/foo, backend_roles=[]`). To grant the role
    we therefore have to add the ARN to the `users` list. We also keep it
    in `backend_roles` so the same mapping works for SAML/IdP flows.
    """
    requested = sorted({arn for arn in iam_arns if arn})
    if not requested:
        return
    if not master_password:
        logger.warning(
            "  Skipping FGAC role mapping — no master password available "
            "(SigV4 callers will get 403 until they are mapped manually)."
        )
        return

    path = f"/_plugins/_security/api/rolesmapping/{OPENSEARCH_BACKEND_ROLE_NAME}"
    get_resp = _opensearch_security_request(
        endpoint_url, "GET", path, master_password
    )
    if get_resp.status_code == 401:
        raise RuntimeError(
            "OpenSearch admin authentication failed while reading "
            f"{OPENSEARCH_BACKEND_ROLE_NAME} role mapping — check the master "
            "password saved in application/config.json."
        )

    if get_resp.status_code == 200:
        current = get_resp.json().get(OPENSEARCH_BACKEND_ROLE_NAME, {}) or {}
    elif get_resp.status_code == 404:
        current = {}
    else:
        raise RuntimeError(
            f"Failed to read {OPENSEARCH_BACKEND_ROLE_NAME} role mapping "
            f"({get_resp.status_code}): {get_resp.text}"
        )

    existing_users = set(current.get("users", []) or [])
    existing_backend = set(current.get("backend_roles", []) or [])
    desired_users = existing_users.union(requested)
    desired_backend = existing_backend.union(requested)
    if desired_users == existing_users and desired_backend == existing_backend:
        logger.info(
            f"  FGAC {OPENSEARCH_BACKEND_ROLE_NAME} already maps "
            f"{sorted(requested)}"
        )
        return

    body: Dict[str, List[str]] = {
        "users": sorted(desired_users),
        "backend_roles": sorted(desired_backend),
    }
    if current.get("hosts"):
        body["hosts"] = current["hosts"]

    put_resp = _opensearch_security_request(
        endpoint_url, "PUT", path, master_password, body=body
    )
    if put_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to update {OPENSEARCH_BACKEND_ROLE_NAME} role mapping "
            f"({put_resp.status_code}): {put_resp.text}"
        )
    added_users = sorted(desired_users - existing_users)
    added_backend = sorted(desired_backend - existing_backend)
    logger.info(
        f"  ✓ Mapped IAM principals to FGAC {OPENSEARCH_BACKEND_ROLE_NAME} "
        f"(users+={added_users}, backend_roles+={added_backend})"
    )


def is_not_exist(os_client: OpenSearch, index_name: str) -> bool:
    """Return True if the index does not exist (lambda_function.py)."""
    if os_client.indices.exists(index=index_name):
        logger.info(f"  Using existing index: {index_name}")
        return False
    logger.info(f"  Index does not exist: {index_name}")
    return True


def _rag_index_body(use_nori: bool) -> Dict:
    """RAG index mapping (lambda_function.py), OpenSearch 3.x k-NN field settings."""
    text_field = (
        {
            "analyzer": "my_analyzer",
            "search_analyzer": "my_analyzer",
            "type": "text",
        }
        if use_nori
        else {"type": "text"}
    )
    settings: Dict = {
        "index": {
            "knn": True,
        },
    }
    if use_nori:
        settings["analysis"] = {
            "analyzer": {
                "my_analyzer": {
                    "char_filter": ["html_strip"],
                    "tokenizer": "nori",
                    "filter": [
                        "nori_number",
                        "lowercase",
                        "trim",
                        "my_nori_part_of_speech",
                    ],
                    "type": "custom",
                }
            },
            "tokenizer": {
                "nori": {
                    "decompound_mode": "mixed",
                    "discard_punctuation": "true",
                    "type": "nori_tokenizer",
                }
            },
            "filter": {
                "my_nori_part_of_speech": {
                    "type": "nori_part_of_speech",
                    "stoptags": [
                        "E",
                        "IC",
                        "J",
                        "MAG",
                        "MAJ",
                        "MM",
                        "SP",
                        "SSC",
                        "SSO",
                        "SC",
                        "SE",
                        "XPN",
                        "XSA",
                        "XSN",
                        "XSV",
                        "UNA",
                        "NA",
                        "VSV",
                    ],
                }
            },
        }

    return {
        "settings": settings,
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "last_updated": {"type": "date"},
                        "project": {"type": "keyword"},
                        "seq_num": {"type": "long"},
                        "title": {"type": "text"},
                        "url": {"type": "text"},
                    }
                },
                "text": text_field,
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "space_type": "cosinesimil",
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                    },
                },
            }
        },
    }


def create_rag_index(os_client: OpenSearch, index_name: str, use_nori: bool) -> None:
    """Create hybrid RAG index (Nori lexical + k-NN vector)."""
    if is_not_exist(os_client, index_name):
        index_body = _rag_index_body(use_nori)
        response = os_client.indices.create(index=index_name, body=index_body)
        analyzer = "nori" if use_nori else "standard"
        logger.info(
            f"  ✓ Created OpenSearch index '{index_name}' "
            f"(analyzer={analyzer}): {response.get('acknowledged')}"
        )


def ensure_opensearch_index(
    endpoint_url: str, use_nori: bool = True, master_password: str = ""
) -> None:
    """Ensure the RAG vector index exists on the managed OpenSearch domain."""
    logger.info(f"  Ensuring OpenSearch index: {project_name}")
    os_client = _get_opensearch_client(endpoint_url, master_password)
    create_rag_index(os_client, project_name, use_nori=use_nori)
    logger.info(f"✓ OpenSearch index ready: {project_name}")


def create_secrets() -> Dict[str, str]:
    """Create Secrets Manager secrets."""
    logger.info("[1/6] Creating Secrets Manager secrets")
    logger.info("Please enter API keys when prompted (press Enter to skip and leave empty):")
    
    secrets = {
        "weather": {
            "name": f"openweathermap-{project_name}",
            "description": "secret for weather api key",
            "secret_value": {
                "project_name": project_name,
                "weather_api_key": ""
            }
        },
        "tavily": {
            "name": f"tavilyapikey-{project_name}",
            "description": "secret for tavily api key",
            "secret_value": {
                "project_name": project_name,
                "tavily_api_key": ""
            }
        }
    }
    
    secret_arns = {}
    
    for key, secret_config in secrets.items():
        # Check if secret already exists before prompting for input
        try:
            response = secrets_client.describe_secret(SecretId=secret_config["name"])
            secret_arns[key] = response["ARN"]
            logger.warning(f"  Secret already exists: {secret_config['name']}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Secret doesn't exist, prompt for API key and create it
                if key == "tavily":
                    logger.info(f"Enter credential of {secret_config['name']} (Tavily API Key):")
                    api_key = input(f"Creating {secret_config['name']} - Tavily API Key: ").strip()
                    secret_config["secret_value"]["tavily_api_key"] = api_key
                
                # Create the secret
                try:
                    response = secrets_client.create_secret(
                        Name=secret_config["name"],
                        Description=secret_config["description"],
                        SecretString=json.dumps(secret_config["secret_value"])
                    )
                    secret_arns[key] = response["ARN"]
                    logger.info(f"  ✓ Created secret: {secret_config['name']}")
                except ClientError as create_error:
                    logger.error(f"  Failed to create secret {secret_config['name']}: {create_error}")
                    raise
            else:
                logger.error(f"  Failed to check secret {secret_config['name']}: {e}")
                raise
    
    logger.info(f"✓ Created {len(secret_arns)} secrets")
    
    return secret_arns


def create_cloudfront_distribution(s3_bucket_name: str) -> Dict[str, str]:
    """Create CloudFront distribution with S3 origin."""
    logger.info("[3/3] Creating CloudFront distribution (S3)")
    
    # Check if CloudFront distribution already exists
    try:
        distributions = cloudfront_client.list_distributions()
        for dist in distributions.get("DistributionList", {}).get("Items", []):
            if cloudfront_comment in dist.get("Comment", ""):
                if dist.get("Enabled", False):
                    logger.warning(f"CloudFront distribution already exists: {dist['DomainName']}")
                    return {
                        "id": dist["Id"],
                        "domain": dist["DomainName"]
                    }
                else:
                    # Distribution exists but is disabled, enable it
                    logger.warning(f"CloudFront distribution exists but is disabled: {dist['DomainName']}")
                    logger.info("  Enabling existing CloudFront distribution...")
                    
                    # Get current distribution config
                    dist_config_response = cloudfront_client.get_distribution_config(Id=dist["Id"])
                    dist_config = dist_config_response["DistributionConfig"]
                    etag = dist_config_response["ETag"]
                    
                    # Enable the distribution
                    dist_config["Enabled"] = True
                    
                    # Update the distribution
                    cloudfront_client.update_distribution(
                        Id=dist["Id"],
                        DistributionConfig=dist_config,
                        IfMatch=etag
                    )
                    
                    logger.info(f"  ✓ Enabled CloudFront distribution: {dist['DomainName']}")
                    logger.warning("  Note: CloudFront distribution may take 15-20 minutes to deploy")
                    
                    return {
                        "id": dist["Id"],
                        "domain": dist["DomainName"]
                    }
    except Exception as e:
        logger.debug(f"Error checking existing distributions: {e}")
    
    # Check for existing Origin Access Identity or create new one (needed before creating distribution)
    logger.info("  Checking for existing Origin Access Identity for S3...")
    oai_id = None
    oai_canonical_user_id = None
    
    try:
        # Check existing OAIs
        oai_list = cloudfront_client.list_cloud_front_origin_access_identities()
        for oai in oai_list.get("CloudFrontOriginAccessIdentityList", {}).get("Items", []):
            if f"OAI for RAG Project" in oai.get("Comment", ""):
                oai_id = oai["Id"]
                oai_canonical_user_id = oai["S3CanonicalUserId"]
                logger.info(f"  ✓ Using existing Origin Access Identity: {oai_id}")
                break
        
        # Create new OAI if none exists
        if not oai_id:
            logger.info("  Creating new Origin Access Identity for S3...")
            oai_response = cloudfront_client.create_cloud_front_origin_access_identity(
                CloudFrontOriginAccessIdentityConfig={
                    "CallerReference": f"{project_name}-s3-oai-{int(time.time())}",
                    "Comment": f"OAI for RAG Project"
                }
            )
            oai_id = oai_response["CloudFrontOriginAccessIdentity"]["Id"]
            oai_canonical_user_id = oai_response["CloudFrontOriginAccessIdentity"]["S3CanonicalUserId"]
            logger.info(f"  ✓ Created Origin Access Identity: {oai_id}")
            
    except ClientError as e:
        logger.error(f"Failed to handle Origin Access Identity: {e}")
        raise
    
    # Update S3 bucket policy to allow CloudFront access
    logger.info("  Updating S3 bucket policy for CloudFront access...")
    
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowCloudFrontAccess",
                "Effect": "Allow",
                "Principal": {
                    "AWS": f"arn:aws:iam::cloudfront:user/CloudFront Origin Access Identity {oai_id}"
                },
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{s3_bucket_name}/*"
            }
        ]
    }
    
    try:
        # Wait for OAI to propagate before applying bucket policy
        logger.info("  Waiting for OAI to propagate...")
        time.sleep(10)
        
        s3_client.put_bucket_policy(
            Bucket=s3_bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        logger.info(f"  ✓ Updated S3 bucket policy")
    except ClientError as e:
        logger.error(f"Failed to update S3 bucket policy: {e}")
        logger.error(f"OAI ID: {oai_id}")
        logger.error(f"Bucket Policy: {json.dumps(bucket_policy, indent=2)}")
        raise

    # Create CloudFront distribution with S3 origin
    logger.info("  Creating CloudFront distribution with S3 origin...")
    distribution_config = {
        "CallerReference": f"rag-project-{int(time.time())}",
        "Comment": cloudfront_comment,
        "DefaultRootObject": "index.html",
        "DefaultCacheBehavior": {
            "TargetOriginId": f"s3-rag-project",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {
                "Quantity": 2,
                "Items": ["GET", "HEAD"],
                "CachedMethods": {
                    "Quantity": 2,
                    "Items": ["GET", "HEAD"]
                }
            },
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
            "Compress": True
        },
        "Origins": {
            "Quantity": 1,
            "Items": [
                {
                    "Id": f"s3-rag-project",
                    "DomainName": f"{s3_bucket_name}.s3.{region}.amazonaws.com",
                    "S3OriginConfig": {
                        "OriginAccessIdentity": f"origin-access-identity/cloudfront/{oai_id}"
                    }
                }
            ]
        },
        "Enabled": True,
        "PriceClass": "PriceClass_200"
    }
    
    logger.info("Creating CloudFront distribution with config:")
    logger.info(f"  Origins: {[origin['Id'] for origin in distribution_config['Origins']['Items']]}")
    logger.info(f"  DefaultCacheBehavior TargetOriginId: {distribution_config['DefaultCacheBehavior']['TargetOriginId']}")
    
    try:
        response = cloudfront_client.create_distribution(DistributionConfig=distribution_config)
        distribution_id = response["Distribution"]["Id"]
        distribution_domain = response["Distribution"]["DomainName"]
        
        logger.info(f"✓ CloudFront distribution created (S3): {distribution_domain}")
        logger.info(f"  Distribution ID: {distribution_id}")
        logger.info(f"  S3 origin: {s3_bucket_name}")
        logger.warning("  Note: CloudFront distribution may take 15-20 minutes to deploy")
        
    except ClientError as e:
        logger.error(f"Error creating CloudFront distribution: {e}")
        raise
    
    return {
        "id": distribution_id,
        "domain": distribution_domain
    }


def _build_lambda_deployment_package(source_dir: str) -> bytes:
    """Zip lambda_function.py and pip dependencies for Lambda deployment."""
    handler_path = os.path.join(source_dir, "lambda_function.py")
    if not os.path.isfile(handler_path):
        raise FileNotFoundError(f"Lambda handler not found: {handler_path}")

    requirements_path = os.path.join(source_dir, "requirements.txt")
    build_root = os.path.join(source_dir, ".lambda_build")
    package_dir = os.path.join(build_root, "package")

    if os.path.isdir(build_root):
        shutil.rmtree(build_root)
    os.makedirs(package_dir)

    if os.path.isfile(requirements_path):
        logger.info("  Installing Lambda dependencies from requirements.txt")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                requirements_path,
                "-t",
                package_dir,
                "--quiet",
                "--upgrade",
            ]
        )

    shutil.copy(handler_path, os.path.join(package_dir, "lambda_function.py"))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(package_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                archive.write(full_path, os.path.relpath(full_path, package_dir))

    shutil.rmtree(build_root, ignore_errors=True)
    return buffer.getvalue()


def _ensure_lambda_s3_event_role(
    bucket_name: str,
    opensearch_domain_arn: str,
) -> str:
    """Create IAM role for lambda-s3-event-manager (S3, OpenSearch, logs)."""
    logger.info(f"  Ensuring IAM role: {LAMBDA_S3_EVENT_ROLE_NAME}")
    role_arn = f"arn:aws:iam::{account_id}:role/{LAMBDA_S3_EVENT_ROLE_NAME}"

    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        iam_client.get_role(RoleName=LAMBDA_S3_EVENT_ROLE_NAME)
        logger.info(f"  ✓ IAM role already exists: {LAMBDA_S3_EVENT_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        iam_client.create_role(
            RoleName=LAMBDA_S3_EVENT_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description=f"Lambda execution role for {LAMBDA_S3_EVENT_FUNCTION_NAME}",
        )
        logger.info(f"  ✓ Created IAM role: {LAMBDA_S3_EVENT_ROLE_NAME}")
        time.sleep(10)

    try:
        iam_client.attach_role_policy(
            RoleName=LAMBDA_S3_EVENT_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    lambda_access_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "es:ESHttpGet",
                    "es:ESHttpPost",
                    "es:ESHttpPut",
                    "es:ESHttpDelete",
                    "es:ESHttpHead",
                ],
                "Resource": [f"{opensearch_domain_arn}/*"],
            },
        ],
    }
    iam_client.put_role_policy(
        RoleName=LAMBDA_S3_EVENT_ROLE_NAME,
        PolicyName=f"{project_name}-lambda-s3-event-access",
        PolicyDocument=json.dumps(lambda_access_policy),
    )
    logger.info("  ✓ Updated inline access policy on Lambda role (S3, OpenSearch)")

    return role_arn


def _deploy_lambda_s3_event_manager(
    role_arn: str,
    bucket_name: str,
    opensearch_endpoint: str,
) -> str:
    """Create or update lambda-s3-event-manager."""
    logger.info(f"  Deploying Lambda function: {LAMBDA_S3_EVENT_FUNCTION_NAME}")
    zip_bytes = _build_lambda_deployment_package(LAMBDA_S3_EVENT_SOURCE_DIR)
    environment = {
        "Variables": {
            "s3_bucket": bucket_name,
            "s3_prefix": S3_DOCS_PREFIX.rstrip("/"),
            "meta_prefix": "metadata/",
            "opensearch_url": opensearch_endpoint,
            "vectorIndexName": project_name,
            "region": region,
        }
    }

    try:
        lambda_client.get_function(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
        lambda_client.update_function_code(
            FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME,
            ZipFile=zip_bytes,
            Publish=True,
        )
        waiter = lambda_client.get_waiter("function_updated")
        waiter.wait(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME,
            Role=role_arn,
            Runtime="python3.12",
            Handler="lambda_function.lambda_handler",
            Timeout=120,
            MemorySize=256,
            Environment=environment,
        )
        waiter.wait(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
        logger.info(f"  ✓ Updated Lambda function: {LAMBDA_S3_EVENT_FUNCTION_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        lambda_client.create_function(
            FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=120,
            MemorySize=256,
            Environment=environment,
            Description="S3 docs/ events: PDF delete → OpenSearch cleanup",
        )
        waiter = lambda_client.get_waiter("function_active")
        waiter.wait(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
        logger.info(f"  ✓ Created Lambda function: {LAMBDA_S3_EVENT_FUNCTION_NAME}")

    response = lambda_client.get_function(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
    return response["Configuration"]["FunctionArn"]


def _grant_s3_invoke_lambda_permission(bucket_name: str, lambda_arn: str) -> None:
    """Allow S3 bucket to invoke the Lambda function."""
    statement_id = f"{project_name}-s3-docs-invoke"
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{bucket_name}",
            SourceAccount=account_id,
        )
        logger.info("  ✓ Added S3 invoke permission on Lambda")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise
        logger.info("  ✓ S3 invoke permission on Lambda already exists")


def _configure_s3_docs_lambda_notification(
    bucket_name: str,
    lambda_arn: str,
    prefix: str = S3_DOCS_PREFIX,
) -> None:
    """Register S3 ObjectCreated and ObjectRemoved notifications for objects under docs/."""
    logger.info(
        f"  Configuring S3 notification on s3://{bucket_name}/{prefix} "
        f"({', '.join(S3_LAMBDA_EVENTS)}) → {LAMBDA_S3_EVENT_FUNCTION_NAME}"
    )

    lambda_config = {
        "Id": S3_EVENT_NOTIFICATION_ID,
        "LambdaFunctionArn": lambda_arn,
        "Events": S3_LAMBDA_EVENTS,
        "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": prefix}]}},
    }

    try:
        existing = s3_client.get_bucket_notification_configuration(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchConfiguration":
            existing = {}
        else:
            raise

    lambda_configs = [
        item
        for item in existing.get("LambdaFunctionConfigurations", [])
        if item.get("Id") != S3_EVENT_NOTIFICATION_ID
    ]
    lambda_configs.append(lambda_config)

    notification_configuration = {"LambdaFunctionConfigurations": lambda_configs}
    for key in ("TopicConfigurations", "QueueConfigurations", "EventBridgeConfiguration"):
        if key in existing:
            notification_configuration[key] = existing[key]

    s3_client.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration=notification_configuration,
    )
    logger.info(f"  ✓ S3 bucket notification configured for prefix: {prefix}")


def deploy_lambda_s3_event_manager(
    s3_bucket_name: str,
    opensearch_endpoint: str,
    opensearch_domain_arn: str,
    opensearch_master_password: str = "",
) -> Dict[str, str]:
    """
    Deploy lambda-s3-event-manager: IAM role, Lambda, S3 trigger on docs/.

    Also maps the Lambda execution role ARN as an FGAC backend_role so the
    function's SigV4 requests can write/delete OpenSearch documents.
    """
    logger.info("[4/4] Deploying lambda-s3-event-manager (S3 docs/ PDF delete)")

    role_arn = _ensure_lambda_s3_event_role(s3_bucket_name, opensearch_domain_arn)

    if opensearch_master_password:
        ensure_opensearch_backend_role_mappings(
            opensearch_endpoint,
            opensearch_master_password,
            [_normalize_iam_arn_for_backend_role(role_arn)],
        )

    lambda_arn = _deploy_lambda_s3_event_manager(
        role_arn, s3_bucket_name, opensearch_endpoint
    )
    _grant_s3_invoke_lambda_permission(s3_bucket_name, lambda_arn)
    _configure_s3_docs_lambda_notification(s3_bucket_name, lambda_arn)

    logger.info(f"✓ lambda-s3-event-manager deployed: {lambda_arn}")
    return {
        "function_name": LAMBDA_S3_EVENT_FUNCTION_NAME,
        "function_arn": lambda_arn,
        "role_name": LAMBDA_S3_EVENT_ROLE_NAME,
        "role_arn": role_arn,
        "s3_docs_prefix": S3_DOCS_PREFIX,
    }


def main():
    """Main function to create all infrastructure."""
    logger.info("="*60)
    logger.info("Starting AWS Infrastructure Deployment")
    logger.info("="*60)
    logger.info(f"Project: {project_name}")
    logger.info(f"Region: {region}")
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Bucket Name: {bucket_name}")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 1. Create secrets
        # secret_arns = create_secrets()
        # logger.info(f"Secrets created...")
        
        # 1. Create S3 bucket
        s3_bucket_name = create_s3_bucket()
        logger.info("S3 bucket created")

        # 2. Create managed OpenSearch domain (application/opensearch.py)
        opensearch_master_password = prompt_opensearch_master_password_if_needed()
        opensearch_info = create_managed_opensearch_domain(opensearch_master_password)
        logger.info("Managed OpenSearch domain created")

        # Verify admin auth against the live cluster; offer retry + AWS-side
        # password reset on 401 so the installer can recover without manual
        # intervention in the AWS console.
        if opensearch_master_password:
            opensearch_master_password = ensure_opensearch_master_password_works(
                opensearch_info["endpoint"],
                opensearch_info["domain_name"],
                opensearch_master_password,
            )

        # Map the IAM caller (this installer + local app via SigV4) as a
        # backend_role on `all_access` so subsequent OpenSearch calls aren't
        # rejected with 403 by the FGAC security plugin.
        if opensearch_master_password:
            ensure_opensearch_backend_role_mappings(
                opensearch_info["endpoint"],
                opensearch_master_password,
                [_get_caller_backend_role_arn()],
            )

        nori_ready = ensure_analysis_nori_plugin(
            opensearch_info["domain_name"],
            engine_version=opensearch_info.get("engine_version"),
            endpoint_url=opensearch_info["endpoint"],
            master_password=opensearch_master_password,
        )
        ensure_opensearch_index(
            opensearch_info["endpoint"],
            use_nori=nori_ready,
            master_password=opensearch_master_password,
        )
        logger.info("OpenSearch index created")

        # 3. Create CloudFront distribution
        cloudfront_info = create_cloudfront_distribution(s3_bucket_name)
        logger.info("CloudFront distribution created")

        # 4. S3 docs/ events → lambda-s3-event-manager
        lambda_s3_event_info = deploy_lambda_s3_event_manager(
            s3_bucket_name,
            opensearch_info["endpoint"],
            opensearch_info["arn"],
            opensearch_master_password=opensearch_master_password,
        )
        logger.info("lambda-s3-event-manager deployed")
        
        # Output summary
        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info("Infrastructure Deployment Completed Successfully!")
        logger.info("="*60)
        logger.info("Summary:")
        logger.info(f"  S3 Bucket: {s3_bucket_name}")
        logger.info(f"  OpenSearch Domain: {opensearch_info['endpoint']}")
        logger.info(f"  OpenSearch Dashboards: {opensearch_info['dashboards_url']}")
        logger.info(
            f"  Dashboards login: {OPENSEARCH_MASTER_USERNAME} "
            "(password set during install)"
        )
        logger.info(f"  CloudFront Domain: https://{cloudfront_info['domain']}")
        logger.info(f"  Lambda (S3 docs/): {lambda_s3_event_info['function_arn']}")
        logger.info(f"  S3 event prefix: {lambda_s3_event_info['s3_docs_prefix']}")
        logger.info("")
        logger.info(f"Total deployment time: {elapsed_time/60:.2f} minutes")
        logger.info("="*60)
        logger.info("Note: CloudFront distribution may take 15-20 minutes to fully deploy")
        logger.info("="*60)
        
        # Update application/config.json
        config_path = "application/config.json"
        config_data = {}
        
        # Read existing config if it exists
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            logger.info(f"Creating new {config_path}")
        except Exception as e:
            logger.warning(f"Could not read existing {config_path}: {e}")
        
        # Update only necessary fields
        config_data.update({
            "projectName": project_name,
            "accountId": account_id,
            "region": region,
            "s3_bucket": s3_bucket_name,
            "s3_arn": f"arn:aws:s3:::{s3_bucket_name}",
            "sharing_url": f"https://{cloudfront_info['domain']}",
            "managed_opensearch_url": opensearch_info["endpoint"],
            "managed_opensearch_arn": opensearch_info["arn"],
            "managed_opensearch_dashboards_url": opensearch_info["dashboards_url"],
            "managed_opensearch_dashboards_user": OPENSEARCH_MASTER_USERNAME,
            "s3_docs_prefix": S3_DOCS_PREFIX,
            "lambda_s3_event_manager_arn": lambda_s3_event_info["function_arn"],
            "lambda_s3_event_manager_name": lambda_s3_event_info["function_name"],
        })
        config_data.pop("lambda_s3_event_sqs_fifo_urls", None)
        if opensearch_master_password:
            config_data["managed_opensearch_dashboards_password"] = (
                opensearch_master_password
            )

        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"✓ Updated {config_path}")
        except Exception as e:
            logger.warning(f"Could not update {config_path}: {e}")
        
        logger.info("="*60)
        logger.info("")
        logger.info("="*60)
        logger.info("  IMPORTANT: CloudFront Domain Address")
        logger.info("="*60)
        logger.info(f" CloudFront URL: https://{cloudfront_info['domain']}")
        logger.info("")
        logger.info("Note: CloudFront distribution may take 15-20 minutes to fully deploy")
        logger.info("      Static content is served from S3 via CloudFront at the URL above")
        logger.info("="*60)
        logger.info("")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("")
        logger.error("="*60)
        logger.error("Deployment Failed!")
        logger.error("="*60)
        logger.error(f"Error: {e}")
        logger.error(f"Deployment time before failure: {elapsed_time/60:.2f} minutes")
        logger.error("="*60)
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

