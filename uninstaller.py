#!/usr/bin/env python3
"""
AWS Infrastructure Uninstaller
Deletes all AWS resources created by installer.py.
"""

import argparse
import logging
import sys
import time

import boto3
from botocore.exceptions import ClientError

# Configuration (must match installer.py)
project_name = "rag-multimodal"
region = "us-west-2"
AGENTCORE_GATEWAY_REGION = "us-east-1"
AGENTCORE_WEBSEARCH_GATEWAY_NAME = "gateway-websearch"
cloudfront_comment = "CloudFront-for-rag-project"
oai_comment = "OAI for RAG Project"

sts_client = boto3.client("sts", region_name=region)
account_id = sts_client.get_caller_identity()["Account"]

opensearch_domain_name = project_name
bucket_name = f"storage-for-rag-project-{account_id}-{region}"

LAMBDA_S3_EVENT_FUNCTION_NAME = f"lambda-s3-event-manager-for-{project_name}"
LAMBDA_S3_EVENT_ROLE_NAME = f"role-lambda-s3-event-manager-for-{project_name}-{region}"
SQS_S3_EVENT_QUEUE_BASE = f"sqs-s3-event-for-{project_name}-{region}"
SQS_S3_EVENT_QUEUE_COUNT = 1
S3_EVENT_NOTIFICATION_ID = f"{project_name}-docs-s3-event"

s3_client = boto3.client("s3", region_name=region)
iam_client = boto3.client("iam", region_name=region)
lambda_client = boto3.client("lambda", region_name=region)
sqs_client = boto3.client("sqs", region_name=region)
secrets_client = boto3.client("secretsmanager", region_name=region)
# Managed Amazon OpenSearch Service (same as installer.py)
es_client = boto3.client("es", region_name=region)
opensearch_client = boto3.client("opensearch", region_name=region)
cloudfront_client = boto3.client("cloudfront", region_name=region)
bedrock_agent_client = boto3.client("bedrock-agent", region_name=region)
agentcore_control_client = boto3.client(
    "bedrock-agentcore-control",
    region_name=AGENTCORE_GATEWAY_REGION,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Return True for yes; empty input uses default."""
    hint = "Y/n" if default else "y/N"
    response = input(f"{prompt} ({hint}): ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def _matches_cloudfront(dist: dict) -> bool:
    return cloudfront_comment in dist.get("Comment", "")


def disable_cloudfront_distributions():
    """Disable CloudFront distributions created by installer."""
    logger.info("[1/7] Disabling CloudFront distributions")

    try:
        distributions = cloudfront_client.list_distributions()
        for dist in distributions.get("DistributionList", {}).get("Items", []):
            if not _matches_cloudfront(dist):
                continue
            if not dist.get("Enabled", True):
                logger.info(f"  Distribution already disabled: {dist['Id']}")
                continue

            dist_id = dist["Id"]
            logger.info(f"  Disabling distribution: {dist_id}")
            config_response = cloudfront_client.get_distribution_config(Id=dist_id)
            config = config_response["DistributionConfig"]
            config["Enabled"] = False
            cloudfront_client.update_distribution(
                Id=dist_id,
                DistributionConfig=config,
                IfMatch=config_response["ETag"],
            )

        logger.info("✓ CloudFront distributions disabled (deployment may take several minutes)")
    except Exception as e:
        logger.error(f"Error disabling CloudFront distributions: {e}")


def wait_for_cloudfront_disabled(max_wait: int = 900, poll_interval: int = 30):
    """Wait until project CloudFront distributions are fully disabled."""
    logger.info("  Waiting for CloudFront distributions to become disabled...")

    waited = 0
    while waited < max_wait:
        still_enabled = []
        distributions = cloudfront_client.list_distributions()
        for dist in distributions.get("DistributionList", {}).get("Items", []):
            if _matches_cloudfront(dist) and dist.get("Enabled", True):
                still_enabled.append(dist["Id"])

        if not still_enabled:
            logger.info("  ✓ All matching CloudFront distributions are disabled")
            return True

        logger.info(
            f"  Still enabled: {still_enabled} ({waited}s/{max_wait}s)"
        )
        time.sleep(poll_interval)
        waited += poll_interval

    logger.warning("  Timed out waiting for CloudFront to disable; delete step may be skipped")
    return False


def delete_cloudfront_distributions():
    """Delete disabled CloudFront distributions."""
    logger.info("[7/7] Deleting CloudFront distributions")

    try:
        distributions = cloudfront_client.list_distributions()
        for dist in distributions.get("DistributionList", {}).get("Items", []):
            if not _matches_cloudfront(dist):
                continue
            if dist.get("Enabled", True):
                logger.info(f"  Skipping enabled distribution: {dist['Id']}")
                continue

            dist_id = dist["Id"]
            try:
                config_response = cloudfront_client.get_distribution_config(Id=dist_id)
                cloudfront_client.delete_distribution(
                    Id=dist_id,
                    IfMatch=config_response["ETag"],
                )
                logger.info(f"  ✓ Deleted distribution: {dist_id}")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "DistributionNotDisabled":
                    logger.info(f"  Distribution {dist_id} is not fully disabled yet, skipping")
                elif code == "NoSuchDistribution":
                    logger.debug(f"  Distribution {dist_id} already deleted")
                else:
                    logger.warning(f"  Could not delete distribution {dist_id}: {e}")

        logger.info("✓ CloudFront distributions processed")
    except Exception as e:
        logger.error(f"Error deleting CloudFront distributions: {e}")


def delete_cloudfront_oai():
    """Delete Origin Access Identity created for the RAG project."""
    logger.info("  Deleting CloudFront Origin Access Identities")

    try:
        oai_list = cloudfront_client.list_cloud_front_origin_access_identities()
        for oai in oai_list.get("CloudFrontOriginAccessIdentityList", {}).get("Items", []):
            if oai_comment not in oai.get("Comment", ""):
                continue
            oai_id = oai["Id"]
            try:
                config_response = cloudfront_client.get_cloud_front_origin_access_identity_config(
                    Id=oai_id
                )
                cloudfront_client.delete_cloud_front_origin_access_identity(
                    Id=oai_id,
                    IfMatch=config_response["ETag"],
                )
                logger.info(f"  ✓ Deleted OAI: {oai_id}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchCloudFrontOriginAccessIdentity":
                    logger.debug(f"  OAI {oai_id} already deleted")
                else:
                    logger.warning(f"  Could not delete OAI {oai_id}: {e}")
    except Exception as e:
        logger.warning(f"  Error deleting OAI: {e}")


def delete_knowledge_bases():
    """Delete Knowledge Bases and their data sources."""
    logger.info("[2/7] Deleting Knowledge Bases")

    try:
        kb_list = bedrock_agent_client.list_knowledge_bases()
        kb_to_delete = [
            kb["knowledgeBaseId"]
            for kb in kb_list.get("knowledgeBaseSummaries", [])
            if kb["name"] == project_name
        ]

        if not kb_to_delete:
            logger.info(f"  No Knowledge Base found with name: {project_name}")
            return

        for kb_id in kb_to_delete:
            logger.info(f"  Deleting Knowledge Base: {kb_id}")

            try:
                data_sources = bedrock_agent_client.list_data_sources(
                    knowledgeBaseId=kb_id,
                    maxResults=100,
                )
                for ds in data_sources.get("dataSourceSummaries", []):
                    try:
                        bedrock_agent_client.delete_data_source(
                            knowledgeBaseId=kb_id,
                            dataSourceId=ds["dataSourceId"],
                        )
                        logger.info(f"    ✓ Deleted data source: {ds['dataSourceId']}")
                    except Exception as e:
                        logger.warning(
                            f"    Could not delete data source {ds['dataSourceId']}: {e}"
                        )
            except Exception as e:
                logger.debug(f"    Error listing/deleting data sources: {e}")

            try:
                bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=kb_id)
                logger.info(f"  ✓ Deleted Knowledge Base: {kb_id}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.debug(f"  Knowledge Base {kb_id} already deleted")
                else:
                    logger.warning(f"  Could not delete Knowledge Base {kb_id}: {e}")
                    continue

            max_wait = 120
            waited = 0
            while waited < max_wait:
                try:
                    kb_response = bedrock_agent_client.get_knowledge_base(
                        knowledgeBaseId=kb_id
                    )
                    status = kb_response["knowledgeBase"]["status"]
                    if status == "DELETED":
                        break
                    time.sleep(5)
                    waited += 5
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ResourceNotFoundException":
                        break
                    raise

        logger.info("✓ Knowledge Bases deleted")
    except Exception as e:
        logger.error(f"Error deleting Knowledge Bases: {e}")


def _dissociate_analysis_nori_package(domain_name: str) -> None:
    """Dissociate analysis-nori package before deleting the domain (best-effort)."""
    try:
        response = opensearch_client.list_packages_for_domain(DomainName=domain_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return
        logger.warning(f"  Could not list domain packages: {e}")
        return

    for pkg in response.get("DomainPackageDetailsList", []):
        if pkg.get("PackageName") != "analysis-nori":
            continue
        package_id = pkg.get("PackageID")
        status = pkg.get("DomainPackageStatus")
        if status in ("DISSOCIATING", "DISSOCIATION_FAILED"):
            logger.info(
                f"  analysis-nori already dissociating (status={status}, "
                f"package {package_id})"
            )
            continue
        try:
            opensearch_client.dissociate_package(
                PackageID=package_id, DomainName=domain_name
            )
            logger.info(
                f"  ✓ Dissociated analysis-nori package {package_id} from {domain_name}"
            )
        except ClientError as e:
            logger.warning(
                f"  Could not dissociate analysis-nori ({package_id}): {e}"
            )


def _wait_for_opensearch_domain_deleted(
    domain_name: str,
    max_wait: int = 1800,
    poll_interval: int = 30,
    log_interval: int = 60,
) -> bool:
    """
    Poll until the managed OpenSearch domain is fully gone.

    Returns True if confirmed deleted, False on timeout.
    Domain deletion is asynchronous and typically takes 10-30 minutes.
    """
    logger.info(
        f"  Waiting for OpenSearch domain '{domain_name}' to be fully deleted "
        f"(typically 10-30 min, timeout {max_wait // 60} min)..."
    )

    waited = 0
    last_log = 0
    while waited < max_wait:
        try:
            response = es_client.describe_elasticsearch_domain(DomainName=domain_name)
            status = response["DomainStatus"]
            deleted_flag = status.get("Deleted", False)
            processing = status.get("Processing", False)
            proc_status = (
                status.get("DomainProcessingStatus")
                or status.get("ProcessingStatus")
            )

            if waited - last_log >= log_interval or waited == 0:
                logger.info(
                    f"  [{waited // 60}m{waited % 60:02d}s/"
                    f"{max_wait // 60}m] deleted={deleted_flag}, "
                    f"processing={processing}, status={proc_status}"
                )
                last_log = waited
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(
                    f"  ✓ OpenSearch domain '{domain_name}' fully deleted "
                    f"(elapsed {waited // 60}m{waited % 60:02d}s)"
                )
                return True
            logger.warning(f"  describe_elasticsearch_domain error: {e}")

        time.sleep(poll_interval)
        waited += poll_interval

    logger.warning(
        f"  Timed out waiting for OpenSearch domain deletion after "
        f"{max_wait // 60} minutes; deletion is still in progress in AWS"
    )
    return False


def delete_opensearch_domain(wait: bool = True, wait_timeout: int = 1800):
    """Delete managed Amazon OpenSearch Service domain created by installer.py."""
    logger.info(f"[3/7] Deleting OpenSearch domain: {opensearch_domain_name}")

    try:
        already_deleting = False
        try:
            response = es_client.describe_elasticsearch_domain(
                DomainName=opensearch_domain_name
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"  Domain {opensearch_domain_name} not found")
                logger.info("✓ OpenSearch domain processed")
                return
            raise

        status = response["DomainStatus"]
        if status.get("Deleted"):
            logger.info(
                f"  Domain {opensearch_domain_name} is already being deleted"
            )
            already_deleting = True
        else:
            _dissociate_analysis_nori_package(opensearch_domain_name)

            try:
                es_client.delete_elasticsearch_domain(
                    DomainName=opensearch_domain_name
                )
                logger.info(
                    f"  ✓ Initiated deletion of OpenSearch domain: "
                    f"{opensearch_domain_name}"
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info(f"  Domain {opensearch_domain_name} already deleted")
                    logger.info("✓ OpenSearch domain processed")
                    return
                raise

        if wait:
            _wait_for_opensearch_domain_deleted(
                opensearch_domain_name, max_wait=wait_timeout
            )
        else:
            note = (
                "deletion already in progress; "
                if already_deleting
                else ""
            )
            logger.info(
                f"  Skipping wait ({note}check AWS console for completion)"
            )

        logger.info("✓ OpenSearch domain processed")
    except Exception as e:
        logger.error(f"Error deleting OpenSearch domain: {e}")


def _remove_s3_docs_lambda_notification():
    """Remove S3 notification for lambda-s3-event-manager on docs/."""
    try:
        existing = s3_client.get_bucket_notification_configuration(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchConfiguration", "NoSuchBucket"):
            return
        raise

    lambda_configs = [
        item
        for item in existing.get("LambdaFunctionConfigurations", [])
        if item.get("Id") != S3_EVENT_NOTIFICATION_ID
    ]

    notification_configuration = {}
    if lambda_configs:
        notification_configuration["LambdaFunctionConfigurations"] = lambda_configs
    for key in ("TopicConfigurations", "QueueConfigurations", "EventBridgeConfiguration"):
        if key in existing:
            notification_configuration[key] = existing[key]

    if notification_configuration:
        s3_client.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=notification_configuration,
        )
    else:
        s3_client.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration={},
        )
    logger.info("  ✓ Removed S3 docs/ Lambda notification")


def delete_lambda_s3_event_manager():
    """Delete lambda-s3-event-manager and related permissions (legacy SQS queues if any)."""
    logger.info("[4/7] Deleting lambda-s3-event-manager resources")

    try:
        _remove_s3_docs_lambda_notification()
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchBucket":
            logger.warning(f"  Could not remove S3 notification: {e}")

    try:
        lambda_client.remove_permission(
            FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME,
            StatementId=f"{project_name}-s3-docs-invoke",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            logger.warning(f"  Could not remove Lambda permission: {e}")

    try:
        lambda_client.delete_function(FunctionName=LAMBDA_S3_EVENT_FUNCTION_NAME)
        logger.info(f"  ✓ Deleted Lambda: {LAMBDA_S3_EVENT_FUNCTION_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            logger.warning(f"  Could not delete Lambda: {e}")

    for index in range(SQS_S3_EVENT_QUEUE_COUNT):
        queue_name = f"{SQS_S3_EVENT_QUEUE_BASE}-{index + 1}.fifo"
        try:
            response = sqs_client.get_queue_url(QueueName=queue_name)
            sqs_client.delete_queue(QueueUrl=response["QueueUrl"])
            logger.info(f"  ✓ Deleted SQS queue: {queue_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "AWS.SimpleQueueService.NonExistentQueue":
                logger.warning(f"  Could not delete SQS queue {queue_name}: {e}")

    logger.info("✓ lambda-s3-event-manager resources processed")


def _list_all_agentcore_gateways():
    gateways = []
    next_token = None
    while True:
        kwargs = {}
        if next_token:
            kwargs["nextToken"] = next_token
        response = agentcore_control_client.list_gateways(**kwargs)
        gateways.extend(response.get("items", []))
        next_token = response.get("nextToken")
        if not next_token:
            break
    return gateways


def _list_all_agentcore_gateway_targets(gateway_id: str):
    targets = []
    next_token = None
    while True:
        kwargs = {"gatewayIdentifier": gateway_id}
        if next_token:
            kwargs["nextToken"] = next_token
        response = agentcore_control_client.list_gateway_targets(**kwargs)
        targets.extend(response.get("items", []))
        next_token = response.get("nextToken")
        if not next_token:
            break
    return targets


def delete_agentcore_websearch_gateway(skip_confirmation: bool = False) -> bool:
    """Delete AgentCore gateway-websearch and its web-search targets."""
    logger.info("[4.5/7] Deleting AgentCore Web Search gateway")

    gateway_id = None
    try:
        for gateway in _list_all_agentcore_gateways():
            if gateway.get("name") == AGENTCORE_WEBSEARCH_GATEWAY_NAME:
                gateway_id = gateway["gatewayId"]
                logger.info(
                    f"  Found gateway: {AGENTCORE_WEBSEARCH_GATEWAY_NAME} ({gateway_id})"
                )
                break

        if not gateway_id:
            logger.info(
                f"  AgentCore gateway not found: {AGENTCORE_WEBSEARCH_GATEWAY_NAME}"
            )
            return True

        if not skip_confirmation:
            print("\n" + "=" * 60)
            print(
                f"AgentCore gateway '{AGENTCORE_WEBSEARCH_GATEWAY_NAME}' "
                f"({gateway_id}) in {AGENTCORE_GATEWAY_REGION} will be deleted."
            )
            print("This includes all gateway targets (web-search connector).")
            print("=" * 60)
            response = input(
                "\nDelete AgentCore Web Search gateway? (yes/no) [no]: "
            ).strip().lower()
            if response != "yes":
                logger.info(
                    "  Skipping AgentCore Web Search gateway deletion (default: no)."
                )
                return False

        for target in _list_all_agentcore_gateway_targets(gateway_id):
            target_id = target.get("targetId")
            target_name = target.get("name", target_id)
            try:
                agentcore_control_client.delete_gateway_target(
                    gatewayIdentifier=gateway_id,
                    targetId=target_id,
                )
                logger.info(f"  ✓ Deleted gateway target: {target_name} ({target_id})")
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFoundException":
                    logger.warning(
                        f"  Could not delete gateway target {target_name}: {e}"
                    )

        for _ in range(18):
            remaining_targets = _list_all_agentcore_gateway_targets(gateway_id)
            if not remaining_targets:
                break
            logger.info(
                f"  Waiting for {len(remaining_targets)} gateway target(s) to be deleted..."
            )
            time.sleep(10)

        agentcore_control_client.delete_gateway(gatewayIdentifier=gateway_id)
        logger.info(f"  ✓ Deleted gateway: {gateway_id}")
        logger.info("✓ AgentCore Web Search gateway deleted")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info(
                f"  AgentCore gateway already deleted: {AGENTCORE_WEBSEARCH_GATEWAY_NAME}"
            )
            return True
        logger.warning(f"  Could not delete AgentCore Web Search gateway: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting AgentCore Web Search gateway: {e}")
        return False


def delete_iam_roles(delete_agentcore_gateway_role: bool = True):
    """Delete IAM roles created by installer."""
    logger.info("[5/7] Deleting IAM roles")

    role_names = [
        f"role-knowledge-base-for-{project_name}-{region}",
        LAMBDA_S3_EVENT_ROLE_NAME,
    ]
    if delete_agentcore_gateway_role:
        role_names.append(f"role-agentcore-gateway-websearch-for-{project_name}")
    else:
        logger.info(
            "  Keeping AgentCore gateway IAM role "
            f"(role-agentcore-gateway-websearch-for-{project_name})"
        )

    for role_name in role_names:
        try:
            attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
            for policy in attached_policies["AttachedPolicies"]:
                iam_client.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy["PolicyArn"],
                )

            inline_policies = iam_client.list_role_policies(RoleName=role_name)
            for policy_name in inline_policies["PolicyNames"]:
                iam_client.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                )

            iam_client.delete_role(RoleName=role_name)
            logger.info(f"  ✓ Deleted role: {role_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchEntity":
                logger.warning(f"  Could not delete role {role_name}: {e}")

    logger.info("✓ IAM roles deleted")


def _empty_s3_bucket(bucket: str):
    """Remove all objects and versions from an S3 bucket."""
    paginator = s3_client.get_paginator("list_object_versions")
    delete_keys = []

    for page in paginator.paginate(Bucket=bucket):
        for version in page.get("Versions", []):
            delete_keys.append(
                {"Key": version["Key"], "VersionId": version["VersionId"]}
            )
        for marker in page.get("DeleteMarkers", []):
            delete_keys.append(
                {"Key": marker["Key"], "VersionId": marker["VersionId"]}
            )

    if not delete_keys:
        return

    for i in range(0, len(delete_keys), 1000):
        batch = delete_keys[i : i + 1000]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})

    logger.info(f"  ✓ Deleted {len(delete_keys)} objects/versions from {bucket}")


def delete_s3_buckets():
    """Delete S3 bucket created by installer."""
    logger.info("[6/7] Deleting S3 buckets")

    for bucket in [bucket_name]:
        try:
            try:
                s3_client.head_bucket(Bucket=bucket)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket", "NotFound"):
                    logger.info(f"  Bucket {bucket} does not exist")
                    continue
                raise

            try:
                _empty_s3_bucket(bucket)
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchBucket":
                    logger.warning(f"  Could not empty bucket {bucket}: {e}")

            try:
                s3_client.delete_bucket_policy(Bucket=bucket)
                logger.info(f"  ✓ Removed bucket policy from {bucket}")
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchBucketPolicy":
                    logger.debug(f"  No bucket policy on {bucket}: {e}")

            s3_client.delete_bucket(Bucket=bucket)
            logger.info(f"  ✓ Deleted bucket: {bucket}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                logger.info(f"  Bucket {bucket} does not exist")
            else:
                logger.warning(f"  Could not delete bucket {bucket}: {e}")

    logger.info("✓ S3 buckets deleted")


def delete_secrets():
    """Delete optional Secrets Manager secrets (if created)."""
    logger.info("Deleting secrets (if present)")

    secret_names = [
        f"openweathermap-{project_name}",
        f"tavilyapikey-{project_name}",
    ]

    for secret_name in secret_names:
        try:
            secrets_client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True,
            )
            logger.info(f"  ✓ Deleted secret: {secret_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                logger.warning(f"  Could not delete secret {secret_name}: {e}")

    logger.info("✓ Secrets processed")


def main():
    """Delete all infrastructure created by installer.py."""
    logger.info("=" * 60)
    logger.info("Starting AWS Infrastructure Cleanup")
    logger.info("=" * 60)
    logger.info(f"Project: {project_name}")
    logger.info(f"Region: {region}")
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Bucket Name: {bucket_name}")
    logger.info("=" * 60)

    parser = argparse.ArgumentParser(
        description="Delete AWS infrastructure created by installer.py"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the main confirmation prompt and proceed with deletion",
    )
    parser.add_argument(
        "--delete-s3",
        action="store_true",
        help="Delete the S3 bucket without prompting (default: keep bucket)",
    )
    parser.add_argument(
        "--delete-cloudfront",
        action="store_true",
        help="Delete CloudFront distribution and OAI without prompting (default: keep)",
    )
    parser.add_argument(
        "--no-wait-opensearch",
        action="store_true",
        help="Do not wait for OpenSearch domain deletion to finish (default: wait)",
    )
    parser.add_argument(
        "--opensearch-wait-timeout",
        type=int,
        default=1800,
        help="Max seconds to wait for OpenSearch domain deletion (default: 1800)",
    )
    parser.add_argument(
        "--delete-agentcore-gateway",
        action="store_true",
        help=(
            "Delete AgentCore gateway-websearch without a separate confirmation prompt "
            "(default: ask, default answer no)"
        ),
    )
    args = parser.parse_args()

    if not args.yes:
        print("\n" + "=" * 60)
        print("WARNING: This will delete AWS resources created by installer.py")
        print("=" * 60)
        print(f"  Project:     {project_name}")
        print(f"  Region:      {region}")
        print(f"  S3 bucket:   {bucket_name}")
        print("")
        print("  S3 and CloudFront are optional (you will be asked; default: keep).")
        print("=" * 60)
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Uninstallation cancelled.")
            sys.exit(0)

    delete_s3 = args.delete_s3
    delete_cloudfront = args.delete_cloudfront
    if not delete_s3:
        delete_s3 = _prompt_yes_no(
            f"Delete S3 bucket '{bucket_name}' and all its objects?",
            default=False,
        )
    if not delete_cloudfront:
        delete_cloudfront = _prompt_yes_no(
            "Delete CloudFront distribution and Origin Access Identity?",
            default=False,
        )

    logger.info(f"Delete S3 bucket: {delete_s3}")
    logger.info(f"Delete CloudFront: {delete_cloudfront}")

    start_time = time.time()

    try:
        # Reverse dependency order of installer.main()
        if delete_cloudfront:
            disable_cloudfront_distributions()
        else:
            logger.info("Skipping CloudFront disable (not requested)")

        delete_knowledge_bases()
        delete_opensearch_domain(
            wait=not args.no_wait_opensearch,
            wait_timeout=args.opensearch_wait_timeout,
        )
        delete_lambda_s3_event_manager()
        agentcore_gateway_deleted = delete_agentcore_websearch_gateway(
            skip_confirmation=args.delete_agentcore_gateway
        )
        delete_iam_roles(delete_agentcore_gateway_role=agentcore_gateway_deleted)

        if delete_s3:
            delete_s3_buckets()
        else:
            logger.info("Skipping S3 bucket deletion (not requested)")

        delete_secrets()

        if delete_cloudfront:
            wait_for_cloudfront_disabled()
            delete_cloudfront_distributions()
            delete_cloudfront_oai()
        else:
            logger.info("Skipping CloudFront deletion (not requested)")

        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("Infrastructure Cleanup Completed!")
        logger.info("=" * 60)
        logger.info(f"Total cleanup time: {elapsed_time / 60:.2f} minutes")
        if delete_cloudfront:
            logger.info(
                "Note: If CloudFront deletion was skipped, re-run with CloudFront "
                "deletion enabled after distributions are fully disabled"
            )
        logger.info("=" * 60)

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("")
        logger.error("=" * 60)
        logger.error("Cleanup Failed!")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error(f"Cleanup time before failure: {elapsed_time / 60:.2f} minutes")
        logger.error("=" * 60)
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
