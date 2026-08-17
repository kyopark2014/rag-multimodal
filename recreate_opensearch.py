#!/usr/bin/env python3
"""Delete and recreate only the managed OpenSearch domain (rag-multimodal).

Uses uninstaller.delete_opensearch_domain + installer create/index/FGAC/lambda
helpers. Other resources (S3, CloudFront, AgentCore) are left untouched.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recreate_opensearch")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "application" / "config.json"

# Known Dashboards SSO principal (path-qualified + short form)
SSO_ROLE_ARNS = [
    (
        "arn:aws:iam::262976740991:role/aws-reserved/sso.amazonaws.com/"
        "ap-northeast-1/AWSReservedSSO_AdministratorAccess_7f581214b7ef7a2d"
    ),
    "arn:aws:iam::262976740991:role/AWSReservedSSO_AdministratorAccess_7f581214b7ef7a2d",
]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from uninstaller import delete_opensearch_domain
    import installer as inst

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    password = (cfg.get("managed_opensearch_dashboards_password") or "").strip()
    if not password:
        logger.error(
            "managed_opensearch_dashboards_password missing in %s", CONFIG_PATH
        )
        return 1

    s3_bucket = cfg.get("s3_bucket") or inst.bucket_name
    start = time.time()

    logger.info("=" * 60)
    logger.info("Deleting managed OpenSearch domain: %s", inst.opensearch_domain_name)
    logger.info("=" * 60)
    delete_opensearch_domain(wait=True, wait_timeout=3600)

    logger.info("=" * 60)
    logger.info("Creating managed OpenSearch domain: %s", inst.opensearch_domain_name)
    logger.info("=" * 60)
    opensearch_info = inst.create_managed_opensearch_domain(password)
    password = inst.ensure_opensearch_master_password_works(
        opensearch_info["endpoint"],
        opensearch_info["domain_name"],
        password,
    )

    map_arns: set[str] = set(inst._caller_backend_role_arns())
    map_arns.update(SSO_ROLE_ARNS)
    try:
        lambda_role_arn = inst.iam_client.get_role(
            RoleName=inst.LAMBDA_S3_EVENT_ROLE_NAME
        )["Role"]["Arn"]
        map_arns.add(inst._normalize_iam_arn_for_backend_role(lambda_role_arn))
    except ClientError as exc:
        logger.warning("Could not resolve Lambda role for FGAC mapping: %s", exc)

    inst.ensure_opensearch_backend_role_mappings(
        opensearch_info["endpoint"],
        password,
        sorted(map_arns),
    )

    nori_ready = inst.ensure_analysis_nori_plugin(
        opensearch_info["domain_name"],
        engine_version=opensearch_info.get("engine_version"),
        endpoint_url=opensearch_info["endpoint"],
        master_password=password,
    )
    inst.ensure_opensearch_index(
        opensearch_info["endpoint"],
        use_nori=nori_ready,
        master_password=password,
    )

    logger.info("Updating lambda-s3-event-manager for new OpenSearch endpoint")
    lambda_info = inst.deploy_lambda_s3_event_manager(
        s3_bucket,
        opensearch_info["endpoint"],
        opensearch_info["arn"],
        opensearch_master_password=password,
    )

    cfg.update(
        {
            "managed_opensearch_url": opensearch_info["endpoint"],
            "managed_opensearch_arn": opensearch_info["arn"],
            "managed_opensearch_dashboards_url": opensearch_info["dashboards_url"],
            "managed_opensearch_dashboards_user": inst.OPENSEARCH_MASTER_USERNAME,
            "managed_opensearch_dashboards_password": password,
            "lambda_s3_event_manager_arn": lambda_info["function_arn"],
            "lambda_s3_event_manager_name": lambda_info["function_name"],
            "s3_docs_prefix": inst.S3_DOCS_PREFIX,
        }
    )
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    logger.info("Updated %s", CONFIG_PATH)

    elapsed = (time.time() - start) / 60
    logger.info("=" * 60)
    logger.info("OpenSearch recreate completed in %.1f minutes", elapsed)
    logger.info("  Domain: %s", opensearch_info["endpoint"])
    logger.info("  Dashboards: %s", opensearch_info["dashboards_url"])
    logger.info("  Index: %s (nori=%s)", inst.project_name, nori_ready)
    logger.info("=" * 60)
    logger.info(
        "Re-upload RAG documents to rebuild the index "
        "(previous vectors were deleted with the domain)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
