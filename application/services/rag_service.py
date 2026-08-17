"""RAG upload orchestration: S3 ingest + multimodal OCR → OpenSearch indexing.

Sidecar ``{file}.metadata.json`` follows Bedrock KB document metadata fields
(same shape as rag-foundation-model) so owner/team/created_time/is_confidential
can be loaded when indexing into managed OpenSearch.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Sequence

from application import multimodal, utils

logger = logging.getLogger("rag_service")

DEFAULT_TEAM = "mycompany"
DEFAULT_IS_CONFIDENTIAL = False

_sync_lock = threading.Lock()
_sync_in_progress = False


class RagServiceError(Exception):
    """Business failure while uploading or syncing a RAG document."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _metadata_attr(
    attr_type: str,
    *,
    string_value: str | None = None,
    number_value: float | int | None = None,
    boolean_value: bool | None = None,
    string_list_value: Sequence[str] | None = None,
    include_for_embedding: bool = False,
) -> dict[str, Any]:
    """Build one Bedrock KB sidecar metadata attribute."""
    value: dict[str, Any] = {"type": attr_type}
    if attr_type == "STRING":
        value["stringValue"] = string_value or ""
    elif attr_type == "NUMBER":
        value["numberValue"] = number_value if number_value is not None else 0
    elif attr_type == "BOOLEAN":
        value["booleanValue"] = bool(boolean_value)
    elif attr_type == "STRING_LIST":
        value["stringListValue"] = list(string_list_value or [])
    else:
        raise ValueError(f"Unsupported metadata type: {attr_type}")

    return {
        "value": value,
        "includeForEmbedding": include_for_embedding,
    }


def build_kb_metadata_document(
    *,
    owners: Sequence[str],
    team: str = DEFAULT_TEAM,
    is_confidential: bool = DEFAULT_IS_CONFIDENTIAL,
    created_time: int | float | None = None,
) -> dict[str, Any]:
    """Return Bedrock Knowledge Base ``.metadata.json`` body for filtering.

    ``owner`` uses STRING_LIST; ``created_time`` is Unix epoch seconds (NUMBER).
    """
    owner_list = [o.strip() for o in owners if o and str(o).strip()]
    if not owner_list:
        raise ValueError("At least one owner is required")

    if created_time is None:
        created_time = int(datetime.now(timezone.utc).timestamp())
    else:
        created_time = int(created_time)

    return {
        "metadataAttributes": {
            "owner": _metadata_attr(
                "STRING_LIST",
                string_list_value=owner_list,
                include_for_embedding=False,
            ),
            "team": _metadata_attr(
                "STRING",
                string_value=team,
                include_for_embedding=False,
            ),
            "created_time": _metadata_attr(
                "NUMBER",
                number_value=created_time,
                include_for_embedding=False,
            ),
            "is_confidential": _metadata_attr(
                "BOOLEAN",
                boolean_value=is_confidential,
                include_for_embedding=False,
            ),
        }
    }


def flatten_kb_metadata_attributes(doc: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten Bedrock KB ``metadataAttributes`` into OpenSearch-friendly fields.

    ``owner`` STRING_LIST becomes the first owner string (user_id) for Document
    metadata used by ``mcp_rag_opensearch._access_metadata``.
    """
    if not doc or not isinstance(doc, dict):
        return {}

    attrs = doc.get("metadataAttributes")
    if not isinstance(attrs, dict):
        # Already flat (legacy / OpenSearch-style)
        flat: dict[str, Any] = {}
        for key in ("owner", "team", "created_time", "is_confidential"):
            if key in doc:
                flat[key] = doc[key]
        return flat

    result: dict[str, Any] = {}
    for key, attr in attrs.items():
        if not isinstance(attr, dict):
            continue
        value = attr.get("value") or {}
        if not isinstance(value, dict):
            continue
        attr_type = value.get("type")
        if attr_type == "STRING":
            result[key] = value.get("stringValue", "") or ""
        elif attr_type == "NUMBER":
            result[key] = value.get("numberValue", 0)
        elif attr_type == "BOOLEAN":
            result[key] = bool(value.get("booleanValue"))
        elif attr_type == "STRING_LIST":
            lst = [
                str(v).strip()
                for v in (value.get("stringListValue") or [])
                if v is not None and str(v).strip()
            ]
            if key == "owner":
                result[key] = lst[0] if lst else ""
                if lst:
                    result["owners"] = lst
            else:
                result[key] = lst
    return result


def load_docs_sidecar_metadata(pdf_s3_key: str) -> dict[str, Any]:
    """Load ``{pdf_s3_key}.metadata.json`` from S3 and return flattened fields."""
    if not pdf_s3_key or not utils.s3_bucket:
        return {}

    sidecar_key = (
        pdf_s3_key
        if pdf_s3_key.endswith(".metadata.json")
        else f"{pdf_s3_key}.metadata.json"
    )
    try:
        import boto3

        client = boto3.client("s3", region_name=utils.bedrock_region)
        obj = client.get_object(Bucket=utils.s3_bucket, Key=sidecar_key)
        body = obj["Body"].read().decode("utf-8")
        doc = json.loads(body)
        flat = flatten_kb_metadata_attributes(doc)
        logger.info(
            "Loaded docs sidecar metadata key=%s fields=%s",
            sidecar_key,
            {k: flat.get(k) for k in ("owner", "team", "created_time", "is_confidential")},
        )
        return flat
    except Exception:
        logger.warning(
            "Docs sidecar metadata not found or unreadable: s3://%s/%s",
            utils.s3_bucket,
            sidecar_key,
            exc_info=True,
        )
        return {}


def ingest_rag_upload(
    file_bytes: bytes,
    file_name: str,
    user_id: str,
    *,
    owners: Sequence[str] | None = None,
    team: str = DEFAULT_TEAM,
    is_confidential: bool = DEFAULT_IS_CONFIDENTIAL,
) -> dict[str, Any]:
    """Upload PDF + ``{file}.metadata.json`` then run multimodal OpenSearch indexing."""
    global _sync_in_progress
    if not _sync_lock.acquire(blocking=False):
        raise RagServiceError(
            409,
            "현재 이전에 업로드된 파일을 처리하고 있습니다. 조금후 다시 시도해주세요.",
        )
    if _sync_in_progress:
        _sync_lock.release()
        raise RagServiceError(
            409,
            "현재 이전에 업로드된 파일을 처리하고 있습니다. 조금후 다시 시도해주세요.",
        )

    _sync_in_progress = True
    try:
        try:
            upload_result = utils.upload_to_s3(file_bytes, file_name, user_id=user_id)
        except Exception:
            logger.exception("S3 upload failed for file=%s user=%s", file_name, user_id)
            raise RagServiceError(500, "Failed to upload file to S3") from None
        if not upload_result:
            raise RagServiceError(500, "Failed to upload file to S3")

        owner_list = list(owners) if owners else [user_id]
        try:
            metadata_doc = build_kb_metadata_document(
                owners=owner_list,
                team=team or DEFAULT_TEAM,
                is_confidential=(
                    DEFAULT_IS_CONFIDENTIAL
                    if is_confidential is None
                    else bool(is_confidential)
                ),
            )
        except ValueError as exc:
            raise RagServiceError(400, str(exc)) from exc

        metadata_file_name = f"{file_name}.metadata.json"
        metadata_bytes = json.dumps(metadata_doc, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        try:
            metadata_result = utils.upload_to_s3(
                metadata_bytes,
                metadata_file_name,
                user_id=user_id,
            )
        except Exception:
            logger.exception(
                "S3 metadata upload failed for file=%s user=%s",
                metadata_file_name,
                user_id,
            )
            raise RagServiceError(
                500,
                "Failed to upload document metadata file to S3",
            ) from None
        if not metadata_result:
            raise RagServiceError(
                500,
                "Failed to upload document metadata file to S3",
            )

        file_url = upload_result.get("url")
        if not file_url:
            s3_bucket = utils.s3_bucket
            s3_key = upload_result.get("s3_key")
            if s3_bucket and s3_key:
                file_url = f"s3://{s3_bucket}/{s3_key}"
            else:
                raise RagServiceError(
                    500,
                    "File uploaded but sharing URL is not configured",
                )

        try:
            extracted = multimodal.sync_data_source(
                file_url,
                access_metadata=flatten_kb_metadata_attributes(metadata_doc),
                user_id=user_id,
            )
        except Exception:
            logger.exception("Multimodal OpenSearch sync failed for file=%s", file_name)
            raise RagServiceError(
                500,
                "File uploaded but multimodal OpenSearch indexing failed",
            ) from None
        if not extracted:
            raise RagServiceError(
                500,
                "File uploaded but multimodal OpenSearch indexing returned no content",
            )

        preview = extracted if isinstance(extracted, str) else str(extracted)
        if len(preview) > 2000:
            preview = preview[:2000] + "\n…"

        logger.info(
            "RAG upload complete: user=%s file=%s s3_key=%s metadata_key=%s",
            user_id,
            file_name,
            upload_result.get("s3_key"),
            metadata_result.get("s3_key"),
        )

        return {
            "ok": True,
            "file_name": upload_result["file_name"],
            "s3_key": upload_result["s3_key"],
            "metadata_file_name": metadata_result["file_name"],
            "metadata_s3_key": metadata_result["s3_key"],
            "metadata": metadata_doc,
            "user_id": user_id,
            "url": upload_result.get("url"),
            "docs_prefix": upload_result.get("docs_prefix"),
            "s3_docs_prefix": upload_result.get("s3_docs_prefix"),
            "extracted_preview": preview,
            "sync": {"status": "COMPLETE", "backend": "multimodal-opensearch"},
            "message": (
                f'"{file_name}"가 S3에 업로드되었고 멀티모달 OCR 후 OpenSearch에 인덱싱되었습니다.'
            ),
        }
    finally:
        _sync_in_progress = False
        _sync_lock.release()
