"""RAG upload orchestration: S3 ingest + multimodal OCR → OpenSearch indexing."""

from __future__ import annotations

import logging
import threading
from typing import Any

from application import multimodal, utils

logger = logging.getLogger("rag_service")

_sync_lock = threading.Lock()
_sync_in_progress = False


class RagServiceError(Exception):
    """Business failure while uploading or syncing a RAG document."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def ingest_rag_upload(
    file_bytes: bytes,
    file_name: str,
    user_id: str,
    *,
    owners: list[str] | None = None,
    team: str | None = None,
    is_confidential: bool | None = None,
) -> dict[str, Any]:
    """Upload PDF to S3 then run multimodal.sync_data_source (OCR → OpenSearch).

    ``owners`` / ``team`` / ``is_confidential`` are accepted for API parity with
    other RAG repos but are unused by the managed-OpenSearch pipeline.
    """
    _ = owners, team, is_confidential

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
            # Keep docs/{file} layout for OpenSearch metadata / Lambda compatibility.
            upload_result = utils.upload_to_s3(file_bytes, file_name, user_id=None)
        except Exception:
            logger.exception("S3 upload failed for file=%s user=%s", file_name, user_id)
            raise RagServiceError(500, "Failed to upload file to S3") from None
        if not upload_result:
            raise RagServiceError(500, "Failed to upload file to S3")

        file_url = upload_result.get("url")
        if not file_url:
            # Fall back to s3:// key so multimodal can still fetch the object
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
            extracted = multimodal.sync_data_source(file_url)
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
            "RAG upload complete: user=%s file=%s s3_key=%s",
            user_id,
            file_name,
            upload_result.get("s3_key"),
        )

        return {
            "ok": True,
            "file_name": upload_result["file_name"],
            "s3_key": upload_result["s3_key"],
            "user_id": user_id,
            "url": upload_result.get("url"),
            "extracted_preview": preview,
            "sync": {"status": "COMPLETE", "backend": "multimodal-opensearch"},
            "message": (
                f'"{file_name}"가 S3에 업로드되었고 멀티모달 OCR 후 OpenSearch에 인덱싱되었습니다.'
            ),
        }
    finally:
        _sync_in_progress = False
        _sync_lock.release()
