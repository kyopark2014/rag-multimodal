import json
import os
import traceback

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth
from urllib.parse import unquote_plus

s3_client = boto3.client("s3")

s3_bucket = os.environ.get("s3_bucket")
s3_prefix = os.environ.get("s3_prefix", "docs")
meta_prefix = os.environ.get("meta_prefix", "metadata/")
opensearch_url = os.environ.get("opensearch_url")
vector_index_name = os.environ.get("vectorIndexName")
region = os.environ.get("AWS_REGION") or os.environ.get("region", "us-west-2")

_os_client = None


def _get_os_client() -> OpenSearch:
    global _os_client
    if _os_client is not None:
        return _os_client

    if not opensearch_url:
        raise ValueError("opensearch_url environment variable is required")

    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token,
    )
    _os_client = OpenSearch(
        hosts=[{"host": opensearch_url.replace("https://", ""), "port": 443}],
        http_compress=True,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )
    return _os_client


def _object_name_from_key(key: str) -> str:
    """Derive object name under docs/ (application/lambda_function.py)."""
    if key.find(s3_prefix) != -1:
        return key[key.find(s3_prefix) + len(s3_prefix) + 1 :]
    return key


def metadata_key_for_doc_key(key: str) -> str:
    object_name = _object_name_from_key(key)
    return f"{meta_prefix}{object_name}.metadata.json"


def _delete_ids_from_opensearch(ids: list) -> tuple:
    if not ids:
        return 0, []
    client = _get_os_client()
    actions = [
        {"_op_type": "delete", "_index": vector_index_name, "_id": doc_id}
        for doc_id in ids
    ]
    success, errors = bulk(client, actions, raise_on_error=False)
    print("delete ids in opensearch: ", success, errors)
    return success, errors


def delete_document_if_exist(metadata_key: str) -> bool:
    """
    Read metadata JSON, delete OpenSearch document ids and related S3 files.
    Mirrors application/lambda_function.py delete_document_if_exist.
    """
    try:
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        print("objs: ", objs)

        if not objs:
            print("no meta file: ", metadata_key)
            return False

        doc = s3r.Object(s3_bucket, metadata_key)
        meta = doc.get()["Body"].read().decode("utf-8")
        print("meta: ", meta)

        meta_json = json.loads(meta)
        ids = meta_json["ids"]
        print("ids: ", ids)

        _delete_ids_from_opensearch(ids)

        files = meta_json.get("files", [])
        print("files: ", files)

        for file_key in files:
            s3r.Object(s3_bucket, file_key).delete()
            print("delete file: ", file_key)

        return True
    except Exception:
        err_msg = traceback.format_exc()
        print("error message: ", err_msg)
        raise Exception("Not able to delete document from metadata")


def handle_pdf_delete(bucket: str, key: str) -> None:
    """On PDF removal under docs/, purge OpenSearch vectors via metadata file."""
    key = unquote_plus(key)
    file_type = key[key.rfind(".") + 1 :].lower()
    if file_type != "pdf":
        print("skip non-pdf delete: ", file_type)
        return

    if f"{s3_prefix}/" not in key and not key.startswith(f"{s3_prefix}/"):
        print("skip delete outside docs prefix: ", key)
        return

    metadata_key = metadata_key_for_doc_key(key)
    print("metadata_key: ", metadata_key)

    try:
        metadata_obj = s3_client.get_object(Bucket=bucket, Key=metadata_key)
        metadata_body = metadata_obj["Body"].read().decode("utf-8")
        metadata = json.loads(metadata_body)
        print("metadata: ", metadata)
        document_id = metadata.get("DocumentId")
        print("documentId: ", document_id)
    except Exception:
        print("err_msg: ", traceback.format_exc())
        return

    if not document_id:
        return

    try:
        if delete_document_if_exist(metadata_key):
            print("delete metadata: ", metadata_key)
            s3_client.delete_object(Bucket=bucket, Key=metadata_key)
    except Exception:
        print("err_msg: ", traceback.format_exc())
        raise


def lambda_handler(event, context):
    print("event: ", json.dumps(event))

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        event_name = record["eventName"]
        print("bucket: ", bucket)
        print("key: ", key)
        print("eventName: ", event_name)

        if event_name.startswith("ObjectRemoved"):
            try:
                handle_pdf_delete(bucket, key)
            except Exception as exc:
                print("Fail to handle pdf delete: ", exc)

    return {"statusCode": 200}
