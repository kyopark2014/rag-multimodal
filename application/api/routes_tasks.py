from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from application.api.routes_auth import require_user_id, _llm_gateway_ready
from application import task_store
from application import utils

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

_LLM_GATEWAY_NOT_CONFIGURED = (
    "LLM Gateway가 설정되어 있지 않아 활성화할 수 없습니다."
)


class TaskCreate(BaseModel):
    model_name: str | None = None
    skills: list[str] | None = None
    mcp_servers: list[str] | None = None
    # None -> per-user settings (applies to all tasks when toggled in UI).
    guardrail_enabled: bool | None = None
    llm_gateway_enabled: bool | None = None
    memory_enabled: bool | None = None
    title: str = "New task"


class TaskPatch(BaseModel):
    title: str | None = None
    model_name: str | None = None
    skills: list[str] | None = None
    mcp_servers: list[str] | None = None
    guardrail_enabled: bool | None = None
    llm_gateway_enabled: bool | None = None
    memory_enabled: bool | None = None
    pinned: bool | None = None


def _require_gateway_if_enabling(enabled: bool | None) -> None:
    if enabled and not _llm_gateway_ready():
        raise HTTPException(status_code=400, detail=_LLM_GATEWAY_NOT_CONFIGURED)


@router.get("")
def list_tasks(request: Request, limit: int = 100):
    user_id = require_user_id(request)
    return {"tasks": task_store.list_tasks(user_id, limit=limit)}


@router.post("")
def create_task(body: TaskCreate, request: Request):
    user_id = require_user_id(request)
    flags = utils.get_user_task_feature_flags(user_id)
    guardrail_enabled = (
        flags["guardrail_enabled"]
        if body.guardrail_enabled is None
        else body.guardrail_enabled
    )
    memory_enabled = (
        flags["memory_enabled"]
        if body.memory_enabled is None
        else body.memory_enabled
    )
    if body.llm_gateway_enabled is None:
        # Inherit Settings; soft-disable when gateway credentials are missing.
        llm_gateway_enabled = flags["llm_gateway_enabled"] and _llm_gateway_ready()
    else:
        llm_gateway_enabled = body.llm_gateway_enabled
        _require_gateway_if_enabling(llm_gateway_enabled)
    preferred_model = utils.get_user_preferred_model(user_id)
    model_name = body.model_name or preferred_model
    task = task_store.create_task(
        user_id,
        model_name=model_name,
        skills=body.skills,
        mcp_servers=body.mcp_servers,
        guardrail_enabled=guardrail_enabled,
        llm_gateway_enabled=llm_gateway_enabled,
        memory_enabled=memory_enabled,
        title=body.title,
    )
    return task


@router.get("/{task_id}")
def get_task(task_id: str, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}")
def patch_task(task_id: str, body: TaskPatch, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    patch = body.model_dump(exclude_unset=True)
    _require_gateway_if_enabling(patch.get("llm_gateway_enabled"))

    # Model / Guardrail / Memory / LLM Gateway apply to every task for this user.
    global_fields = {
        key: patch.pop(key)
        for key in (
            "model_name",
            "guardrail_enabled",
            "memory_enabled",
            "llm_gateway_enabled",
        )
        if key in patch
    }
    if global_fields:
        task_store.update_all_tasks(user_id, **global_fields)
        utils.save_user_task_feature_flags(user_id, **global_fields)

    updated = (
        task_store.update_task(task_id, user_id, **patch)
        if patch
        else task_store.get_task(task_id, user_id)
    )
    return updated


@router.delete("/{task_id}")
def remove_task(task_id: str, request: Request):
    user_id = require_user_id(request)
    if not task_store.delete_task(task_id, user_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}


@router.get("/{task_id}/messages")
def get_messages(task_id: str, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"messages": task_store.list_messages(task_id, user_id)}
