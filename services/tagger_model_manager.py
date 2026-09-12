from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.paths import user_path
from services.tagger_models.registry import ModelProfile, get_model_profile, list_model_profiles


MODEL_ROOT_PARTS = ("models", "taggers")
_JOBS: Dict[str, "DownloadJob"] = {}
_LOCK = threading.Lock()


def model_root() -> Path:
    return user_path(*MODEL_ROOT_PARTS)


def model_dir(profile_key: str) -> Path:
    return model_root() / get_model_profile(profile_key).key


def _profile_status(profile: ModelProfile) -> Dict[str, Any]:
    target = model_dir(profile.key)
    status = profile.adapter().model_status(target)
    return {
        "key": profile.key,
        "display_name": profile.display_name,
        "repo_id": profile.repo_id,
        "family": profile.family,
        "runtime": profile.runtime,
        "recommended": profile.recommended,
        "legacy_default": profile.legacy_default,
        **status,
    }


def list_status() -> List[Dict[str, Any]]:
    return [_profile_status(profile) for profile in list_model_profiles()]


def install_from_local(profile_key: str, source: Path) -> Dict[str, Any]:
    profile = get_model_profile(profile_key)
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        return {"ok": False, "error": f"Local model folder not found: {source}"}
    missing = profile.adapter().validate_model_dir(source)
    if missing:
        return {"ok": False, "error": f"Local model folder is incomplete; missing: {', '.join(missing)}"}

    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    target = model_dir(profile.key)
    staging = root / f".{profile.key}.install-{uuid.uuid4().hex}"
    previous = root / f".{profile.key}.previous-{uuid.uuid4().hex}"
    copied: List[str] = []
    try:
        staging.mkdir(parents=True)
        for name in profile.required_files:
            shutil.copy2(source / name, staging / name)
            copied.append(name)
        if target.exists():
            target.replace(previous)
        staging.replace(target)
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
    except Exception as exc:
        if not target.exists() and previous.exists():
            previous.replace(target)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        return {"ok": False, "error": f"Could not install local model: {exc}", "copied": copied}
    return {
        "ok": True,
        "profile": profile.key,
        "path": str(target),
        "copied": copied,
        "logs": [
            f"[model] Installing {profile.display_name} from local folder",
            *[f"[model] Installed {name}" for name in copied],
            "[model] Model ready",
        ],
    }


def _friendly_download_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if any(token in lowered for token in ("gated", "401", "403", "unauthorized", "forbidden", "access to model")):
        return (
            "Model access is not authorized. Accept the model conditions on Hugging Face and "
            "authenticate BatchBench/Hugging Face, then retry."
        )
    return f"Model download failed: {text}"


@dataclass
class DownloadJob:
    job_id: str
    profile_key: str
    status: str = "queued"
    current_file: str = ""
    completed_files: int = 0
    total_files: int = 0
    error: str = ""
    logs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "profile": self.profile_key,
            "status": self.status,
            "current_file": self.current_file,
            "completed_files": self.completed_files,
            "total_files": self.total_files,
            "error": self.error,
            "logs": list(self.logs),
            "created_at": self.created_at,
        }


def _run_download(job: DownloadJob) -> None:
    profile = get_model_profile(job.profile_key)
    job.status = "running"
    job.total_files = len(profile.required_files)
    job.logs.append(f"[model] Downloading {profile.display_name}")
    staging: Optional[Path] = None
    try:
        from huggingface_hub import hf_hub_download

        root = model_root()
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".{profile.key}.download-{job.job_id}"
        staging.mkdir(parents=True, exist_ok=True)
        for name in profile.required_files:
            job.current_file = name
            downloaded = Path(hf_hub_download(repo_id=profile.repo_id, filename=name))
            shutil.copy2(downloaded, staging / name)
            job.completed_files += 1
            job.logs.append(f"[model] Downloaded {name}")
        result = install_from_local(profile.key, staging)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Downloaded model validation failed")
        job.current_file = ""
        job.status = "completed"
        job.logs.append("[model] Model ready")
    except Exception as exc:
        job.status = "failed"
        job.error = _friendly_download_error(exc)
        job.logs.append(f"[model] {job.error}")
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def start_download(profile_key: str) -> Dict[str, Any]:
    profile = get_model_profile(profile_key)
    if _profile_status(profile)["ready"]:
        return {"ok": True, "already_ready": True, "status": _profile_status(profile)}
    job = DownloadJob(uuid.uuid4().hex, profile.key)
    with _LOCK:
        _JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_download, args=(job,), name=f"TaggerModelDownload-{profile.key}", daemon=True)
    thread.start()
    return {"ok": True, "job": job.payload()}


def download_status(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
    return job.payload() if job else None
