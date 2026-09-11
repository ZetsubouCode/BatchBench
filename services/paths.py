from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - python-dotenv is a runtime dependency.
    load_dotenv = None  # type: ignore

_ENV_LOADED = False

_DATA_MARKER_WEIGHTS = {
    "data/tag_catalog/danbooru_tags.sqlite3": 100,
    "data/tag_catalog/danbooru_tags.csv": 90,
    "data/settings/tag_suggestions.json": 50,
    "_config/review_quiz.json": 40,
    "settings/tagging_quiz.json": 40,
    "tag_editor_glossary.json": 10,
}


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parent.parent


def _user_data_score(path: Path) -> int:
    return sum(weight for marker, weight in _DATA_MARKER_WEIGHTS.items() if (path / marker).exists())


def _candidate_roots(*, include_resource: bool = True) -> Iterable[Path]:
    cwd = Path.cwd().resolve()
    yield cwd

    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        # Common local build layout: <repo>/dist/BatchBench/BatchBench.exe.
        yield exe_dir.parent.parent
        yield exe_dir.parent
        yield exe_dir

    if include_resource:
        yield resource_root()


def _load_env_files() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    if load_dotenv is None:
        return
    seen: set[Path] = set()
    for root in _candidate_roots(include_resource=True):
        env_path = root / ".env"
        try:
            resolved = env_path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)


def user_data_root() -> Path:
    _load_env_files()
    configured = os.getenv("BATCHBENCH_DATA_DIR", "").strip() or os.getenv("BATCHBENCH_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    best_root: Path | None = None
    best_score = 0
    for candidate in _candidate_roots(include_resource=not is_frozen()):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        score = _user_data_score(resolved)
        if score > best_score:
            best_root = resolved
            best_score = score
    if best_root is not None:
        return best_root

    if is_frozen():
        return Path(sys.executable).resolve().parent
    return resource_root()


def user_path(*parts: str) -> Path:
    return user_data_root().joinpath(*parts)


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)
