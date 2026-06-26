from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import danbooru_client


CATALOG_ROOT = Path(__file__).resolve().parent.parent / "data" / "tag_catalog"
CSV_PATH = CATALOG_ROOT / "danbooru_tags.csv"
DB_PATH = CATALOG_ROOT / "danbooru_tags.sqlite3"
STATE_PATH = CATALOG_ROOT / "catalog_state.json"
STAGING_ROOT = CATALOG_ROOT / "_staging"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings" / "tag_suggestions.json"

CSV_FIELDS = ["id", "name", "category", "category_name", "post_count", "is_deprecated", "updated_at"]
MIN_CATALOG_POST_COUNT = 1
CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "deprecated",
    3: "copyright",
    4: "character",
    5: "meta",
}
CATEGORY_IDS = {value: key for key, value in CATEGORY_NAMES.items()}
DEFAULT_SECTIONS = {
    "appearance": {"label": "Appearance", "enabled": True},
    "accessory": {"label": "Accessory", "enabled": True},
    "outfit": {"label": "Outfit", "enabled": True},
    "pose_expression": {"label": "Pose & Expression", "enabled": True},
    "camera_pov": {"label": "Camera & POV", "enabled": True},
    "body_composition": {"label": "Body Composition", "enabled": True},
    "lighting": {"label": "Lighting", "enabled": True},
    "background": {"label": "Background", "enabled": True},
    "manual_tags": {"label": "Manual Tags", "enabled": True},
}
DEFAULT_SETTINGS = {
    "enabled": False,
    "min_post_count": 5,
    "max_suggestions": 12,
    "include_categories": {
        "general": True,
        "character": False,
        "copyright": True,
        "artist": False,
        "meta": True,
    },
    "include_deprecated": False,
    "aliases": {},
    "sections": DEFAULT_SECTIONS,
}

_LOCK = threading.RLock()
_SYNC_THREAD: Optional[threading.Thread] = None
_CANCEL_EVENT = threading.Event()


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_root() -> None:
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_tag_name(raw: Any) -> str:
    tag = str(raw or "").strip().lower()
    if not tag:
        return ""
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    return tag.strip("_")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def category_name(category: Any) -> str:
    return CATEGORY_NAMES.get(_to_int(category, -1), "unknown")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    delay = 0.05
    last_error: Optional[OSError] = None
    for _ in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)
            delay = min(delay * 1.6, 0.5)
        except OSError as exc:
            last_error = exc
            if getattr(exc, "winerror", None) != 5:
                break
            time.sleep(delay)
            delay = min(delay * 1.6, 0.5)
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    if last_error:
        raise last_error


def _append_log(state: Dict[str, Any], message: str) -> None:
    logs = list(state.get("logs") or [])
    logs.append(message)
    state["logs"] = logs[-200:]
    state["message"] = message


def _base_state() -> Dict[str, Any]:
    return {
        "state": "idle",
        "started_at": "",
        "finished_at": "",
        "current_page": 0,
        "fetched_rows": 0,
        "accepted_rows": 0,
        "malformed_rows": 0,
        "duplicate_rows": 0,
        "last_seen_id": 0,
        "message": "",
        "last_error": "",
        "tag_count": 0,
        "last_successful_sync": "",
        "logs": [],
    }


def _load_state() -> Dict[str, Any]:
    state = _base_state()
    state.update(_read_json(STATE_PATH, {}))
    return state


def _save_state(state: Dict[str, Any]) -> None:
    _atomic_write_json(STATE_PATH, state)


def _update_state(**updates: Any) -> Dict[str, Any]:
    with _LOCK:
        _ensure_root()
        state = _load_state()
        state.update(updates)
        _save_state(state)
        return state


def normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = normalize_tag_name(raw.get("name"))
    if not name:
        return None
    category = _to_int(raw.get("category"), 0)
    post_count = max(0, _to_int(raw.get("post_count"), 0))
    if post_count < MIN_CATALOG_POST_COUNT:
        return None
    tag_id = max(0, _to_int(raw.get("id"), 0))
    updated_at = str(raw.get("updated_at") or "").strip()
    return {
        "id": tag_id,
        "name": name,
        "category": category,
        "category_name": category_name(category),
        "post_count": post_count,
        "is_deprecated": category == 2,
        "updated_at": updated_at,
    }


def _prefer_row(current: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    cur_key = (current.get("post_count", 0), str(current.get("updated_at") or ""), current.get("id", 0))
    cand_key = (candidate.get("post_count", 0), str(candidate.get("updated_at") or ""), candidate.get("id", 0))
    return candidate if cand_key > cur_key else current


def _merge_records(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    merged: Dict[str, Dict[str, Any]] = {}
    stats = {"accepted": 0, "malformed": 0, "duplicates": 0}
    for raw in records:
        row = normalize_record(raw)
        if not row:
            stats["malformed"] += 1
            continue
        name = row["name"]
        if name in merged:
            stats["duplicates"] += 1
            merged[name] = _prefer_row(merged[name], row)
        else:
            merged[name] = row
        stats["accepted"] = len(merged)
    return sorted(merged.values(), key=lambda item: item["name"]), stats


def _read_csv_records(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"CSV header must be exactly: {','.join(CSV_FIELDS)}")
        for row in reader:
            category = _to_int(row.get("category"), 0)
            records.append(
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "category": category,
                    "post_count": row.get("post_count"),
                    "updated_at": row.get("updated_at") or "",
                }
            )
            if str(row.get("category_name") or "").strip() and row.get("category_name") != category_name(category):
                warnings.append(f"Category name corrected for {row.get('name') or '(blank)'}.")
    merged, stats = _merge_records(records)
    return merged, stats, warnings[:20]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "category": int(row["category"]),
                    "category_name": row["category_name"],
                    "post_count": int(row["post_count"]),
                    "is_deprecated": _bool_text(bool(row["is_deprecated"])),
                    "updated_at": row.get("updated_at") or "",
                }
            )
            count += 1
    return count


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=DELETE")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS tag;
        CREATE TABLE tag (
            id INTEGER NOT NULL,
            name TEXT PRIMARY KEY,
            category INTEGER NOT NULL,
            category_name TEXT NOT NULL,
            post_count INTEGER NOT NULL DEFAULT 0,
            is_deprecated INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX idx_tag_name ON tag(name);
        CREATE INDEX idx_tag_post_count ON tag(post_count DESC);
        CREATE INDEX idx_tag_category ON tag(category);
        """
    )


def _insert_rows(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> int:
    batch = []
    count = 0
    for row in rows:
        batch.append(
            (
                int(row["id"]),
                row["name"],
                int(row["category"]),
                row["category_name"],
                int(row["post_count"]),
                1 if row["is_deprecated"] else 0,
                row.get("updated_at") or "",
            )
        )
        if len(batch) >= 5000:
            conn.executemany("INSERT OR REPLACE INTO tag VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            count += len(batch)
            batch.clear()
    if batch:
        conn.executemany("INSERT OR REPLACE INTO tag VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        count += len(batch)
    conn.commit()
    return count


def build_sqlite(rows: Iterable[Dict[str, Any]], db_path: Path) -> int:
    conn = _connect(db_path)
    try:
        _init_db(conn)
        return _insert_rows(conn, rows)
    finally:
        conn.close()


def _db_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM tag").fetchone()[0])
    finally:
        conn.close()


def _backup_existing() -> None:
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    for path in (CSV_PATH, DB_PATH):
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + f".{stamp}.bak"))


def _swap_in(staged_csv: Path, staged_db: Path, tag_count: int) -> None:
    _ensure_root()
    _backup_existing()
    os.replace(staged_csv, CSV_PATH)
    os.replace(staged_db, DB_PATH)
    state = _load_state()
    state.update(
        {
            "state": "completed",
            "finished_at": _utc_now(),
            "tag_count": tag_count,
            "last_successful_sync": _utc_now(),
            "last_error": "",
        }
    )
    _append_log(state, "[sync] Catalog swap complete.")
    _append_log(state, "[sync] Finished successfully.")
    _save_state(state)


def _status_label(csv_exists: bool, db_exists: bool, state_name: str) -> str:
    if state_name == "running":
        return "Syncing"
    if state_name == "failed":
        return "Failed"
    if state_name == "cancelled":
        return "Cancelled"
    if csv_exists and db_exists:
        return "Ready"
    if csv_exists or db_exists:
        return "Partial"
    return "Not installed"


def get_catalog_status() -> Dict[str, Any]:
    _ensure_root()
    state = _load_state()
    running_thread = bool(_SYNC_THREAD and _SYNC_THREAD.is_alive())
    if str(state.get("state") or "") == "running" and not running_thread:
        state.update(
            {
                "state": "failed",
                "finished_at": state.get("finished_at") or _utc_now(),
                "last_error": state.get("last_error")
                or "Catalog sync was interrupted before the staged catalog was swapped in. Run sync again or import a CSV.",
            }
        )
        _append_log(state, "[sync] Marked interrupted: no active sync worker is running.")
        _save_state(state)
    csv_exists = CSV_PATH.exists()
    db_exists = DB_PATH.exists()
    tag_count = 0
    if db_exists:
        try:
            tag_count = _db_count(DB_PATH)
        except Exception:
            tag_count = 0
    return {
        "ok": True,
        "status": _status_label(csv_exists, db_exists, str(state.get("state") or "idle")),
        "ready": csv_exists and db_exists and tag_count > 0,
        "csv_exists": csv_exists,
        "db_exists": db_exists,
        "csv_path": str(CSV_PATH),
        "db_path": str(DB_PATH),
        "tag_count": tag_count or int(state.get("tag_count") or 0),
        "state": state,
        "warnings": [],
        "info": [],
    }


def _staging_dir(prefix: str) -> Path:
    _ensure_root()
    path = STAGING_ROOT / f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run_full_sync(staging: Path) -> None:
    rows_by_name: Dict[str, Dict[str, Any]] = {}
    state = _load_state()
    after_id: Optional[int] = None
    page = 0
    fetched_total = 0
    malformed = 0
    duplicates = 0
    try:
        _append_log(state, "[sync] Started full Danbooru tag sync.")
        _save_state(state)
        while True:
            if _CANCEL_EVENT.is_set():
                raise KeyboardInterrupt()
            page += 1
            payload, err = danbooru_client.fetch_tag_page(after_id=after_id, limit=1000)
            for retry_line in danbooru_client.drain_catalog_retry_logs():
                _append_log(state, retry_line)
            if err:
                raise RuntimeError(err)
            if not payload:
                break
            fetched_total += len(payload)
            accepted_this = 0
            malformed_this = 0
            page_lowest_id: Optional[int] = None
            for raw in payload:
                raw_id = _to_int(raw.get("id") if isinstance(raw, dict) else None, 0)
                if raw_id:
                    page_lowest_id = raw_id if page_lowest_id is None else min(page_lowest_id, raw_id)
                row = normalize_record(raw)
                if not row:
                    malformed += 1
                    malformed_this += 1
                    continue
                name = row["name"]
                if name in rows_by_name:
                    duplicates += 1
                    rows_by_name[name] = _prefer_row(rows_by_name[name], row)
                else:
                    rows_by_name[name] = row
                    accepted_this += 1
            if page_lowest_id is not None:
                after_id = page_lowest_id
            state.update(
                {
                    "state": "running",
                    "current_page": page,
                    "fetched_rows": fetched_total,
                    "accepted_rows": len(rows_by_name),
                    "malformed_rows": malformed,
                    "duplicate_rows": duplicates,
                    "last_seen_id": after_id or 0,
                }
            )
            _append_log(state, f"[sync] Page {page}: fetched {len(payload)} rows, accepted {accepted_this}, skipped {malformed_this} malformed rows.")
            _append_log(state, f"[sync] Last processed tag ID: {after_id or 0}.")
            _save_state(state)

        if _CANCEL_EVENT.is_set():
            raise KeyboardInterrupt()
        rows = sorted(rows_by_name.values(), key=lambda item: item["name"])
        if not rows:
            raise ValueError("Danbooru sync returned no usable tags.")
        staged_csv = staging / "danbooru_tags.csv"
        staged_db = staging / "danbooru_tags.sqlite3"
        _append_log(state, "[sync] Building staged SQLite index.")
        _save_state(state)
        csv_count = _write_csv(staged_csv, rows)
        db_count = build_sqlite(rows, staged_db)
        if csv_count != db_count or db_count <= 0:
            raise ValueError("Staged catalog validation failed.")
        _append_log(state, f"[sync] Validated staged catalog: {db_count:,} tags.")
        _append_log(state, "[sync] Backed up previous CSV and SQLite index.")
        _save_state(state)
        _swap_in(staged_csv, staged_db, db_count)
    except KeyboardInterrupt:
        state = _load_state()
        state.update({"state": "cancelled", "finished_at": _utc_now()})
        _append_log(state, "[sync] Cancelled. Existing catalog left unchanged.")
        _save_state(state)
    except Exception as exc:
        state = _load_state()
        state.update({"state": "failed", "finished_at": _utc_now(), "last_error": str(exc)})
        _append_log(state, f"[sync] Failed: {exc}")
        _save_state(state)


def start_full_sync() -> Dict[str, Any]:
    global _SYNC_THREAD
    with _LOCK:
        if _SYNC_THREAD and _SYNC_THREAD.is_alive():
            return {"ok": False, "error": "A catalog sync is already running.", "warnings": [], "info": []}
        _CANCEL_EVENT.clear()
        staging = _staging_dir("sync")
        state = _base_state()
        state.update({"state": "running", "started_at": _utc_now(), "message": "Starting full sync."})
        _append_log(state, "[sync] Started full Danbooru tag sync.")
        _save_state(state)
        _SYNC_THREAD = threading.Thread(target=_run_full_sync, args=(staging,), daemon=True)
        _SYNC_THREAD.start()
    return {"ok": True, "state": get_sync_status(), "warnings": [], "info": ["Catalog sync started."]}


def get_sync_status() -> Dict[str, Any]:
    status = get_catalog_status()
    status["running"] = bool(_SYNC_THREAD and _SYNC_THREAD.is_alive())
    return status


def cancel_sync() -> Dict[str, Any]:
    if not (_SYNC_THREAD and _SYNC_THREAD.is_alive()):
        return {"ok": True, "cancelled": False, "warnings": [], "info": ["No active sync."]}
    _CANCEL_EVENT.set()
    state = _update_state(message="Cancellation requested.")
    _append_log(state, "[sync] Cancellation requested.")
    _save_state(state)
    return {"ok": True, "cancelled": True, "warnings": [], "info": ["Cancellation requested."]}


def import_csv(source_path: Path) -> Dict[str, Any]:
    _ensure_root()
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        return {"ok": False, "error": "CSV file not found.", "warnings": [], "info": []}
    staging = _staging_dir("import")
    logs = ["[import] Reading CSV."]
    try:
        rows, stats, warnings = _read_csv_records(source)
        if not rows:
            raise ValueError("CSV contains no usable tags.")
        staged_csv = staging / "danbooru_tags.csv"
        staged_db = staging / "danbooru_tags.sqlite3"
        csv_count = _write_csv(staged_csv, rows)
        logs.append(f"[import] Accepted {csv_count:,} tags.")
        logs.append(f"[import] Skipped {stats['malformed']:,} malformed rows.")
        logs.append(f"[import] Merged {stats['duplicates']:,} duplicate tags.")
        db_count = build_sqlite(rows, staged_db)
        if csv_count != db_count or db_count <= 0:
            raise ValueError("Imported catalog validation failed.")
        logs.append("[import] Rebuilt SQLite index.")
        _backup_existing()
        os.replace(staged_csv, CSV_PATH)
        os.replace(staged_db, DB_PATH)
        state = _load_state()
        state.update({"state": "completed", "finished_at": _utc_now(), "tag_count": db_count, "last_error": ""})
        for line in logs:
            _append_log(state, line)
        _append_log(state, "[import] Catalog import complete.")
        _save_state(state)
        return {"ok": True, "tag_count": db_count, "warnings": warnings, "info": logs}
    except Exception as exc:
        state = _load_state()
        state.update({"state": "failed", "finished_at": _utc_now(), "last_error": str(exc)})
        for line in logs:
            _append_log(state, line)
        _append_log(state, f"[import] Failed: {exc}")
        _save_state(state)
        return {"ok": False, "error": str(exc), "warnings": [], "info": logs}


def export_csv() -> Path:
    if not CSV_PATH.exists():
        raise FileNotFoundError("No Danbooru CSV catalog is installed.")
    return CSV_PATH


def rebuild_sqlite_from_csv() -> Dict[str, Any]:
    _ensure_root()
    if not CSV_PATH.exists():
        return {"ok": False, "error": "CSV catalog not found.", "warnings": [], "info": []}
    staging = _staging_dir("rebuild")
    try:
        rows, stats, warnings = _read_csv_records(CSV_PATH)
        if not rows:
            raise ValueError("CSV contains no usable tags.")
        staged_csv = staging / "danbooru_tags.csv"
        staged_db = staging / "danbooru_tags.sqlite3"
        csv_count = _write_csv(staged_csv, rows)
        count = build_sqlite(rows, staged_db)
        if count <= 0 or csv_count != count:
            raise ValueError("SQLite rebuild produced no rows.")
        if CSV_PATH.exists():
            shutil.copy2(CSV_PATH, CSV_PATH.with_suffix(CSV_PATH.suffix + f".{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"))
        if DB_PATH.exists():
            shutil.copy2(DB_PATH, DB_PATH.with_suffix(DB_PATH.suffix + f".{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"))
        os.replace(staged_csv, CSV_PATH)
        os.replace(staged_db, DB_PATH)
        state = _load_state()
        state.update({"state": "completed", "finished_at": _utc_now(), "tag_count": count, "last_error": ""})
        _append_log(state, f"[rebuild] Rewrote CSV and SQLite index with {count:,} nonzero-post tags.")
        _save_state(state)
        return {"ok": True, "tag_count": count, "warnings": warnings, "info": [f"Rewrote CSV and SQLite index with {count:,} nonzero-post tags."], "stats": stats}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "warnings": [], "info": []}


def _normalize_settings(payload: Any) -> Dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    defaults = json.loads(json.dumps(DEFAULT_SETTINGS))
    cats = src.get("include_categories") if isinstance(src.get("include_categories"), dict) else {}
    sections = src.get("sections") if isinstance(src.get("sections"), dict) else {}
    aliases_src = src.get("aliases") if isinstance(src.get("aliases"), dict) else {}
    try:
        min_count = int(src.get("min_post_count", defaults["min_post_count"]))
    except Exception:
        min_count = defaults["min_post_count"]
    try:
        max_suggestions = int(src.get("max_suggestions", defaults["max_suggestions"]))
    except Exception:
        max_suggestions = defaults["max_suggestions"]
    out = {
        "enabled": bool(src.get("enabled", defaults["enabled"])),
        "min_post_count": max(0, min_count),
        "max_suggestions": max(3, min(30, max_suggestions)),
        "include_categories": {
            key: bool(cats.get(key, defaults["include_categories"][key]))
            for key in defaults["include_categories"]
        },
        "include_deprecated": bool(src.get("include_deprecated", defaults["include_deprecated"])),
        "aliases": {},
        "sections": {},
    }
    for alias, canonical in aliases_src.items():
        clean_alias = normalize_tag_name(alias)
        clean_canonical = normalize_tag_name(canonical)
        if clean_alias and clean_canonical and clean_alias != clean_canonical:
            out["aliases"][clean_alias] = clean_canonical
    for key, meta in DEFAULT_SECTIONS.items():
        raw = sections.get(key) if isinstance(sections.get(key), dict) else {}
        out["sections"][key] = {"label": raw.get("label") or meta["label"], "enabled": bool(raw.get("enabled", meta["enabled"]))}
    for key, raw in sections.items():
        clean_key = normalize_tag_name(key)
        if not clean_key or clean_key in out["sections"]:
            continue
        label = str(raw.get("label") or clean_key.replace("_", " ").title()) if isinstance(raw, dict) else clean_key.replace("_", " ").title()
        enabled = bool(raw.get("enabled", True)) if isinstance(raw, dict) else True
        out["sections"][clean_key] = {"label": label, "enabled": enabled}
    return out


def load_settings() -> Dict[str, Any]:
    return _normalize_settings(_read_json(SETTINGS_PATH, DEFAULT_SETTINGS))


def save_settings(payload: Any) -> Dict[str, Any]:
    settings = _normalize_settings(payload)
    _atomic_write_json(SETTINGS_PATH, settings)
    return settings


def settings_with_sections(section_defs: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    settings = load_settings()
    changed = False
    for item in section_defs or []:
        key = normalize_tag_name(item.get("id"))
        if not key:
            continue
        if key not in settings["sections"]:
            settings["sections"][key] = {"label": item.get("label") or key.replace("_", " ").title(), "enabled": True}
            changed = True
    if changed:
        save_settings(settings)
    return settings


def _rank_boost(name: str, project_tags: List[str], glossary_tags: List[str]) -> Tuple[int, int]:
    project = [normalize_tag_name(tag) for tag in project_tags or []]
    glossary = {normalize_tag_name(tag) for tag in glossary_tags or []}
    project_score = sum(1 for tag in project if tag == name)
    glossary_score = 1 if name in glossary else 0
    return project_score, glossary_score


def lookup_tag(name: str) -> Optional[Dict[str, Any]]:
    clean = normalize_tag_name(name)
    if not clean or not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT name, category_name, post_count, is_deprecated FROM tag WHERE name = ? AND post_count >= ? LIMIT 1",
            (clean, MIN_CATALOG_POST_COUNT),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "tag": row[0],
        "canonical": row[0],
        "category": row[1],
        "post_count": int(row[2] or 0),
        "is_deprecated": bool(row[3]),
        "validation_status": "Deprecated or invalid" if row[3] else "Verified Danbooru tag",
        "source": "danbooru",
    }


def resolve_alias(name: str, aliases: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    clean = normalize_tag_name(name)
    alias_map = aliases if aliases is not None else load_settings().get("aliases", {})
    canonical = normalize_tag_name((alias_map or {}).get(clean))
    if not clean or not canonical:
        return None
    record = lookup_tag(canonical) or {
        "tag": canonical,
        "canonical": canonical,
        "category": "",
        "post_count": 0,
        "is_deprecated": False,
        "source": "alias",
    }
    record.update(
        {
            "tag": canonical,
            "canonical": canonical,
            "alias": clean,
            "validation_status": "Alias that resolves to a canonical tag",
            "source": "alias",
        }
    )
    return record


def validate_tags(tags: Iterable[str], custom_tags: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    custom = {normalize_tag_name(tag) for tag in custom_tags or [] if normalize_tag_name(tag)}
    aliases = load_settings().get("aliases", {})
    out: List[Dict[str, Any]] = []
    for raw in tags or []:
        clean = normalize_tag_name(raw)
        if not clean:
            continue
        alias = resolve_alias(clean, aliases)
        if alias:
            out.append(alias)
            continue
        record = lookup_tag(clean)
        if record:
            out.append(record)
            continue
        out.append(
            {
                "tag": clean,
                "canonical": clean,
                "category": "",
                "post_count": 0,
                "is_deprecated": False,
                "validation_status": "Custom / unknown tag",
                "source": "custom" if clean in custom else "unknown",
                "whitelisted": clean in custom,
            }
        )
    return out


def search_suggestions(
    query: str,
    limit: int,
    min_post_count: int,
    include_categories: List[str],
    include_deprecated: bool,
    existing_tags: List[str],
    project_tags: List[str],
    glossary_tags: List[str],
) -> List[Dict[str, Any]]:
    clean_query = normalize_tag_name(query)
    settings = load_settings()
    aliases = settings.get("aliases", {})
    alias_hit = resolve_alias(clean_query, aliases) if clean_query else None
    if len(clean_query) < 2 or not DB_PATH.exists():
        return [alias_hit] if alias_hit else []
    exact_hit = lookup_tag(clean_query)
    if exact_hit and (include_deprecated or not exact_hit.get("is_deprecated")):
        exact_hit["source"] = "danbooru"
    else:
        exact_hit = None
    if alias_hit and not include_deprecated and alias_hit.get("is_deprecated"):
        alias_hit = None
    limit = max(1, min(30, int(limit or 12)))
    min_post_count = max(MIN_CATALOG_POST_COUNT, int(min_post_count or 0))
    category_ids = [CATEGORY_IDS[name] for name in include_categories or [] if name in CATEGORY_IDS]
    if not category_ids:
        return []
    existing = {normalize_tag_name(tag) for tag in existing_tags or []}
    params: List[Any] = [clean_query, f"{clean_query}%", f"%{clean_query}%", min_post_count]
    category_sql = ",".join("?" for _ in category_ids)
    params.extend(category_ids)
    deprecated_clause = "" if include_deprecated else "AND is_deprecated = 0"
    sql = f"""
        SELECT name, category_name, post_count, is_deprecated
        FROM tag
        WHERE name <> ?
          AND (name LIKE ? OR name LIKE ?)
          AND post_count >= ?
          AND category IN ({category_sql})
          {deprecated_clause}
        LIMIT 250
    """
    rows: List[Dict[str, Any]] = []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        for name, cat_name, post_count, is_deprecated in conn.execute(sql, params):
            if name in existing:
                continue
            project_score, glossary_score = _rank_boost(name, project_tags, glossary_tags)
            status = "Deprecated or invalid" if bool(is_deprecated) else "Verified Danbooru tag"
            rows.append(
                {
                    "tag": name,
                    "canonical": name,
                    "category": cat_name,
                    "post_count": int(post_count or 0),
                    "is_deprecated": bool(is_deprecated),
                    "validation_status": status,
                    "source": "project" if project_score else ("glossary" if glossary_score else "danbooru"),
                    "_rank": (
                        2 if name == clean_query else 0,
                        1 if name.startswith(clean_query) else 0,
                        project_score,
                        glossary_score,
                        int(post_count or 0),
                        -len(name),
                    ),
                }
            )
    finally:
        conn.close()
    rows.sort(key=lambda item: item["_rank"], reverse=True)
    prefix: List[Dict[str, Any]] = []
    for special in (exact_hit, alias_hit):
        if special and special.get("tag") not in {item.get("tag") for item in prefix}:
            special["_rank"] = (3 if not special.get("alias") else 2, 1, 0, 0, int(special.get("post_count") or 0), 0)
            prefix.append(special)
    if prefix:
        rows = prefix + [row for row in rows if row.get("tag") not in {item.get("tag") for item in prefix}]
    for item in rows:
        item.pop("_rank", None)
    return rows[:limit]
