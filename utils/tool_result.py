from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.io import log_join


ToolMeta = Dict[str, Any]
ToolTuple = Tuple[str, str]
ToolTupleWithMeta = Tuple[str, str, ToolMeta]


def build_tool_result(
    active_tab: str,
    lines: Iterable[str],
    *,
    ok: bool,
    error: str = "",
    artifacts: Optional[Dict[str, Any]] = None,
) -> ToolTupleWithMeta:
    meta: ToolMeta = {
        "ok": bool(ok),
        "error": str(error or ""),
        "artifacts": dict(artifacts or {}),
    }
    return active_tab, log_join([str(line) for line in lines]), meta


def unpack_tool_result(raw: Any) -> Tuple[str, str, ToolMeta]:
    if isinstance(raw, tuple):
        if len(raw) >= 3 and isinstance(raw[2], dict):
            tab = str(raw[0] or "")
            log = str(raw[1] or "")
            meta = dict(raw[2] or {})
            meta.setdefault("ok", True)
            meta.setdefault("error", "")
            if not isinstance(meta.get("artifacts"), dict):
                meta["artifacts"] = {}
            return tab, log, meta
        if len(raw) >= 2:
            tab = str(raw[0] or "")
            log = str(raw[1] or "")
            return tab, log, {"ok": True, "error": "", "artifacts": {}}
    if isinstance(raw, dict):
        tab = str(raw.get("active_tab") or "")
        log = str(raw.get("log") or "")
        meta = {
            "ok": bool(raw.get("ok", True)),
            "error": str(raw.get("error") or ""),
            "artifacts": dict(raw.get("artifacts") or {}),
        }
        return tab, log, meta
    return "", "", {"ok": False, "error": "Invalid tool response", "artifacts": {}}
