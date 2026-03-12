from __future__ import annotations

from typing import Any, List, Optional


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    text = str(val).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def parse_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        text = str(val).strip()
        if not text:
            return default
        return int(text)
    except Exception:
        return default


def parse_optional_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def parse_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        text = str(val).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def parse_exts(raw: Any, default: Optional[List[str]] = None) -> List[str]:
    if isinstance(raw, list):
        items = [str(x).strip().lower() for x in raw if str(x).strip()]
    elif raw:
        items = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    else:
        items = []

    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        if item in seen:
            continue
        seen.add(item)
        out.append(item)

    if out:
        return out
    return list(default or [])


def _chunks(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def parse_tag_list(raw: Any, dedupe: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for chunk in _chunks(raw):
        for line in chunk.replace("\r", "\n").split("\n"):
            for token in line.split(","):
                value = token.strip()
                if not value:
                    continue
                if dedupe:
                    if value in seen:
                        continue
                    seen.add(value)
                out.append(value)
    return out


def parse_line_list(raw: Any, dedupe: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for chunk in _chunks(raw):
        for line in chunk.replace("\r", "\n").split("\n"):
            value = line.strip()
            if not value:
                continue
            if dedupe:
                if value in seen:
                    continue
                seen.add(value)
            out.append(value)
    return out
