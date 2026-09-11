import json
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


BLACKLIST_PATH = Path(__file__).resolve().parent.parent / "trigger_safety_blacklist.json"
BLACKLIST_VERSION = 1
ONLINE_CACHE_TTL = 24 * 60 * 60
_ONLINE_CACHE = {}
_CACHE_LOCK = threading.Lock()

SEVERITY_SCORE = {
    "safe": 0,
    "warning": 65,
    "block": 100,
}
SEVERITY_RANK = {
    "safe": 0,
    "warning": 1,
    "block": 2,
}

MINOR_BLOCK_TERMS = [
    "loli",
    "lolicon",
    "shota",
    "shotacon",
    "underage",
    "minor",
    "child",
    "toddler",
    "infant",
]
AGE_WARNING_TERMS = [
    "young",
    "baby",
    "kid",
    "teen",
    "schoolgirl",
    "schoolboy",
]
ONLINE_POI_KEYWORDS = [
    "rapper",
    "singer",
    "actor",
    "actress",
    "musician",
    "politician",
    "athlete",
    "model",
    "adult film actor",
    "public figure",
    "human",
]


def normalize_text(text: str) -> dict:
    raw = str(text or "").strip()
    lower = raw.lower()
    split_digits = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", lower)
    tokenized = [part for part in re.split(r"[\W_]+", split_digits) if part]
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    return {
        "raw": raw,
        "lower": lower,
        "tokenized": tokenized,
        "compact": compact,
    }


def load_blacklist() -> dict:
    if not BLACKLIST_PATH.exists():
        payload = _default_blacklist_payload()
        _write_blacklist(payload, backup=False)
        payload["_info"] = ["blacklist file created"]
        return payload

    try:
        raw = BLACKLIST_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        timestamp = int(time.time())
        broken_path = BLACKLIST_PATH.with_name(f"{BLACKLIST_PATH.stem}.broken.{timestamp}.json")
        warnings = []
        try:
            BLACKLIST_PATH.rename(broken_path)
            warnings.append(f"corrupt blacklist renamed to {broken_path.name}")
        except Exception as exc:
            warnings.append(f"corrupt blacklist could not be renamed: {exc}")
        payload = _default_blacklist_payload()
        _write_blacklist(payload, backup=False)
        payload["_warnings"] = warnings
        payload["_info"] = ["blacklist file recreated"]
        return payload

    if not isinstance(payload, dict):
        payload = _default_blacklist_payload()
        payload["_warnings"] = ["blacklist payload was invalid; defaults loaded"]

    normalized = _normalize_blacklist_payload(payload)
    return normalized


def save_blacklist(payload: dict) -> None:
    normalized = _normalize_blacklist_payload(payload if isinstance(payload, dict) else {})
    normalized["updated_at"] = int(time.time())
    normalized.pop("_warnings", None)
    normalized.pop("_info", None)
    _write_blacklist(normalized, backup=True)


def scan_trigger_payload(payload: dict, online_poi: bool = True) -> dict:
    src = payload if isinstance(payload, dict) else {}
    fields = {
        "trigger": str(src.get("trigger") or ""),
        "model_name": str(src.get("model_name") or ""),
        "version_name": str(src.get("version_name") or ""),
        "sample_prompt": str(src.get("sample_prompt") or ""),
    }
    normalized_fields = {field: normalize_text(value) for field, value in fields.items()}
    warnings = []
    hits = []

    blacklist = load_blacklist()
    warnings.extend(blacklist.get("_warnings") or [])
    hits.extend(_scan_blacklist(normalized_fields, blacklist.get("entries") or []))
    hits.extend(_scan_builtin_age_terms(normalized_fields))

    if online_poi:
        online_hits, online_warnings = _scan_online_poi(fields)
        hits.extend(online_hits)
        warnings.extend(online_warnings)

    risk, score = _risk_from_hits(hits)
    suggestions = suggest_safe_triggers(fields.get("trigger") or "") if risk in {"warning", "block"} else []
    return {
        "ok": True,
        "risk": risk,
        "score": score,
        "fields_scanned": fields,
        "hits": hits,
        "warnings": warnings,
        "info": blacklist.get("_info") or [],
        "suggestions": suggestions,
    }


def add_blacklist_entry(
    term: str,
    reason: str,
    category: str = "poi",
    severity: str = "block",
    match_mode: str = "compact_contains",
) -> dict:
    clean_term = str(term or "").strip()
    if not clean_term:
        return {"ok": False, "error": "term is required", "warnings": [], "info": []}

    payload = load_blacklist()
    warnings = list(payload.get("_warnings") or [])
    info = list(payload.get("_info") or [])
    entries = [_normalize_entry(entry) for entry in payload.get("entries") or [] if isinstance(entry, dict)]

    entry = _normalize_entry({
        "term": clean_term,
        "aliases": [],
        "severity": severity,
        "category": category,
        "match_mode": match_mode,
        "reason": str(reason or "").strip() or "Manual blacklist entry",
        "source": "manual",
    })
    key = _entry_key(entry)
    duplicate = any(_entry_key(existing) == key for existing in entries)
    if duplicate:
        info.append("duplicate skipped")
        return {
            "ok": True,
            "entry": entry,
            "entry_count": len(entries),
            "duplicate": True,
            "warnings": warnings,
            "info": info,
        }

    entries.append(entry)
    payload["entries"] = _dedupe_entries(entries)
    save_blacklist(payload)
    info.append("blacklist file updated")
    return {
        "ok": True,
        "entry": entry,
        "entry_count": len(payload["entries"]),
        "duplicate": False,
        "warnings": warnings,
        "info": info,
    }


def suggest_safe_triggers(base: str = "") -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    suggestions = []
    prefixes = ["trg", "trg", "bbx"]
    while len(suggestions) < 3:
        code = "".join(secrets.choice(alphabet) for _ in range(5))
        value = f"{prefixes[len(suggestions)]}_{code}"
        if value not in suggestions:
            suggestions.append(value)
    return suggestions


def _default_blacklist_payload() -> dict:
    entries = [
        {
            "term": "nelly",
            "compact": "nelly",
            "aliases": [],
            "severity": "block",
            "category": "poi",
            "match_mode": "compact_contains",
            "reason": "Known POI collision",
            "source": "seed",
        },
        {
            "term": "vanilla ice",
            "compact": "vanillaice",
            "aliases": [],
            "severity": "block",
            "category": "poi",
            "match_mode": "compact_contains",
            "reason": "Known POI collision",
            "source": "seed",
        },
    ]
    return {
        "version": BLACKLIST_VERSION,
        "updated_at": int(time.time()),
        "entries": entries,
    }


def _write_blacklist(payload: dict, backup: bool) -> None:
    BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if backup and BLACKLIST_PATH.exists():
        try:
            backup_path = Path(str(BLACKLIST_PATH) + ".bak")
            backup_path.write_text(BLACKLIST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    BLACKLIST_PATH.write_text(text, encoding="utf-8")


def _normalize_blacklist_payload(payload: dict) -> dict:
    warnings = list(payload.get("_warnings") or [])
    info = list(payload.get("_info") or [])
    entries = [_normalize_entry(entry) for entry in payload.get("entries") or [] if isinstance(entry, dict)]
    normalized = {
        "version": int(payload.get("version") or BLACKLIST_VERSION),
        "updated_at": int(payload.get("updated_at") or 0),
        "entries": _dedupe_entries(entries),
    }
    if warnings:
        normalized["_warnings"] = warnings
    if info:
        normalized["_info"] = info
    return normalized


def _normalize_entry(entry: dict) -> dict:
    term = str(entry.get("term") or "").strip()
    norm = normalize_text(term)
    aliases = []
    for alias in entry.get("aliases") or []:
        clean = str(alias or "").strip()
        if clean:
            aliases.append(clean)
    severity = str(entry.get("severity") or "block").strip().lower()
    if severity not in {"warning", "block"}:
        severity = "block"
    match_mode = str(entry.get("match_mode") or "compact_contains").strip().lower()
    if match_mode not in {"exact", "token_exact", "compact_exact", "compact_contains"}:
        match_mode = "compact_contains"
    category = str(entry.get("category") or "poi").strip().lower() or "poi"
    return {
        "term": term,
        "compact": str(entry.get("compact") or norm["compact"]).strip().lower(),
        "aliases": aliases,
        "severity": severity,
        "category": category,
        "match_mode": match_mode,
        "reason": str(entry.get("reason") or "Blacklist match").strip(),
        "source": str(entry.get("source") or "manual").strip(),
    }


def _entry_key(entry: dict) -> tuple:
    return (
        str(entry.get("compact") or ""),
        str(entry.get("category") or ""),
        str(entry.get("match_mode") or ""),
    )


def _dedupe_entries(entries: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for entry in entries:
        key = _entry_key(entry)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _scan_blacklist(normalized_fields: dict, entries: list[dict]) -> list[dict]:
    hits = []
    for field, norm in normalized_fields.items():
        if not norm["raw"]:
            continue
        for entry in entries:
            entry_hits = _match_entry(norm, entry)
            for matched_text in entry_hits:
                hits.append({
                    "source": "local_blacklist",
                    "field": field,
                    "term": entry.get("term") or "",
                    "matched_text": matched_text,
                    "match_mode": entry.get("match_mode") or "",
                    "severity": entry.get("severity") or "block",
                    "category": entry.get("category") or "poi",
                    "reason": entry.get("reason") or "Blacklist match",
                })
    return hits


def _match_entry(norm: dict, entry: dict) -> list[str]:
    terms = [entry.get("term") or ""]
    terms.extend(entry.get("aliases") or [])
    hits = []
    for term in terms:
        term_norm = normalize_text(term)
        term_lower = term_norm["lower"]
        term_compact = entry.get("compact") if term == entry.get("term") else term_norm["compact"]
        if not term_compact:
            continue
        mode = entry.get("match_mode") or "compact_contains"
        if mode == "exact" and norm["lower"] == term_lower:
            hits.append(norm["raw"])
        elif mode == "token_exact" and term_lower in norm["tokenized"]:
            hits.append(term)
        elif mode == "compact_exact" and norm["compact"] == term_compact:
            hits.append(norm["raw"])
        elif mode == "compact_contains" and term_compact in norm["compact"]:
            hits.append(norm["raw"])
    return hits


def _scan_builtin_age_terms(normalized_fields: dict) -> list[dict]:
    hits = []
    for field, norm in normalized_fields.items():
        if not norm["raw"]:
            continue
        for term in MINOR_BLOCK_TERMS:
            if _careful_builtin_match(norm, term, "block"):
                hits.append(_builtin_hit(field, norm["raw"], term, "block", "minor"))
        for term in AGE_WARNING_TERMS:
            if _careful_builtin_match(norm, term, "warning"):
                hits.append(_builtin_hit(field, norm["raw"], term, "warning", "age_term"))
    return hits


def _careful_builtin_match(norm: dict, term: str, severity: str) -> bool:
    term_norm = normalize_text(term)
    if term_norm["lower"] in norm["tokenized"]:
        return True
    if norm["compact"] == term_norm["compact"]:
        return True
    if severity == "warning" and term == "baby" and term_norm["compact"] in norm["compact"]:
        return True
    return False


def _builtin_hit(field: str, matched_text: str, term: str, severity: str, category: str) -> dict:
    return {
        "source": "builtin",
        "field": field,
        "term": term,
        "matched_text": matched_text,
        "match_mode": "token_exact",
        "severity": severity,
        "category": category,
        "reason": "Built-in trigger safety term",
    }


def _risk_from_hits(hits: list[dict]) -> tuple[str, int]:
    risk = "safe"
    score = 0
    for hit in hits:
        severity = str(hit.get("severity") or "safe").lower()
        if severity not in SEVERITY_RANK:
            severity = "safe"
        hit_score = SEVERITY_SCORE.get(severity, 0)
        if SEVERITY_RANK[severity] > SEVERITY_RANK[risk]:
            risk = severity
            score = hit_score
        elif SEVERITY_RANK[severity] == SEVERITY_RANK[risk]:
            score = max(score, hit_score)
    return risk, score


def _scan_online_poi(fields: dict) -> tuple[list[dict], list[str]]:
    hits = []
    warnings = []
    candidates = _online_candidates(fields)
    for field, candidate in candidates:
        if not candidate:
            continue
        try:
            hit = _lookup_online_poi(candidate, field)
        except Exception:
            warnings.append("online_poi_unavailable")
            break
        if hit:
            hits.append(hit)
    return hits, warnings


def _online_candidates(fields: dict) -> list[tuple[str, str]]:
    out = []
    seen = set()

    def add(field, value):
        clean = str(value or "").strip()
        if len(clean) < 3:
            return
        key = (field, normalize_text(clean)["compact"])
        if not key[1] or key in seen:
            return
        seen.add(key)
        out.append((field, clean))

    trigger_norm = normalize_text(fields.get("trigger") or "")
    add("trigger", trigger_norm["raw"])
    add("trigger", trigger_norm["compact"])
    add("trigger", " ".join(trigger_norm["tokenized"]))
    add("model_name", fields.get("model_name") or "")
    version = str(fields.get("version_name") or "").strip()
    if version and not re.fullmatch(r"v?\d+(\.\d+)*", version.lower()):
        add("version_name", version)
    return out[:8]


def _lookup_online_poi(candidate: str, field: str) -> dict | None:
    norm = normalize_text(candidate)
    compact = norm["compact"]
    now = time.time()
    with _CACHE_LOCK:
        cached = _ONLINE_CACHE.get(compact)
        if cached and now - cached.get("ts", 0) < ONLINE_CACHE_TTL:
            return cached.get("hit")

    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "limit": "5",
        "search": candidate,
    })
    url = f"https://www.wikidata.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "BatchBench Trigger Safety/1.0"})
    with urllib.request.urlopen(req, timeout=3) as response:
        data = json.loads(response.read().decode("utf-8"))

    hit = _wikidata_hit_from_search(data, candidate, field)
    with _CACHE_LOCK:
        _ONLINE_CACHE[compact] = {"ts": now, "hit": hit}
    return hit


def _wikidata_hit_from_search(data: dict, candidate: str, field: str) -> dict | None:
    candidate_norm = normalize_text(candidate)
    candidate_compact = candidate_norm["compact"]
    for item in data.get("search") or []:
        label = str(item.get("label") or "")
        description = str(item.get("description") or "")
        aliases = [str(alias or "") for alias in item.get("aliases") or []]
        desc_lower = description.lower()
        if not any(keyword in desc_lower for keyword in ONLINE_POI_KEYWORDS):
            continue
        names = [label] + aliases
        for name in names:
            name_compact = normalize_text(name)["compact"]
            if name_compact and name_compact == candidate_compact:
                return {
                    "source": "online_poi",
                    "field": field,
                    "term": label or name,
                    "matched_text": candidate,
                    "match_mode": "compact_exact",
                    "severity": "warning",
                    "category": "poi",
                    "reason": f"Exact public figure match: {label or name} ({description})",
                }
    return None
