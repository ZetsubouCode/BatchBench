import json
import os
import re
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_DEFAULT_BASE_URL = "https://danbooru.donmai.us"
_DEFAULT_TIMEOUT_SECONDS = 3.0
_DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24
_DEFAULT_CACHE_MAX_ENTRIES = 300
_DEFAULT_PREVIEW_LIMIT = 8
_MAX_TAG_LENGTH = 120

CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "deprecated",
    3: "copyright",
    4: "character",
    5: "meta",
}
_CATALOG_PAGE_LIMIT_MAX = 1000

_CACHE: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CATALOG_RETRY_LOGS: List[str] = []
_CATALOG_RETRY_LOGS_LOCK = threading.Lock()


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, val))


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, val))


def _base_url() -> str:
    raw = (os.getenv("DANBOORU_BASE_URL") or _DEFAULT_BASE_URL).strip()
    raw = raw.rstrip("/")
    return raw or _DEFAULT_BASE_URL


def _user_agent() -> str:
    raw = (os.getenv("DANBOORU_USER_AGENT") or "").strip()
    return raw or "BatchBench/1.0 (Danbooru proxy)"


def _request_timeout_seconds() -> float:
    return _env_float(
        "DANBOORU_TIMEOUT_SECONDS",
        _DEFAULT_TIMEOUT_SECONDS,
        min_value=0.2,
        max_value=15.0,
    )


def _cache_ttl_seconds() -> int:
    return _env_int(
        "DANBOORU_CACHE_TTL_SECONDS",
        _DEFAULT_CACHE_TTL_SECONDS,
        min_value=60,
        max_value=60 * 60 * 24 * 30,
    )


def _cache_max_entries() -> int:
    return _env_int(
        "DANBOORU_CACHE_MAX_ENTRIES",
        _DEFAULT_CACHE_MAX_ENTRIES,
        min_value=50,
        max_value=5000,
    )


def _preview_limit_default() -> int:
    return _env_int(
        "DANBOORU_PREVIEW_LIMIT",
        _DEFAULT_PREVIEW_LIMIT,
        min_value=1,
        max_value=20,
    )


def _normalize_preview_limit(value: Any) -> int:
    if value is None:
        return _preview_limit_default()
    try:
        val = int(value)
    except Exception:
        return _preview_limit_default()
    return max(1, min(20, val))


def _auth_params() -> Dict[str, str]:
    params: Dict[str, str] = {}
    login = (os.getenv("DANBOORU_LOGIN") or "").strip()
    api_key = (os.getenv("DANBOORU_API_KEY") or "").strip()
    if login:
        params["login"] = login
    if api_key:
        params["api_key"] = api_key
    return params


def normalize_tag(raw_tag: Any) -> str:
    val = str(raw_tag or "").strip().lower()
    if not val:
        return ""
    val = val.replace(" ", "_")
    while "__" in val:
        val = val.replace("__", "_")
    return val.strip("_")


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _cache_key(tag: str, include_related: bool, include_preview: bool, preview_limit: int) -> str:
    return f"{tag}|r{1 if include_related else 0}|p{1 if include_preview else 0}|l{preview_limit}"


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    ttl = _cache_ttl_seconds()
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if not item:
            return None
        ts, payload = item
        if now - ts > ttl:
            _CACHE.pop(key, None)
            return None
        _CACHE.move_to_end(key)
        return deepcopy(payload)


def _cache_put(key: str, payload: Dict[str, Any]) -> None:
    max_entries = _cache_max_entries()
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), deepcopy(payload))
        _CACHE.move_to_end(key)
        while len(_CACHE) > max_entries:
            _CACHE.popitem(last=False)


def _http_get_json(path: str, params: Dict[str, Any], timeout_seconds: float) -> Tuple[Optional[Any], Optional[str]]:
    query_params = dict(params or {})
    query_params.update(_auth_params())
    query = urlencode({k: v for k, v in query_params.items() if v not in (None, "")}, doseq=True)
    url = f"{_base_url()}{path}"
    if query:
        url = f"{url}?{query}"

    req = Request(
        url,
        headers={
            "User-Agent": _user_agent(),
            "Accept": "application/json",
        },
    )

    payload: bytes
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            payload = resp.read()
    except HTTPError as exc:
        message = f"HTTP {exc.code} from Danbooru"
        return None, message
    except URLError as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        return None, f"Danbooru request failed: {reason}"
    except TimeoutError:
        return None, "Danbooru request timed out"
    except Exception as exc:
        return None, f"Danbooru request failed: {exc}"

    if not payload:
        return None, None
    try:
        return json.loads(payload.decode("utf-8", errors="replace")), None
    except Exception as exc:
        return None, f"Invalid JSON from Danbooru: {exc}"


def _fetch_tag_meta(tag: str, timeout_seconds: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload, err = _http_get_json(
        "/tags.json",
        {"search[name]": tag, "limit": 1},
        timeout_seconds,
    )
    if err:
        return None, err
    if not isinstance(payload, list) or not payload:
        return None, None
    item = payload[0]
    if not isinstance(item, dict):
        return None, None
    category = _to_int(item.get("category"))
    return (
        {
            "name": str(item.get("name") or tag),
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "post_count": _to_int(item.get("post_count")),
        },
        None,
    )


def _is_transient_catalog_error(error: Optional[str]) -> bool:
    if not error:
        return False
    text = str(error)
    if "timed out" in text.lower() or "timeout" in text.lower():
        return True
    if "HTTP 429" in text:
        return True
    for code in range(500, 600):
        if f"HTTP {code}" in text:
            return True
    return False


def _normalize_catalog_record(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    name = normalize_tag(item.get("name"))
    if not name:
        return None
    category = _to_int(item.get("category"))
    return {
        "id": _to_int(item.get("id")) or 0,
        "name": name,
        "category": category if category is not None else 0,
        "post_count": _to_int(item.get("post_count")) or 0,
        "updated_at": str(item.get("updated_at") or ""),
    }


def drain_catalog_retry_logs() -> List[str]:
    with _CATALOG_RETRY_LOGS_LOCK:
        logs = list(_CATALOG_RETRY_LOGS)
        _CATALOG_RETRY_LOGS.clear()
    return logs


def _catalog_retry_log(message: str) -> None:
    with _CATALOG_RETRY_LOGS_LOCK:
        _CATALOG_RETRY_LOGS.append(message)


def fetch_tag_page(after_id: Optional[int], limit: int = 1000) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Fetch one Danbooru tag catalog page using Danbooru's id cursor pagination.

    Danbooru returns tag lists newest/highest-id first for search[order]=id. To
    keep walking older rows, pass page=b<lowest_seen_id> on the next request.
    """
    try:
        page_limit = int(limit)
    except Exception:
        page_limit = _CATALOG_PAGE_LIMIT_MAX
    page_limit = max(1, min(_CATALOG_PAGE_LIMIT_MAX, page_limit))

    params: Dict[str, Any] = {
        "limit": page_limit,
        "search[order]": "id",
    }
    if after_id is not None:
        params["page"] = f"b{max(0, int(after_id))}"

    timeout_seconds = _request_timeout_seconds()
    attempts = 4
    last_error: Optional[str] = None
    for attempt in range(1, attempts + 1):
        payload, err = _http_get_json("/tags.json", params, timeout_seconds)
        if not err:
            if not isinstance(payload, list):
                return [], "Invalid tag page payload from Danbooru"
            rows = []
            for item in payload:
                record = _normalize_catalog_record(item)
                if record:
                    rows.append(record)
            return rows, None
        last_error = err
        if not _is_transient_catalog_error(err) or attempt >= attempts:
            return [], err
        wait_seconds = min(4.0, 0.5 * (2 ** (attempt - 1)))
        _catalog_retry_log(f"[sync] Retry {attempt}/{attempts} after {err}. Waiting {wait_seconds:.1f} seconds.")
        time.sleep(wait_seconds)
    return [], last_error or "Danbooru request failed"


def lookup_tag_summaries(raw_tags: Any) -> Dict[str, Any]:
    values = raw_tags if isinstance(raw_tags, list) else []
    tags: List[str] = []
    seen = set()
    for raw_tag in values:
        tag = normalize_tag(raw_tag)
        if not tag or len(tag) > _MAX_TAG_LENGTH or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= 20:
            break

    timeout_seconds = _request_timeout_seconds()
    summaries: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for tag in tags:
        cache_key = f"summary|{tag}"
        cached = _cache_get(cache_key)
        if cached is not None:
            summaries[tag] = cached
            continue
        info, err = _fetch_tag_meta(tag, timeout_seconds)
        if err:
            errors[tag] = err
            continue
        summary = info or {
            "name": tag,
            "category": None,
            "category_name": "unknown",
            "post_count": 0,
        }
        summary = {**summary, "found": bool(info)}
        summaries[tag] = summary
        _cache_put(cache_key, summary)

    return {"ok": True, "summaries": summaries, "errors": errors}


def _fetch_wiki(tag: str, timeout_seconds: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload, err = _http_get_json(
        "/wiki_pages.json",
        {"search[title]": tag, "limit": 1},
        timeout_seconds,
    )
    if err:
        return None, err
    if not isinstance(payload, list) or not payload:
        return None, None
    item = payload[0]
    if not isinstance(item, dict):
        return None, None
    body = item.get("body")
    return (
        {
            "title": str(item.get("title") or tag),
            "body": str(body or ""),
            "other_names": [str(name) for name in (item.get("other_names") or []) if str(name or "").strip()],
            "updated_at": str(item.get("updated_at") or ""),
        },
        None,
    )


def _extract_related_tag(item: Any) -> str:
    if isinstance(item, str):
        return normalize_tag(item)
    if isinstance(item, (list, tuple)) and item:
        first = item[0]
        if isinstance(first, str):
            return normalize_tag(first)
    if isinstance(item, dict):
        nested = item.get("tag")
        if isinstance(nested, dict):
            value = nested.get("name")
            if isinstance(value, str):
                return normalize_tag(value)
        for key in ("name", "tag", "title"):
            value = item.get(key)
            if isinstance(value, str):
                return normalize_tag(value)
    return ""


def _extract_related_tags(payload: Any, source_tag: str) -> List[str]:
    source = normalize_tag(source_tag)
    raw_items: List[Any] = []
    if isinstance(payload, dict):
        for key in ("tags", "related_tags", "wiki_page_tags"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items.extend(value)
    elif isinstance(payload, list):
        raw_items.extend(payload)

    seen = set()
    out: List[str] = []
    for raw in raw_items:
        tag = _extract_related_tag(raw)
        if not tag or tag == source or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= 30:
            break
    return out


def _fetch_related(tag: str, timeout_seconds: float) -> Tuple[List[str], Optional[str]]:
    payload, err = _http_get_json(
        "/related_tag.json",
        {"query": tag},
        timeout_seconds,
    )
    if err:
        return [], err
    return _extract_related_tags(payload, tag), None


def _tag_summary(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    tag = item.get("tag") if isinstance(item.get("tag"), dict) else item
    name = normalize_tag(tag.get("name"))
    if not name:
        return None
    category = _to_int(tag.get("category"))
    summary: Dict[str, Any] = {
        "name": name,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "post_count": _to_int(tag.get("post_count")),
    }
    for key in ("cosine_similarity", "jaccard_similarity", "overlap_coefficient", "frequency"):
        try:
            summary[key] = float(item.get(key))
        except Exception:
            continue
    return summary


def _extract_related_context(payload: Any, source_tag: str) -> Dict[str, List[Dict[str, Any]]]:
    source = normalize_tag(source_tag)
    related: List[Dict[str, Any]] = []
    wiki_tags: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_related = payload.get("related_tags")
        raw_wiki_tags = payload.get("wiki_page_tags")
        if isinstance(raw_related, list):
            for item in raw_related:
                summary = _tag_summary(item)
                if not summary or summary["name"] == source:
                    continue
                related.append(summary)
                if len(related) >= 40:
                    break
        if isinstance(raw_wiki_tags, list):
            seen = set()
            for item in raw_wiki_tags:
                summary = _tag_summary(item)
                if not summary or summary["name"] == source or summary["name"] in seen:
                    continue
                seen.add(summary["name"])
                wiki_tags.append(summary)
                if len(wiki_tags) >= 40:
                    break
    if not related:
        related = [{"name": name} for name in _extract_related_tags(payload, source)[:40]]
    return {"related": related, "wiki_tags": wiki_tags}


def _fetch_related_context(tag: str, timeout_seconds: float) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
    payload, err = _http_get_json(
        "/related_tag.json",
        {"query": tag},
        timeout_seconds,
    )
    if err:
        return {"related": [], "wiki_tags": []}, err
    return _extract_related_context(payload, tag), None


def _relationship_names(payload: Any, field: str) -> List[str]:
    names: List[str] = []
    seen = set()
    if not isinstance(payload, list):
        return names
    for item in payload:
        if not isinstance(item, dict) or str(item.get("status") or "").lower() != "active":
            continue
        name = normalize_tag(item.get(field))
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _fetch_relationships(tag: str, timeout_seconds: float) -> Tuple[Dict[str, List[str]], List[str]]:
    requests = (
        ("implies", "/tag_implications.json", {"search[antecedent_name]": tag, "limit": 40}, "consequent_name"),
        ("implied_by", "/tag_implications.json", {"search[consequent_name]": tag, "limit": 40}, "antecedent_name"),
        ("aliases_to", "/tag_aliases.json", {"search[antecedent_name]": tag, "limit": 40}, "consequent_name"),
        ("aliases_from", "/tag_aliases.json", {"search[consequent_name]": tag, "limit": 40}, "antecedent_name"),
    )
    relationships: Dict[str, List[str]] = {}
    errors: List[str] = []
    for key, path, params, field in requests:
        payload, err = _http_get_json(path, params, timeout_seconds)
        relationships[key] = _relationship_names(payload, field)
        if err:
            errors.append(err)
    return relationships, errors


_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_DText_POST_RE = re.compile(r"!post\s+#?\d+(?::\s*)?")


def _clean_dtext(raw: Any) -> str:
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _WIKI_LINK_RE.sub(lambda match: match.group(1).replace("_", " "), text)
    text = _DText_POST_RE.sub("", text)
    text = re.sub(r"(?m)^h[1-6]\.\s*", "", text)
    text = re.sub(r"(?m)^\*\s*", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _wiki_links(raw: Any) -> Tuple[List[str], List[str]]:
    tag_links: List[str] = []
    group_links: List[str] = []
    seen_tags = set()
    seen_groups = set()
    for match in _WIKI_LINK_RE.finditer(str(raw or "")):
        title = match.group(1).strip()
        if not title:
            continue
        if title.lower().startswith("tag group:"):
            group = title.split(":", 1)[1].strip()
            if group and group.lower() not in seen_groups:
                seen_groups.add(group.lower())
                group_links.append(group)
            continue
        tag = normalize_tag(title)
        if tag and tag not in seen_tags:
            seen_tags.add(tag)
            tag_links.append(tag)
    return tag_links, group_links


def _wiki_guidance(raw: Any) -> Dict[str, List[str]]:
    text = _clean_dtext(raw)
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0] if paragraphs else "")
    avoid_markers = (" instead", "do not ", "don't ", " not ", " should not ", " rather than ", " use ", " see ")
    avoid = [sentence.strip() for sentence in sentences if any(marker in sentence.lower() for marker in avoid_markers)]
    use_when = [sentence.strip() for sentence in sentences if sentence.strip() and sentence.strip() not in avoid]
    return {
        "use_when": use_when[:3],
        "avoid_when": avoid[:4],
    }


def _absolute_url(raw: Any) -> str:
    url = str(raw or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{_base_url()}{url}"
    return f"{_base_url()}/{url.lstrip('/')}"


def _extract_preview_url(post: Dict[str, Any]) -> str:
    for key in ("preview_file_url", "large_file_url", "file_url"):
        value = _absolute_url(post.get(key))
        if value:
            return value

    media_asset = post.get("media_asset")
    if isinstance(media_asset, dict):
        variants = media_asset.get("variants")
        if isinstance(variants, list):
            preferred = []
            fallback = []
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                url = _absolute_url(variant.get("url"))
                if not url:
                    continue
                vtype = str(variant.get("type") or "").lower()
                if "180x180" in vtype or "preview" in vtype:
                    preferred.append(url)
                else:
                    fallback.append(url)
            if preferred:
                return preferred[0]
            if fallback:
                return fallback[0]
    return ""


def _extract_display_url(post: Dict[str, Any]) -> str:
    for key in ("large_file_url", "file_url", "preview_file_url"):
        value = _absolute_url(post.get(key))
        if value:
            return value
    return _extract_preview_url(post)


def _fetch_previews(tag: str, timeout_seconds: float, preview_limit: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    payload, err = _http_get_json(
        "/posts.json",
        {"tags": tag, "limit": preview_limit},
        timeout_seconds,
    )
    if err:
        return [], err
    if not isinstance(payload, list) or not payload:
        return [], None

    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        post_id = _to_int(item.get("id"))
        if not post_id:
            continue
        preview_url = _extract_preview_url(item)
        display_url = _extract_display_url(item)
        if not preview_url and not display_url:
            continue
        out.append(
            {
                "id": post_id,
                "preview_url": preview_url or display_url,
                "display_url": display_url or preview_url,
                "post_url": f"{_base_url()}/posts/{post_id}",
                "rating": str(item.get("rating") or ""),
                "width": _to_int(item.get("image_width")),
                "height": _to_int(item.get("image_height")),
                "file_ext": str(item.get("file_ext") or ""),
            }
        )
        if len(out) >= preview_limit:
            break
    return out, None


def _error_result(tag: str, message: str, error_code: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "tag": tag,
        "found": False,
        "cached": False,
        "info": None,
        "wiki": None,
        "related": [],
        "previews": [],
        "error": message,
        "error_code": error_code,
    }


def lookup_tag_info(
    raw_tag: Any,
    include_related: bool = True,
    include_preview: bool = True,
    preview_limit: Optional[int] = None,
) -> Dict[str, Any]:
    tag = normalize_tag(raw_tag)
    if not tag:
        return _error_result("", "tag is required", "invalid_input")
    if len(tag) > _MAX_TAG_LENGTH:
        return _error_result(tag, f"tag must be <= {_MAX_TAG_LENGTH} chars", "invalid_input")

    include_related = bool(include_related)
    include_preview = bool(include_preview)
    normalized_preview_limit = _normalize_preview_limit(preview_limit)
    cache_key = _cache_key(tag, include_related, include_preview, normalized_preview_limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    timeout_seconds = _request_timeout_seconds()
    info, info_err = _fetch_tag_meta(tag, timeout_seconds)
    wiki, wiki_err = _fetch_wiki(tag, timeout_seconds)
    related: List[str] = []
    related_err: Optional[str] = None
    if include_related:
        related, related_err = _fetch_related(tag, timeout_seconds)
    previews: List[Dict[str, Any]] = []
    preview_err: Optional[str] = None
    if include_preview:
        previews, preview_err = _fetch_previews(tag, timeout_seconds, normalized_preview_limit)

    found = bool(info or wiki or previews)
    if not found and (info_err or wiki_err or preview_err):
        err = info_err or wiki_err or preview_err or "Failed to fetch from Danbooru"
        return _error_result(tag, err, "fetch_failed")

    result: Dict[str, Any] = {
        "ok": True,
        "tag": tag,
        "found": found,
        "cached": False,
        "info": info,
        "wiki": wiki,
        "related": related[:30],
        "previews": previews[:normalized_preview_limit],
        "error": "",
    }

    # Related lookup is optional; if it fails we still return primary info.
    if include_related and related_err and found:
        result["related"] = []

    # Preview lookup is optional; if it fails we still return primary info.
    if include_preview and preview_err and found:
        result["previews"] = []

    _cache_put(cache_key, result)
    return result


def lookup_tag_wiki(raw_tag: Any, preview_limit: Optional[int] = None) -> Dict[str, Any]:
    tag = normalize_tag(raw_tag)
    if not tag:
        return _error_result("", "tag is required", "invalid_input")
    if len(tag) > _MAX_TAG_LENGTH:
        return _error_result(tag, f"tag must be <= {_MAX_TAG_LENGTH} chars", "invalid_input")

    normalized_preview_limit = _normalize_preview_limit(preview_limit)
    cache_key = f"wiki|{tag}|l{normalized_preview_limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    base = lookup_tag_info(
        tag,
        include_related=False,
        include_preview=True,
        preview_limit=normalized_preview_limit,
    )
    if not base.get("ok"):
        return base

    timeout_seconds = _request_timeout_seconds()
    related_context, related_err = _fetch_related_context(tag, timeout_seconds)
    relationships, relationship_errors = _fetch_relationships(tag, timeout_seconds)
    wiki = base.get("wiki") or {}
    wiki_body = str(wiki.get("body") or "")
    wiki_links, group_links = _wiki_links(wiki_body)
    wiki_linked = list(related_context.get("wiki_tags") or [])
    known_linked = {str(item.get("name") or "") for item in wiki_linked}
    for linked_tag in wiki_links:
        if linked_tag == tag or linked_tag in known_linked:
            continue
        known_linked.add(linked_tag)
        wiki_linked.append({"name": linked_tag})

    warnings = []
    if related_err:
        warnings.append(related_err)
    warnings.extend(relationship_errors)
    result: Dict[str, Any] = {
        **base,
        "cached": False,
        "wiki_plain": _clean_dtext(wiki_body),
        "guidance": _wiki_guidance(wiki_body),
        "related_details": related_context.get("related") or [],
        "wiki_linked_tags": wiki_linked[:60],
        "tag_groups": group_links[:30],
        "relationships": relationships,
        "warnings": list(dict.fromkeys(warnings)),
    }
    _cache_put(cache_key, result)
    return result
