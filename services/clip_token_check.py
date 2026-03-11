from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.io import readable_path
from utils.parse import parse_bool, parse_exts, parse_int
from utils.text_io import read_text_best_effort
from utils.tool_result import build_tool_result


def estimate_token_count(text: str) -> int:
    normalized = (text or "").replace("_", " ")
    chunks = re.findall(r"\w+", normalized, flags=re.UNICODE)
    return len(chunks) + 2


def try_get_clip_tokenizer() -> Optional[object]:
    try:
        from transformers import CLIPTokenizerFast  # type: ignore
    except Exception:
        return None

    model_or_path = (os.getenv("CLIP_TOKENIZER_PATH", "") or "").strip()
    try:
        if model_or_path:
            return CLIPTokenizerFast.from_pretrained(model_or_path, local_files_only=True)
        return CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        return None


def exact_token_count(tokenizer: object, text: str) -> int:
    encoded = tokenizer(text or "", truncation=False, add_special_tokens=True)  # type: ignore[misc]
    return len(encoded["input_ids"])


def _normalize_exts(exts: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for raw in exts or []:
        val = (raw or "").strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.add(val)
    return out


def _list_images(folder: Path, exts: Sequence[str], recursive: bool, include_temp: bool) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    extset = _normalize_exts(exts)
    if not extset:
        return []

    exclude_dir: Optional[Path] = None
    if recursive and not include_temp and folder.name.lower() != "_temp":
        temp = folder / "_temp"
        if temp.exists():
            exclude_dir = temp

    images: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if exclude_dir and exclude_dir in p.parents:
                continue
            if p.suffix.lower() in extset:
                images.append(p)
    else:
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extset]
    return sorted(images, key=lambda p: str(p).lower())


def _preview_text(text: str, max_len: int = 90) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def scan_caption_files(
    folder: Path,
    exts: List[str],
    recursive: bool,
    limit: int,
    mode: str,
    include_temp: bool,
) -> Dict[str, Any]:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}

    mode_normalized = (mode or "estimate").strip().lower()
    requested_exact = mode_normalized == "exact"
    tokenizer = try_get_clip_tokenizer() if requested_exact else None
    use_exact = requested_exact and tokenizer is not None

    images = _list_images(folder, exts, recursive=recursive, include_temp=include_temp)
    scanned_captions = 0
    over_limit = 0
    max_tokens = 0
    offenders: List[Dict[str, Any]] = []

    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        try:
            text, _, _ = read_text_best_effort(txt)
        except Exception:
            continue

        scanned_captions += 1
        if use_exact:
            try:
                tokens = exact_token_count(tokenizer, text)
            except Exception:
                tokens = estimate_token_count(text)
        else:
            tokens = estimate_token_count(text)

        if tokens > limit:
            over_limit += 1
        if tokens > max_tokens:
            max_tokens = tokens

        try:
            rel_name = img.relative_to(folder).as_posix()
        except Exception:
            rel_name = img.name
        offenders.append(
            {
                "file": rel_name,
                "token_count": int(tokens),
                "preview": _preview_text(text),
            }
        )

    offenders.sort(key=lambda item: (-int(item["token_count"]), str(item["file"]).lower()))
    return {
        "ok": True,
        "folder": str(folder),
        "mode_requested": mode_normalized if mode_normalized in {"estimate", "exact"} else "estimate",
        "mode_used": "exact" if use_exact else "estimate",
        "tokenizer_loaded": bool(tokenizer is not None),
        "total_images": len(images),
        "scanned_captions": scanned_captions,
        "limit": int(limit),
        "over_limit": over_limit,
        "max_tokens": max_tokens,
        "offenders": offenders,
    }


def handle(form, ctx):
    active_tab = "clip_tokens"
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    folder_raw = (form.get("folder_clip_tokens", "") or "").strip()
    folder = readable_path(folder_raw)
    exts = parse_exts(form.get("exts_clip_tokens"), default=[".jpg", ".jpeg", ".png", ".webp"])
    recursive = parse_bool(form.get("recursive_clip_tokens"), default=True)
    include_temp = parse_bool(form.get("include_temp_clip_tokens"), default=False)
    limit = max(0, parse_int(form.get("limit_clip_tokens"), default=77) or 77)
    mode = (form.get("mode_clip_tokens", "estimate") or "estimate").strip().lower()
    topn = max(1, parse_int(form.get("topn_clip_tokens"), default=20) or 20)

    if not folder_raw:
        lines.append("Folder is required.")
        return _done(False, "Folder is required.")

    result = scan_caption_files(
        folder=folder,
        exts=exts,
        recursive=recursive,
        limit=limit,
        mode=mode,
        include_temp=include_temp,
    )
    if not result.get("ok"):
        err = str(result.get("error") or "Scan failed")
        lines.append(err)
        return _done(False, err)

    mode_used = str(result.get("mode_used") or "estimate")
    mode_requested = str(result.get("mode_requested") or "estimate")
    if mode_requested == "exact" and mode_used != "exact":
        lines.append("Mode: estimate (fallback, exact tokenizer unavailable)")
    else:
        lines.append(f"Mode: {mode_used}")
    lines.append(f"Folder: {folder}")
    lines.append(f"Scanned: {int(result.get('scanned_captions') or 0)} captions")
    lines.append(f"Over limit (>{limit}): {int(result.get('over_limit') or 0)}")
    lines.append(f"Max tokens: {int(result.get('max_tokens') or 0)}")
    lines.append("Top offenders:")

    offenders = list(result.get("offenders") or [])[:topn]
    if not offenders:
        lines.append("- (none)")
    else:
        for item in offenders:
            token_count = int(item.get("token_count") or 0)
            file_name = str(item.get("file") or "")
            preview = str(item.get("preview") or "")
            lines.append(f"- [{token_count}] {file_name}  :: {preview}")

    return _done(True)
