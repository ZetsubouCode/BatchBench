from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Any, Dict, Optional


ACTIVITY_PAYLOADS: Dict[str, Dict[str, str]] = {
    "home": {
        "details": "Organizing datasets",
        "state": "Ready to work",
    },
    "guide": {
        "details": "Reviewing workflow guide",
        "state": "BatchBench Home",
    },
    "webp": {
        "details": "Converting images to PNG",
        "state": "Image Tools",
    },
    "batch": {
        "details": "Tuning image adjustments",
        "state": "Image Tools",
    },
    "blur_brush": {
        "details": "Brushing soft focus",
        "state": "Image Tools",
    },
    "color_brush": {
        "details": "Painting color details",
        "state": "Image Tools",
    },
    "palette_helper": {
        "details": "Planning manga palettes",
        "state": "Image Tools",
    },
    "epub_extractor": {
        "details": "Extracting EPUB artwork",
        "state": "Dataset Assembly",
    },
    "webtoon": {
        "details": "Splitting webtoon panels",
        "state": "Dataset Assembly",
    },
    "merge": {
        "details": "Stitching image groups",
        "state": "Dataset Assembly",
    },
    "rename": {
        "details": "Renaming dataset files",
        "state": "Dataset Assembly",
    },
    "combine": {
        "details": "Combining datasets",
        "state": "Dataset Assembly",
    },
    "normalize": {
        "details": "Normalizing dataset",
        "state": "Dataset Preparation",
    },
    "tags": {
        "details": "Editing dataset tags",
        "state": "Dataset Preparation",
    },
    "offline": {
        "details": "Auto-tagging images",
        "state": "Dataset Preparation",
    },
    "clip_tokens": {
        "details": "Checking CLIP tokens",
        "state": "Dataset Preparation",
    },
    "pipeline": {
        "details": "Preparing pipeline",
        "state": "Dataset Preparation",
    },
    "tag_wiki": {
        "details": "Browsing tag glossary",
        "state": "Reference Library",
    },
    "settings": {
        "details": "Configuring BatchBench",
        "state": "Settings",
    },
}

SAFE_PRESENCE_PHASES: Dict[str, Dict[str, Dict[str, str]]] = {
    "tags": {
        "default": {"details": "Tagging dataset", "progress": "image"},
        "guided_appearance": {"details": "Reviewing appearance details", "progress": "image"},
        "guided_outfit": {"details": "Reviewing outfit details", "progress": "image"},
        "guided_camera": {"details": "Reviewing camera angle", "progress": "image"},
        "guided_composition": {"details": "Reviewing body composition", "progress": "image"},
        "guided_lighting": {"details": "Reviewing lighting details", "progress": "image"},
        "guided_background": {"details": "Reviewing background details", "progress": "image"},
        "guided_generic": {"details": "Reviewing image tags", "progress": "image"},
        "manual": {"details": "Editing image tags", "progress": "image"},
    },
    "offline": {
        "running": {"details": "Auto-tagging images", "progress": "processed"},
        "preview": {"details": "Previewing auto-tags", "progress": "processed"},
        "complete": {"details": "Auto-tagging complete", "progress": "images_processed", "force": "1"},
    },
    "pipeline": {
        "running": {"details": "Running dataset pipeline", "progress": "step"},
        "paused": {"details": "Pipeline paused", "progress": "step"},
        "complete": {"details": "Pipeline complete", "progress": "steps_completed", "force": "1"},
    },
    "color_brush": {
        "editing": {"details": "Painting image details", "progress": "image"},
    },
    "blur_brush": {
        "editing": {"details": "Brushing blur effects", "progress": "image"},
    },
}

SAFE_PIPELINE_STEP_LABELS: Dict[str, str] = {
    "validate": "Validating dataset",
    "prepare_workspace": "Preparing workspace",
    "offline_tagger_wd_v3": "Auto-tagging images",
    "dataset_tag_editor": "Editing tags",
    "dataset_normalization": "Normalizing captions",
    "zip_result": "Packaging dataset",
    "tag_cleanup_dedup": "Cleaning captions",
    "image_png": "Converting images",
    "photo_adjust_preset": "Adjusting images",
    "combine_dataset": "Combining datasets",
    "flatten_renumber": "Renaming files",
    "stitch_groups": "Stitching images",
    "webtoon_panel_splitter": "Splitting panels",
    "webp_to_png": "Converting images",
    "batch_adjust": "Adjusting images",
    "autotag_offline": "Auto-tagging images",
    "normalize": "Normalizing captions",
    "tag_editor": "Editing tags",
    "combine": "Combining datasets",
    "rename": "Renaming files",
    "merge": "Stitching images",
    "webtoon": "Splitting panels",
    "manual": "Manual review",
    "cleanup": "Cleaning captions",
    "zip": "Packaging dataset",
}

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_activity_key(activity_key: Any) -> str:
    key = str(activity_key or "").strip()
    return key if key in ACTIVITY_PAYLOADS else "home"


def normalize_presence_phase(tool: Any, phase: Any) -> str:
    tool_key = normalize_activity_key(tool)
    phases = SAFE_PRESENCE_PHASES.get(tool_key)
    if not phases:
        return "default"
    phase_key = str(phase or "default").strip()
    return phase_key if phase_key in phases else ("guided_generic" if tool_key == "tags" else "default")


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str) and not value.strip().isdigit():
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_count_pair(current: Any, total: Any) -> Optional[tuple[int, int]]:
    total_int = _safe_int(total)
    current_int = _safe_int(current)
    if total_int is None or current_int is None or total_int < 1 or current_int < 0:
        return None
    return min(current_int, total_int), total_int


def _safe_step_label(raw: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
    return SAFE_PIPELINE_STEP_LABELS.get(key, "")


class NullDiscordPresence:
    @classmethod
    def from_env(cls) -> "NullDiscordPresence":
        return cls()

    @property
    def enabled(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def set_activity(self, activity_key: str) -> None:
        return None

    def report_activity(self, tool: str, **context: Any) -> None:
        return None

    def stop(self) -> None:
        return None


class DiscordPresenceService:
    def __init__(
        self,
        application_id: str,
        *,
        asset_key: str = "batchbench",
        debug: bool = False,
        rate_limit_seconds: float = 12.0,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
    ) -> None:
        self.application_id = str(application_id).strip()
        self.asset_key = (asset_key or "batchbench").strip() or "batchbench"
        self.debug = debug
        self.rate_limit_seconds = max(0.0, float(rate_limit_seconds))
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.retry_max_seconds = max(self.retry_base_seconds, float(retry_max_seconds))
        self.session_started = int(time.time())

        self._activity_context: Dict[str, Any] = {"tool": "home"}
        self._activity_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._logged_unavailable = False
        self._logged_stopped = False

    @classmethod
    def from_env(cls) -> "DiscordPresenceService | NullDiscordPresence":
        if not _env_flag("DISCORD_RICH_PRESENCE_ENABLED", default=False):
            return NullDiscordPresence()

        application_id = os.getenv("DISCORD_APPLICATION_ID", "").strip()
        if not application_id or not application_id.isdigit():
            return NullDiscordPresence()

        return cls(
            application_id,
            asset_key=os.getenv("DISCORD_PRESENCE_ASSET_KEY", "batchbench"),
            debug=_env_flag("DISCORD_PRESENCE_DEBUG", default=False),
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def worker_thread(self) -> Optional[threading.Thread]:
        return self._thread

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="BatchBenchDiscordPresence",
                daemon=True,
            )
            self._thread.start()

    def set_activity(self, activity_key: str) -> None:
        self.report_activity(tool=activity_key)

    def report_activity(self, tool: str, **context: Any) -> None:
        normalized = self._normalize_context({"tool": tool, **context})
        self._activity_context = normalized
        self._queue_latest(normalized)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue_latest(self._activity_context)
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        if self._started and not self._logged_stopped:
            LOGGER.info("Discord Rich Presence stopped.")
            self._logged_stopped = True

    def _queue_latest(self, context: Dict[str, Any]) -> None:
        while True:
            try:
                self._activity_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._activity_queue.put_nowait(dict(context))
        except queue.Full:
            try:
                self._activity_queue.get_nowait()
                self._activity_queue.put_nowait(dict(context))
            except queue.Empty:
                pass

    def _normalize_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        tool = normalize_activity_key(raw.get("tool") or raw.get("activity"))
        context: Dict[str, Any] = {"tool": tool}
        phases = SAFE_PRESENCE_PHASES.get(tool)
        has_structured_context = any(
            key in raw and raw.get(key) not in (None, "")
            for key in ("phase", "current", "total", "step", "step_key")
        )
        if phases and has_structured_context:
            phase = normalize_presence_phase(tool, raw.get("phase"))
            context["phase"] = phase
            phase_cfg = phases.get(phase) or phases.get("default") or {}
            if phase_cfg.get("force") == "1":
                context["force"] = True
            progress = _safe_count_pair(raw.get("current"), raw.get("total"))
            if progress:
                context["current"], context["total"] = progress
            step_label = _safe_step_label(raw.get("step") or raw.get("step_key"))
            if step_label:
                context["step_label"] = step_label
        return context

    def _build_payload(self, activity_key: Any) -> Dict[str, Any]:
        context = activity_key if isinstance(activity_key, dict) else {"tool": activity_key}
        context = self._normalize_context(context)
        tool = context["tool"]
        fixed = ACTIVITY_PAYLOADS[tool]
        phases = SAFE_PRESENCE_PHASES.get(tool)
        if phases and "phase" in context:
            phase_cfg = phases.get(context.get("phase") or "default") or phases.get("default") or {}
            details = phase_cfg.get("details") or fixed["details"]
            state = self._format_progress_state(context, phase_cfg) or fixed["state"]
        else:
            details = fixed["details"]
            state = fixed["state"]
        return {
            "details": details,
            "state": state,
            "large_image": self.asset_key,
            "large_text": "BatchBench",
            "start": self.session_started,
        }

    def _format_progress_state(self, context: Dict[str, Any], phase_cfg: Dict[str, str]) -> str:
        progress_type = phase_cfg.get("progress", "")
        current = context.get("current")
        total = context.get("total")
        if not isinstance(current, int) or not isinstance(total, int) or total < 1 or current < 0:
            return ""
        current = min(current, total)
        if progress_type == "image":
            return f"Image {current} of {total}"
        if progress_type == "processed":
            return f"Processed {current} of {total}"
        if progress_type == "images_processed":
            return f"{current} images processed"
        if progress_type == "steps_completed":
            return f"{current} steps completed"
        if progress_type == "step":
            state = f"Step {current} of {total}"
            step_label = context.get("step_label")
            if isinstance(step_label, str) and step_label:
                state = f"{state} - {step_label}"
            return state
        return ""

    def _worker(self) -> None:
        Presence = self._load_presence_class()
        if Presence is None:
            return

        backoff = self.retry_base_seconds
        was_connected = False
        while not self._stop_event.is_set():
            rpc = None
            try:
                rpc = Presence(self.application_id)
                rpc.connect()
                LOGGER.info("Discord Rich Presence connected.")
                was_connected = True
                backoff = self.retry_base_seconds
                self._connected_loop(rpc)
            except Exception:
                if self.debug:
                    LOGGER.debug("Discord Rich Presence IPC operation failed.", exc_info=True)
                if not self._logged_unavailable and not was_connected:
                    LOGGER.info("Discord Rich Presence unavailable; running without presence.")
                    self._logged_unavailable = True
                elif was_connected:
                    LOGGER.info(
                        "Discord Rich Presence disconnected; retrying in %ss.",
                        int(backoff),
                    )
                    was_connected = False
                self._close_rpc(rpc, clear=False)
                self._sleep_until_stop(backoff)
                backoff = min(self.retry_max_seconds, backoff * 2)
        self._close_rpc(rpc, clear=True)

    def _load_presence_class(self):
        try:
            from pypresence import Presence  # type: ignore

            return Presence
        except Exception:
            if self.debug:
                LOGGER.debug("Discord Rich Presence dependency is unavailable.", exc_info=True)
            LOGGER.info("Discord Rich Presence unavailable; running without presence.")
            self._logged_unavailable = True
            return None

    def _connected_loop(self, rpc: Any) -> None:
        pending_key = self._collect_latest(self._activity_context, 0.0)
        if self._stop_event.is_set():
            return
        rpc.update(**self._build_payload(pending_key))
        last_update = time.monotonic()
        last_sent_key = pending_key

        while not self._stop_event.is_set():
            now = time.monotonic()
            wait_seconds = max(0.0, last_update + self.rate_limit_seconds - now)
            if wait_seconds > 0:
                pending_key = self._collect_latest(pending_key, wait_seconds)
                if self._stop_event.is_set():
                    break
            else:
                pending_key = self._collect_latest(pending_key, 0.25)

            if pending_key == last_sent_key:
                continue
            rpc.update(**self._build_payload(pending_key))
            last_update = time.monotonic()
            last_sent_key = pending_key

    def _collect_latest(self, fallback: Dict[str, Any], max_wait_seconds: float) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.0, max_wait_seconds)
        latest = dict(fallback)
        first = True
        while not self._stop_event.is_set():
            timeout = 0.0 if first else min(0.25, max(0.0, deadline - time.monotonic()))
            first = False
            try:
                latest = self._activity_queue.get(timeout=timeout)
                while True:
                    try:
                        latest = self._activity_queue.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if latest.get("force"):
                return latest
            if time.monotonic() >= deadline:
                return latest
        return latest

    def _sleep_until_stop(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))

    def _close_rpc(self, rpc: Any, *, clear: bool) -> None:
        if rpc is None:
            return
        try:
            if clear:
                try:
                    rpc.clear()
                except Exception:
                    pass
            try:
                rpc.close()
            except Exception:
                pass
        except Exception:
            pass
