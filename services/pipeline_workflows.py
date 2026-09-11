from __future__ import annotations

from typing import Any, Dict, List


WORKFLOW_SCHEMA_VERSION = 2


WORKFLOW_CATALOG: Dict[str, Dict[str, str]] = {
    "raw_images": {
        "id": "raw_images",
        "label": "Raw Images -> Tagged Dataset",
        "summary": "Prepare images, auto-tag, review captions, validate, and export.",
    },
    "captioned_dataset": {
        "id": "captioned_dataset",
        "label": "Existing Captioned Dataset -> Final Export",
        "summary": "Normalize, audit, and export existing image + caption pairs.",
    },
    "image_preparation": {
        "id": "image_preparation",
        "label": "Image Preparation Only",
        "summary": "Run image preparation, validate image files, and optionally export.",
    },
}


STRUCTURAL_STEP_BY_OPTION = {
    "webtoon_split": "webtoon_split",
    "merge_groups": "merge_groups",
    "flatten_renumber": "flatten_renumber",
}


PLAN_TEXT = {
    "prepare_workspace": {
        "title": "Prepare isolated working copy",
        "description": "Copy the source dataset into a job workspace before any edits run.",
        "changes_files": False,
        "optional": False,
    },
    "webtoon_split": {
        "title": "Split webtoon pages",
        "description": "Detect panel breaks and create individual panel images.",
        "changes_files": True,
        "optional": True,
    },
    "merge_groups": {
        "title": "Stitch numbered image groups",
        "description": "Combine grouped image parts into stitched images.",
        "changes_files": True,
        "optional": True,
    },
    "flatten_renumber": {
        "title": "Flatten and renumber folders",
        "description": "Flatten nested folders into a clean numbered image set.",
        "changes_files": True,
        "optional": True,
    },
    "webp_to_png": {
        "title": "Convert images to PNG",
        "description": "Create a PNG working set for downstream captioning.",
        "changes_files": True,
        "optional": True,
    },
    "batch_adjust": {
        "title": "Apply photo adjustment preset",
        "description": "Apply the selected image adjustment preset to the working images.",
        "changes_files": True,
        "optional": True,
    },
    "offline_tagger": {
        "title": "Auto-tag images",
        "description": "Generate caption tags with Offline Tagger.",
        "changes_files": True,
        "optional": False,
    },
    "manual_review": {
        "title": "Pause for manual caption review",
        "description": "Open the isolated working copy in Dataset Tag Editor, then resume here.",
        "changes_files": True,
        "optional": True,
    },
    "normalize": {
        "title": "Normalize dataset tags",
        "description": "Apply the selected normalization preset and protected tag rules.",
        "changes_files": True,
        "optional": False,
    },
    "dataset_audit": {
        "title": "Audit image and caption pairs",
        "description": "Validate pairs, empty captions, temp files, backups, duplicate tags, and token length.",
        "changes_files": False,
        "optional": False,
    },
    "export_final": {
        "title": "Export final dataset",
        "description": "Copy clean training files to a timestamped final output folder and optional ZIP.",
        "changes_files": True,
        "optional": False,
    },
}


def bool_opt(options: Dict[str, Any], key: str, default: bool = False) -> bool:
    raw = options.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _clean_csv(raw: Any) -> str:
    return str(raw or "").strip()


def _structural_step(options: Dict[str, Any]) -> List[Dict[str, Any]]:
    choice = str(options.get("structural_preparation") or "none").strip().lower()
    step_id = STRUCTURAL_STEP_BY_OPTION.get(choice)
    if not step_id:
        return []
    return [{"id": step_id, "config": dict(options.get(f"{step_id}_config") or {})}]


def _image_transform_steps(options: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if bool_opt(options, "convert_png"):
        steps.append({"id": "webp_to_png", "config": {}})
    if bool_opt(options, "photo_adjust"):
        cfg = {"preset_name": str(options.get("photo_adjust_preset") or "").strip()}
        for key, value in dict(options.get("photo_adjust_overrides") or {}).items():
            cfg[key] = value
        steps.append({"id": "batch_adjust", "config": cfg})
    return steps


def _normalization_step(options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": "normalize",
        "config": {
            "preset_type": str(options.get("preset_type") or "anime").strip(),
            "preset_file": str(options.get("preset_file") or "").strip(),
            "identity_tags": _clean_csv(options.get("identity_tags")),
            "extra_remove": _clean_csv(options.get("extra_remove")),
            "extra_keep": _clean_csv(options.get("extra_keep")),
            "include_missing_txt": bool_opt(options, "include_missing_txt", True),
            "normalize_order": True,
            "backup_enabled": bool_opt(options, "normalization_backups", True),
        },
    }


def _audit_step(
    require_caption_pairs: bool,
    fail_on_audit_errors: bool,
    include_token_check: bool,
    token_warning_limit: Any,
) -> Dict[str, Any]:
    return {
        "id": "dataset_audit",
        "config": {
            "require_caption_pairs": require_caption_pairs,
            "fail_on_audit_errors": fail_on_audit_errors,
            "include_token_check": include_token_check,
            "token_warning_limit": token_warning_limit or 77,
        },
    }


def _export_step(options: Dict[str, Any], default_zip: bool = True) -> Dict[str, Any]:
    return {
        "id": "export_final",
        "config": {
            "create_zip": bool_opt(options, "create_zip", default_zip),
            "include_txt": bool_opt(options, "include_txt", True),
        },
    }


def compile_workflow(workflow_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = str(workflow_id or "raw_images").strip().lower()
    options = dict(options or {})
    steps: List[Dict[str, Any]] = []
    cfg: Dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_id": workflow_id,
        "workflow_options": options,
    }

    if workflow_id == "captioned_dataset":
        if bool_opt(options, "manual_review", True):
            steps.append({"id": "manual_review", "config": {"message": "Review captions, then return here and resume."}})
        steps.append(_normalization_step(options))
        require_pairs = bool_opt(options, "require_caption_pairs", True)
        steps.append(_audit_step(require_pairs, True, True, options.get("token_warning_limit")))
        steps.append(_export_step(options, default_zip=True))
        cfg.update(
            {
                "require_caption_pairs": require_pairs,
                "fail_on_audit_errors": True,
                "include_token_check": True,
                "token_warning_limit": int(options.get("token_warning_limit") or 77),
            }
        )
    elif workflow_id == "image_preparation":
        steps.extend(_structural_step(options))
        steps.extend(_image_transform_steps(options))
        steps.append(_audit_step(False, False, False, options.get("token_warning_limit")))
        if bool_opt(options, "export_final", True):
            steps.append(_export_step(options, default_zip=bool_opt(options, "create_zip", False)))
        cfg.update(
            {
                "require_caption_pairs": False,
                "fail_on_audit_errors": False,
                "include_token_check": False,
                "token_warning_limit": int(options.get("token_warning_limit") or 77),
            }
        )
    else:
        workflow_id = "raw_images"
        cfg["workflow_id"] = workflow_id
        steps.extend(_structural_step(options))
        steps.extend(_image_transform_steps(options))
        if bool_opt(options, "auto_tag", True):
            steps.append(
                {
                    "id": "offline_tagger",
                    "config": {
                        "threshold_mode": str(options.get("threshold_mode") or "mcut").strip(),
                        "general_threshold": options.get("general_threshold", 0.35),
                        "character_threshold": options.get("character_threshold", 0.75),
                        "trigger_tag": str(options.get("trigger_tag") or "").strip(),
                        "tag_policy": str(options.get("tag_policy") or "character_identity_omitted").strip(),
                        "replace_existing_captions": bool_opt(options, "replace_existing_captions", True),
                        "policy_mcut_min_general_tags": options.get("policy_mcut_min_general_tags", 0),
                        "block_permanent_marks": bool_opt(options, "block_permanent_marks", False),
                        "policy_keep_tags": _clean_csv(options.get("policy_keep_tags")),
                        "policy_block_tags": _clean_csv(options.get("policy_block_tags")),
                        "policy_block_regex": _clean_csv(options.get("policy_block_regex")),
                        "include_character": True,
                        "include_rating": False,
                        "dedupe": True,
                        "sort_tags": True,
                        "keep_existing_tags": False,
                    },
                }
            )
        if bool_opt(options, "manual_review", True):
            steps.append({"id": "manual_review", "config": {"message": "Review generated captions, then return here and resume."}})
        steps.append(_normalization_step(options))
        steps.append(_audit_step(True, True, True, options.get("token_warning_limit")))
        steps.append(_export_step(options, default_zip=True))
        cfg.update(
            {
                "trigger_tag": str(options.get("trigger_tag") or "").strip(),
                "require_caption_pairs": True,
                "fail_on_audit_errors": True,
                "include_token_check": True,
                "token_warning_limit": int(options.get("token_warning_limit") or 77),
            }
        )

    cfg["steps"] = steps
    return cfg


def plan_items_for_steps(steps: List[Dict[str, Any]], output_dir: str = "") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    prep = dict(PLAN_TEXT["prepare_workspace"])
    prep["id"] = "prepare_workspace"
    items.append(prep)
    for step in steps:
        step_id = str(step.get("id") or "")
        base = dict(PLAN_TEXT.get(step_id) or {})
        if not base:
            base = {
                "title": step_id.replace("_", " ").title(),
                "description": "Run this custom workflow step.",
                "changes_files": True,
                "optional": False,
            }
        base["id"] = step_id
        if step_id == "export_final" and output_dir:
            base["expected_output"] = output_dir
        items.append(base)
    return items
