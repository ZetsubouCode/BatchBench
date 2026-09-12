from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .base import TaggerAdapter


CAFORMER_PROFILE = "caformer_s36_dbv4"
LEGACY_WD_PROFILE = "wd_swinv2_v3"


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    repo_id: str
    family: str
    adapter_class: Type[TaggerAdapter]
    required_files: tuple[str, ...]
    recommended: bool = False
    legacy_default: bool = False
    runtime: str = "transformers"

    def adapter(self) -> TaggerAdapter:
        return self.adapter_class()


def _profiles() -> Dict[str, ModelProfile]:
    # Imports are local so adapter modules can import shared registry constants safely.
    from .animetimm import AnimeTimmDbv4Adapter
    from .smilingwolf import SmilingWolfWdAdapter

    return {
        CAFORMER_PROFILE: ModelProfile(
            key=CAFORMER_PROFILE,
            display_name="CAFormer S36 dbv4",
            repo_id="animetimm/caformer_s36.dbv4-full",
            family="animetimm_dbv4",
            adapter_class=AnimeTimmDbv4Adapter,
            required_files=("model.onnx", "selected_tags.csv", "preprocess.json", "categories.json"),
            recommended=True,
            runtime="onnx",
        ),
        LEGACY_WD_PROFILE: ModelProfile(
            key=LEGACY_WD_PROFILE,
            display_name="WD SwinV2 Tagger v3",
            repo_id="SmilingWolf/wd-swinv2-tagger-v3",
            family="smilingwolf_wd",
            adapter_class=SmilingWolfWdAdapter,
            required_files=("model.safetensors", "config.json", "selected_tags.csv"),
            legacy_default=True,
            runtime="timm",
        ),
    }


def list_model_profiles() -> List[ModelProfile]:
    return list(_profiles().values())


def get_model_profile(key: str) -> ModelProfile:
    try:
        return _profiles()[str(key or "").strip()]
    except KeyError as exc:
        raise ValueError(f"Unknown curated model profile: {key}") from exc


def resolve_model_profile(model_profile: Optional[str] = None, model_id: Optional[str] = None) -> ModelProfile:
    if str(model_profile or "").strip():
        return get_model_profile(str(model_profile).strip())
    normalized_id = str(model_id or "").strip().lower()
    for profile in list_model_profiles():
        if normalized_id and normalized_id == profile.repo_id.lower():
            return profile
    # Saved configurations predating model_profile were WD-only. Empty and unknown
    # legacy values therefore retain that behavior; HTTP/UI entry points expose only
    # curated profile keys.
    return get_model_profile(LEGACY_WD_PROFILE)
