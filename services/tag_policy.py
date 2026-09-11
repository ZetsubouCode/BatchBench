from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

from utils.tags import tag_compare_key


POLICY_NONE = "none"
POLICY_CHARACTER_IDENTITY_OMITTED = "character_identity_omitted"
DEFAULT_TAG_POLICY = POLICY_CHARACTER_IDENTITY_OMITTED

GROUP_CHARACTER_NAME = "character_name"
GROUP_DEMOGRAPHIC = "demographic"
GROUP_IDENTITY_APPEARANCE = "identity_appearance"
GROUP_IDENTITY_BODY = "identity_body"
GROUP_PERMANENT_MARK = "permanent_mark"


def normalize_tag(tag: str) -> str:
    text = tag_compare_key(tag).replace("-", "_")
    text = re.sub(r"[^a-z0-9_:*]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class TagPolicyProfile:
    name: str
    label: str
    version: int
    blocked_groups: frozenset[str]
    blocked_exact: frozenset[str]
    keep_exact: frozenset[str]
    blocked_regex: Tuple[Pattern[str], ...]
    permanent_mark_exact: frozenset[str] = frozenset()
    permanent_mark_regex: Tuple[Pattern[str], ...] = ()


@dataclass
class CompiledTagPolicy:
    profile: TagPolicyProfile
    blocked_mask: List[bool]
    reason_by_index: List[str]
    group_by_index: List[str]
    custom_keep: frozenset[str] = frozenset()
    custom_block: frozenset[str] = frozenset()
    custom_block_regex: Tuple[Pattern[str], ...] = ()
    permanent_marks: bool = False
    label_to_index: Dict[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.profile.name != POLICY_NONE

    def decision_for_index(self, index: int) -> Tuple[bool, str, str]:
        if index < 0 or index >= len(self.blocked_mask):
            return False, "", ""
        return self.blocked_mask[index], self.reason_by_index[index], self.group_by_index[index]

    def decision_for_tag(self, tag: str) -> Tuple[bool, str, str]:
        norm = normalize_tag(tag)
        index = self.label_to_index.get(norm)
        if index is not None:
            return self.decision_for_index(index)
        return decide_tag(self.profile, norm, None, self.custom_keep, self.custom_block, self.custom_block_regex)


KEEP_EXACT = frozenset(
    normalize_tag(t)
    for t in (
        "closed_eyes",
        "half-closed_eyes",
        "wink",
        "looking_at_viewer",
        "looking_away",
        "hair_ornament",
        "hair_ribbon",
        "hairclip",
        "smile",
        "open_mouth",
        "standing",
        "sitting",
        "cowboy_shot",
        "upper_body",
        "full_body",
    )
)

DEMOGRAPHIC_EXACT = frozenset(
    normalize_tag(t)
    for t in (
        "solo",
        "multiple_girls",
        "multiple_boys",
        "multiple_girls",
        "multiple_boys",
        "multiple_others",
        "no_humans",
        "girl",
        "boy",
        "man",
        "woman",
        "male",
        "female",
    )
)

APPEARANCE_EXACT = frozenset(
    normalize_tag(t)
    for t in (
        "long_hair",
        "short_hair",
        "medium_hair",
        "very_long_hair",
        "ponytail",
        "twintails",
        "braid",
        "braids",
        "bangs",
        "blunt_bangs",
        "hair_between_eyes",
        "freckles",
        "dark_skin",
        "tan",
        "pale_skin",
        "pointy_ears",
        "animal_ears",
    )
)

BODY_EXACT = frozenset(
    normalize_tag(t)
    for t in (
        "breasts",
        "large_breasts",
        "small_breasts",
        "medium_breasts",
        "flat_chest",
        "curvy",
        "wide_hips",
        "thick_thighs",
        "narrow_waist",
        "muscular",
        "abs",
    )
)

PERMANENT_MARK_EXACT = frozenset(
    normalize_tag(t)
    for t in (
        "tattoo",
        "scar",
        "birthmark",
        "mole",
        "piercing",
        "arm_tattoo",
        "facial_scar",
        "mole_under_eye",
        "ear_piercing",
    )
)


def _compile(patterns: Iterable[str]) -> Tuple[Pattern[str], ...]:
    out: List[Pattern[str]] = []
    for pattern in patterns:
        out.append(re.compile(pattern, flags=re.IGNORECASE))
    return tuple(out)


DEMOGRAPHIC_REGEX = _compile(
    (
        r"^[0-9]+(?:girl|boy|other)s?$",
        r"^[0-9]+(?:girls|boys|others)$",
        r"^(?:multiple|group)_",
    )
)

APPEARANCE_REGEX = _compile(
    (
        r"^(?:aqua|black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white|yellow|multicolored|two_tone)_hair$",
        r"^(?:aqua|black|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white|yellow|heterochromia)_eyes$",
        r"^(?:dark|light|pale|tan|brown|black|white)_skin$",
        r"^(?:long|short|medium|very_long|messy|spiked|curly|wavy|straight)_hair$",
        r"^(?:side|blunt|swept|crossed|parted)_bangs$",
    )
)

BODY_REGEX = _compile(
    (
        r"^(?:large|small|medium|huge|gigantic)_breasts$",
        r"^(?:wide|narrow)_hips$",
        r"^(?:thick|slender)_thighs$",
    )
)

PERMANENT_MARK_REGEX = _compile(
    (
        r"^(?:.+_)?tattoo$",
        r"^(?:.+_)?scar$",
        r"^(?:.+_)?birthmark$",
        r"^(?:.+_)?mole(?:_.+)?$",
        r"^(?:.+_)?piercing$",
    )
)


NONE_PROFILE = TagPolicyProfile(
    name=POLICY_NONE,
    label="None",
    version=1,
    blocked_groups=frozenset(),
    blocked_exact=frozenset(),
    keep_exact=frozenset(),
    blocked_regex=(),
)

CHARACTER_IDENTITY_OMITTED_PROFILE = TagPolicyProfile(
    name=POLICY_CHARACTER_IDENTITY_OMITTED,
    label="Character - omit identity",
    version=1,
    blocked_groups=frozenset(
        {
            GROUP_CHARACTER_NAME,
            GROUP_DEMOGRAPHIC,
            GROUP_IDENTITY_APPEARANCE,
            GROUP_IDENTITY_BODY,
        }
    ),
    blocked_exact=DEMOGRAPHIC_EXACT | APPEARANCE_EXACT | BODY_EXACT,
    keep_exact=KEEP_EXACT,
    blocked_regex=DEMOGRAPHIC_REGEX + APPEARANCE_REGEX + BODY_REGEX,
    permanent_mark_exact=PERMANENT_MARK_EXACT,
    permanent_mark_regex=PERMANENT_MARK_REGEX,
)


PROFILES = {
    POLICY_NONE: NONE_PROFILE,
    POLICY_CHARACTER_IDENTITY_OMITTED: CHARACTER_IDENTITY_OMITTED_PROFILE,
}


def get_profile(name: str) -> TagPolicyProfile:
    return PROFILES.get((name or DEFAULT_TAG_POLICY).strip().lower(), CHARACTER_IDENTITY_OMITTED_PROFILE)


def _regex_group(profile: TagPolicyProfile, tag: str) -> str:
    if profile.name != POLICY_CHARACTER_IDENTITY_OMITTED:
        return ""
    if any(p.search(tag) for p in DEMOGRAPHIC_REGEX):
        return GROUP_DEMOGRAPHIC
    if any(p.search(tag) for p in APPEARANCE_REGEX):
        return GROUP_IDENTITY_APPEARANCE
    if any(p.search(tag) for p in BODY_REGEX):
        return GROUP_IDENTITY_BODY
    if any(p.search(tag) for p in profile.permanent_mark_regex):
        return GROUP_PERMANENT_MARK
    return ""


def _exact_group(tag: str) -> str:
    if tag in DEMOGRAPHIC_EXACT:
        return GROUP_DEMOGRAPHIC
    if tag in APPEARANCE_EXACT:
        return GROUP_IDENTITY_APPEARANCE
    if tag in BODY_EXACT:
        return GROUP_IDENTITY_BODY
    if tag in PERMANENT_MARK_EXACT:
        return GROUP_PERMANENT_MARK
    return ""


def decide_tag(
    profile: TagPolicyProfile,
    tag: str,
    category: Optional[int],
    custom_keep: Iterable[str] = (),
    custom_block: Iterable[str] = (),
    custom_block_regex: Iterable[Pattern[str]] = (),
    character_category_id: Optional[int] = None,
    permanent_marks: bool = False,
) -> Tuple[bool, str, str]:
    norm = normalize_tag(tag)
    if profile.name == POLICY_NONE:
        return False, "", ""
    keep_set = {normalize_tag(t) for t in custom_keep}
    block_set = {normalize_tag(t) for t in custom_block}
    if norm in keep_set or norm in profile.keep_exact:
        return False, "", ""
    if norm in block_set:
        return True, "custom_block", "custom_block"
    if category is not None and character_category_id is not None and category == character_category_id:
        return True, GROUP_CHARACTER_NAME, GROUP_CHARACTER_NAME
    if norm in profile.blocked_exact:
        group = _exact_group(norm) or "blocked_exact"
        return True, group, group
    if permanent_marks and norm in profile.permanent_mark_exact:
        return True, GROUP_PERMANENT_MARK, GROUP_PERMANENT_MARK
    for pattern in custom_block_regex:
        if pattern.search(norm):
            return True, "custom_block_regex", "custom_block_regex"
    group = _regex_group(profile, norm)
    if group == GROUP_PERMANENT_MARK and not permanent_marks:
        return False, "", ""
    if group:
        return True, group, group
    return False, "", ""


def compile_regex(patterns: Iterable[str]) -> Tuple[Pattern[str], ...]:
    out: List[Pattern[str]] = []
    for pattern in patterns or []:
        try:
            out.append(re.compile(str(pattern), flags=re.IGNORECASE))
        except re.error:
            continue
    return tuple(out)


def compile_policy(
    name: str,
    labels: List[str],
    categories: Optional[List[Optional[int]]],
    character_category_id: Optional[int],
    *,
    custom_keep: Iterable[str] = (),
    custom_block: Iterable[str] = (),
    custom_block_regex: Iterable[str] = (),
    permanent_marks: bool = False,
) -> CompiledTagPolicy:
    profile = get_profile(name)
    keep = frozenset(normalize_tag(t) for t in custom_keep or () if normalize_tag(t))
    block = frozenset(normalize_tag(t) for t in custom_block or () if normalize_tag(t))
    block_regex = compile_regex(custom_block_regex)
    blocked_mask: List[bool] = []
    reason_by_index: List[str] = []
    group_by_index: List[str] = []
    label_to_index: Dict[str, int] = {}
    for idx, raw in enumerate(labels or []):
        norm = normalize_tag(raw)
        if norm and norm not in label_to_index:
            label_to_index[norm] = idx
        category = categories[idx] if categories is not None and idx < len(categories) else None
        blocked, reason, group = decide_tag(
            profile,
            norm,
            category,
            keep,
            block,
            block_regex,
            character_category_id=character_category_id,
            permanent_marks=permanent_marks,
        )
        blocked_mask.append(blocked)
        reason_by_index.append(reason)
        group_by_index.append(group)
    return CompiledTagPolicy(
        profile=profile,
        blocked_mask=blocked_mask,
        reason_by_index=reason_by_index,
        group_by_index=group_by_index,
        custom_keep=keep,
        custom_block=block,
        custom_block_regex=block_regex,
        permanent_marks=permanent_marks,
        label_to_index=label_to_index,
    )


def audit_tags(tags: Iterable[str], compiled: Optional[CompiledTagPolicy], trigger_tag: str = "") -> List[Tuple[str, str, str]]:
    if compiled is None or not compiled.enabled:
        return []
    trigger_norm = normalize_tag(trigger_tag)
    leaks: List[Tuple[str, str, str]] = []
    for tag in tags or []:
        if trigger_norm and normalize_tag(tag) == trigger_norm:
            continue
        blocked, reason, group = compiled.decision_for_tag(tag)
        if blocked:
            leaks.append((tag, reason, group))
    return leaks
