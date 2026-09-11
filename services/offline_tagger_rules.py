import re
from typing import List, Pattern

BUCKET_BACKGROUND_PLACE = "background_place"
BUCKET_OBJECT_PROP = "object_prop"
BUCKET_POSE_ACTION = "pose_action"
BUCKET_LIMB_ACTION = "limb_action"
BUCKET_CAMERA_COMPOSITION = "camera_composition"
BUCKET_LIGHTING_ENVIRONMENT = "lighting_environment"
BUCKET_APPEARANCE_IDENTITY = "appearance_identity"
BUCKET_CLOTHING_OUTFIT = "clothing_outfit"
BUCKET_UNKNOWN = "unknown"

_SELECTIVE_NORM_RE = re.compile(r"[^a-z0-9_]+")


def normalize_rule_tag(tag: str) -> str:
    text = (tag or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = _SELECTIVE_NORM_RE.sub("_", text)
    return re.sub(r"_+", "_", text).strip("_")


EXACT_ALLOW_BACKGROUND = {
    "background",
    "simple_background",
    "white_background",
    "black_background",
    "gradient_background",
    "blurred_background",
    "detailed_background",
    "scenery",
    "landscape",
    "cityscape",
    "indoors",
    "outdoors",
    "forest",
    "street",
    "bedroom",
    "classroom",
    "office",
    "library",
    "kitchen",
    "bathroom",
    "living_room",
    "hallway",
    "corridor",
    "window",
    "door",
    "building",
    "sky",
    "blue_sky",
    "cloud",
    "clouds",
    "night",
    "day",
    "sunset",
    "sunrise",
    "moon",
    "stars",
    "starry_sky",
    "tree",
    "trees",
    "grass",
    "flower",
    "flowers",
    "river",
    "mountain",
    "beach",
    "sea",
    "ocean",
    "lake",
    "road",
    "bridge",
    "sidewalk",
    "alley",
    "park",
    "garden",
}

EXACT_ALLOW_CAMERA = {
    "close_up",
    "portrait",
    "upper_body",
    "cowboy_shot",
    "full_body",
    "from_above",
    "from_below",
    "dutch_angle",
    "side_view",
    "profile",
    "depth_of_field",
}

EXACT_ALLOW_LIGHTING = {
    "day",
    "night",
    "sunset",
    "sunrise",
    "indoors",
    "outdoors",
    "rain",
    "snow",
    "dramatic_lighting",
    "backlighting",
    "rim_light",
    "shadow",
    "cloudy_sky",
}

EXACT_ALLOW_OBJECT = {
    "chair",
    "table",
    "desk",
    "book",
    "books",
    "cup",
    "mug",
    "bottle",
    "glass",
    "phone",
    "smartphone",
    "laptop",
    "computer",
    "keyboard",
    "mouse",
    "sword",
    "gun",
    "staff",
    "umbrella",
    "bag",
    "backpack",
    "car",
    "bicycle",
    "train",
    "bus",
    "airplane",
    "ship",
    "boat",
    "food",
    "drink",
    "camera",
    "microphone",
    "instrument",
    "guitar",
    "piano",
    "violin",
    "holding_book",
    "holding_sword",
    "holding_gun",
    "holding_staff",
    "holding_umbrella",
    "holding_phone",
    "holding_cup",
    "holding_camera",
}

EXACT_ALLOW_POSE = {
    "standing",
    "sitting",
    "kneeling",
    "lying",
    "walking",
    "running",
    "jumping",
    "leaning",
    "crouching",
    "squatting",
    "bent_over",
    "from_behind",
    "side_view",
    "looking_back",
}

EXACT_ALLOW_LIMB = {
    "arms_up",
    "arm_up",
    "crossed_arms",
    "hand_on_hip",
    "reaching",
    "pointing",
    "spread_legs",
    "crossed_legs",
    "raised_leg",
    "one_leg_up",
    "legs_apart",
    "hands_up",
    "hand_up",
}

EXACT_DENY_APPEARANCE = {
    "1girl",
    "1boy",
    "2girls",
    "2boys",
    "3girls",
    "3boys",
    "multiple_girls",
    "multiple_boys",
    "solo",
    "blue_hair",
    "black_hair",
    "blonde_hair",
    "brown_hair",
    "green_hair",
    "pink_hair",
    "purple_hair",
    "orange_hair",
    "yellow_hair",
    "red_hair",
    "white_hair",
    "gray_hair",
    "silver_hair",
    "blue_eyes",
    "red_eyes",
    "green_eyes",
    "brown_eyes",
    "yellow_eyes",
    "purple_eyes",
    "pink_eyes",
    "black_eyes",
    "long_hair",
    "short_hair",
    "very_long_hair",
    "twintails",
    "ponytail",
    "braid",
    "ahoge",
    "bangs",
    "hair_ornament",
    "pointy_ears",
    "fang",
    "freckles",
    "dark_skin",
    "pale_skin",
    "tan",
    "breasts",
    "medium_breasts",
    "huge_breasts",
    "flat_chest",
    "cleavage",
    "navel",
    "thighs",
    "wide_hips",
    "long_legs",
    "bare_arms",
    "bare_legs",
    "bare_shoulders",
    "barefoot",
    "feet",
    "toes",
    "fingernails",
    "lips",
    "nose",
    "closed_eyes",
    "half_closed_eyes",
    "one_eye_closed",
    "heterochromia",
    "animal_ears",
    "cat_ears",
    "fox_ears",
    "elf",
    "dark_skinned_female",
    "dark_skinned_male",
    "large_breasts",
    "small_breasts",
}

EXACT_DENY_CLOTHING = {
    "dress",
    "shirt",
    "jacket",
    "coat",
    "hoodie",
    "robe",
    "armor",
    "uniform",
    "school_uniform",
    "skirt",
    "pants",
    "shorts",
    "jeans",
    "kimono",
    "bikini",
    "swimsuit",
    "leotard",
    "lingerie",
    "gloves",
    "boots",
    "shoes",
    "socks",
    "thighhighs",
    "pantyhose",
    "stockings",
    "hat",
    "cap",
    "cape",
    "ribbon",
    "hair_ribbon",
    "neck_ribbon",
    "necktie",
    "tie",
    "scarf",
    "headwear",
    "short_sleeves",
    "long_sleeves",
    "sleeveless",
    "detached_sleeves",
    "puffy_sleeves",
    "collar",
    "sailor_collar",
    "hood",
    "belt",
    "apron",
    "glasses",
    "earrings",
    "necklace",
    "bracelet",
    "wristband",
    "choker",
}

REGEX_ALLOW_BACKGROUND = [
    r"(?:^|_)(background|scenery|landscape|cityscape)(?:$|_)",
    r"(?:^|_)(indoors|outdoors)(?:$|_)",
    r"(?:^|_)(forest|street|bedroom|classroom|office|library|kitchen|bathroom|room|hallway|corridor|park|garden)(?:$|_)",
    r"(?:^|_)(window|door|building|road|bridge|sidewalk|alley)(?:$|_)",
    r"(?:^|_)(sky|cloud|night|day|sunset|sunrise|moon|star)(?:$|_)",
    r"(?:^|_)(tree|grass|flower|river|mountain|beach|sea|ocean|lake)(?:$|_)",
    r"_background$",
]

REGEX_ALLOW_CAMERA = [
    r"^(close_up|portrait|upper_body|cowboy_shot|full_body)$",
    r"^(from_above|from_below|dutch_angle|side_view|profile)$",
    r"^(depth_of_field)$",
]

REGEX_ALLOW_LIGHTING = [
    r"^(day|night|sunset|sunrise|indoors|outdoors)$",
    r"^(rain|snow|dramatic_lighting|backlighting|rim_light|shadow|cloudy_sky)$",
]

REGEX_ALLOW_OBJECT = [
    r"(?:^|_)(chair|table|desk|book|cup|mug|bottle|phone|laptop|camera|microphone)(?:$|_)",
    r"(?:^|_)(sword|gun|staff|umbrella|bag|backpack)(?:$|_)",
    r"(?:^|_)(car|bicycle|train|bus|airplane|ship|boat)(?:$|_)",
    r"(?:^|_)(food|drink|instrument|guitar|piano|violin)(?:$|_)",
    r"^holding_(book|sword|gun|staff|umbrella|bag|phone|cup|camera)$",
]

REGEX_ALLOW_POSE = [
    r"(?:^|_)(standing|sitting|kneeling|lying|walking|running|jumping)(?:$|_)",
    r"(?:^|_)(leaning|crouching|squatting|bent_over)(?:$|_)",
    r"(?:^|_)(from_behind|side_view|looking_back)(?:$|_)",
]

REGEX_ALLOW_LIMB = [
    r"(?:^|_)(arm_up|arms_up|hand_up|hands_up|crossed_arms|hand_on_hip)(?:$|_)",
    r"(?:^|_)(reaching|pointing)(?:$|_)",
    r"(?:^|_)(spread_legs|crossed_legs|raised_leg|one_leg_up|legs_apart)(?:$|_)",
]

REGEX_DENY_APPEARANCE = [
    r"(?:^|_)(?:red|blue|green|brown|black|blonde|pink|purple|orange|yellow|gray|white|silver)_(?:hair|eyes|skin)(?:$|_)",
    r"(?:^|_)(hair|eyes|eye|skin|face|mouth|lips|nose|fang|freckles|ears?)(?:$|_)",
    r"(?:^|_)(long_hair|short_hair|very_long_hair|medium_hair|twintails|ponytail|braid|braids|single_braid|hair_bun|drill_hair|messy_hair|wavy_hair|straight_hair|ahoge|bangs|hair_between_eyes|hair_over_one_eye|one_eye_covered|hair_ornament)(?:$|_)",
    r"(?:^|_)(closed_eyes|half_closed_eyes|one_eye_closed|heterochromia)(?:$|_)",
    r"(?:^|_)(1girl|1boy|2girls|2boys|3girls|3boys|multiple_girls|multiple_boys|solo|girl|boy|male|female|male_focus|female_focus)(?:$|_)",
    r"(?:^|_)(pointy_ears|animal_ears|cat_ears|fox_ears|elf|dark_skin|dark_skinned|pale_skin|tan|colored_skin)(?:$|_)",
    r"(?:^|_)(breasts?|large_breasts|small_breasts|medium_breasts|huge_breasts|flat_chest|cleavage|navel|thighs?|wide_hips|long_legs|bare_arms|bare_legs|bare_shoulders|barefoot|feet|toes|fingernails)(?:$|_)",
    r"_(hair|eyes|skin|ears|breasts|thighs|legs|arms|shoulders|feet)$",
]

REGEX_DENY_CLOTHING = [
    r"(?:^|_)(dress|shirt|jacket|coat|hoodie|robe|armor|uniform|school_uniform)(?:$|_)",
    r"(?:^|_)(skirt|pants|shorts|jeans|kimono|bikini|swimsuit|leotard|lingerie|underwear|panties|bra)(?:$|_)",
    r"(?:^|_)(glove|gloves|boot|boots|shoe|shoes|sock|socks|thighhighs|pantyhose|stockings)(?:$|_)",
    r"(?:^|_)(hat|cap|cape|ribbon|hair_ribbon|neck_ribbon|necktie|tie|scarf|headwear|hood)(?:$|_)",
    r"(?:^|_)(short_sleeves|long_sleeves|sleeveless|detached_sleeves|puffy_sleeves|collar|sailor_collar|belt|apron)(?:$|_)",
    r"(?:^|_)(glasses|earrings|necklace|bracelet|wristband|choker)(?:$|_)",
    r"_(dress|shirt|jacket|coat|hoodie|robe|armor|uniform|skirt|pants|shorts|jeans|kimono|bikini|swimsuit|leotard|lingerie|sleeves|collar|ribbon)$",
]


def _compile(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


COMPILED_REGEX_ALLOW_BACKGROUND = _compile(REGEX_ALLOW_BACKGROUND)
COMPILED_REGEX_ALLOW_CAMERA = _compile(REGEX_ALLOW_CAMERA)
COMPILED_REGEX_ALLOW_LIGHTING = _compile(REGEX_ALLOW_LIGHTING)
COMPILED_REGEX_ALLOW_OBJECT = _compile(REGEX_ALLOW_OBJECT)
COMPILED_REGEX_ALLOW_POSE = _compile(REGEX_ALLOW_POSE)
COMPILED_REGEX_ALLOW_LIMB = _compile(REGEX_ALLOW_LIMB)
COMPILED_REGEX_DENY_APPEARANCE = _compile(REGEX_DENY_APPEARANCE)
COMPILED_REGEX_DENY_CLOTHING = _compile(REGEX_DENY_CLOTHING)
