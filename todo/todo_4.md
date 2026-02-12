# ADD-ON CODEX TASK: Global fix for wrong color detection (no blacklist dependency)

## Goal
Implement a global, per-image "color sanity" system to reduce false-positive color tags (e.g., "blue hair", "blue skin") WITHOUT using blacklist/exclusion as the main solution.
Also add a safe preprocessing verification option (RGB vs BGR), because wrong channel ordering can cause systematic color bias in some tagger pipelines.

References:
- WD14 tagger threshold concept and default `--thresh` behavior (kohya sd-scripts wd14 docs)
- Known RGB->BGR preprocessing fix in a wd-swinv2-tagger-v3-related image processor fork (commit note)

---

## A) Add preprocessing toggle: RGB vs BGR (evidence-based safeguard)
### Why
Some implementations of WD SwinV2 taggers have required converting RGB->BGR to fix color-related issues. Provide a toggle to test this quickly and avoid systemic color bias.

### Implement
1) Extend TaggerOptions:
- `input_color_order: str`  # "rgb" (default) or "bgr"
2) In `run_tagger`, right after `im.convert("RGB")`:
- if `opts.input_color_order == "bgr"`:
  - swap channels before passing to `processor`
  - (PIL) convert to numpy, `arr = arr[..., ::-1]`, then back to Image

3) Log the chosen mode.
4) Keep default "rgb" to preserve current behavior.

### Quick sanity test
Add a CLI/self-check utility:
- pick 10 images, run preview with rgb and bgr
- count how often false color tags appear (e.g., "* hair", "* skin", "* eyes")
- user can choose mode

---

## B) Implement per-image Color Sanity (data-driven, no blacklist)
### Core idea
Compute a cheap color histogram / hue presence score from the image pixels (downscaled), then validate color-attribute tags:
- If tagger predicts `<color> hair/skin/eyes` but the image does not contain that color above a minimum ratio, DROP it
- If the tag score is extremely high, optionally keep it (safety valve), because hair could occupy small area

This is NOT a static blacklist: it is a dynamic, per-image validation.

### Add options
Extend TaggerOptions:
- `enable_color_sanity: bool` (default true)
- `color_ratio_threshold: float` (default 0.006)  # 0.6% of pixels
- `color_min_saturation: float` (default 0.20)    # ignore low-sat pixels for hue bins
- `color_min_value: float` (default 0.15)         # ignore very dark pixels for hue bins
- `color_keep_if_score_ge: float` (default 0.92)  # if model is extremely confident, keep even if ratio slightly low
- `color_downscale: int` (default 256)            # resize longest side for speed

### Color mapping (HSV)
Implement `_estimate_color_presence(im) -> dict[color_name] = ratio`:
1) Downscale (keep aspect), convert to HSV
2) For each pixel:
- ignore if sat < min_saturation OR val < min_value
- count hue into bins
3) Map hue ranges to color names:
- red (wrap), orange, yellow, green, cyan, blue, purple, pink
Special cases (not hue-only):
- white: val high & sat low
- black: val low
- gray: sat low & mid val
- brown: often low-ish val + mid sat around orange/yellow hue (approx heuristic)

Return ratios.

### Tag filtering logic
Modify `_build_tags()` to accept an additional argument:
- `color_presence: Optional[Dict[str, float]]`
and ALSO keep access to the per-tag `score`.

Implement `_is_color_attribute_tag(tag: str) -> Optional[color_name]`:
- Match patterns:
  - `"{color} hair"`, `"{color} eyes"`, `"{color} skin"`
  - handle underscores if replace_underscore is false; normalize with spaces for matching
Return the detected color, else None.

When iterating labels:
- If it is a color-attribute tag and `enable_color_sanity`:
  - presence = color_presence.get(color, 0.0)
  - if presence < color_ratio_threshold AND score < color_keep_if_score_ge:
      - SKIP this tag (do not add to general/character lists)
  - else keep normal threshold checks

Important:
- Apply this filter BEFORE comparing to `general_threshold` so it can suppress false positives even if score is above threshold.

### Where to compute `color_presence`
Inside `run_tagger` loop, you already load PIL images per batch.
- Build a `color_presence_list` aligned to `batch_paths`:
  - if enable_color_sanity: compute per-image presence dictionary
  - else: None
- Pass the corresponding dict to `_build_tags(...)`.

### Performance
- Downscale to max 256px
- Use numpy vectorization if possible (fast)
- This adds very small overhead compared to model inference.

---

## C) Add logging for debugging & trust
For preview samples, include optional debug line:
- If a color tag was dropped due to sanity check, record:
  - dropped tag name, model score, measured presence ratio

But keep default logs clean; enable with `debug_color_sanity` flag.

---

## D) Tests (must-have)
### Unit test: color presence
- Create synthetic images (solid blue, solid red, grayscale) and verify presence dict:
  - blue image => blue ratio high
  - gray image => gray/white ratio high, hue colors low

### Unit test: color tag gating
- Feed mocked labels and probs:
  - tag "blue hair" score 0.7, presence blue 0.0 -> should be dropped
  - tag "blue hair" score 0.97, presence blue 0.0 -> kept (safety valve)

### Regression test
- Ensure non-color tags are unaffected.

---

## E) Acceptance criteria
- If an image has no blue pixels (by HSV presence), "blue hair/blue skin/blue eyes" should NOT appear unless model confidence is extremely high (>= keep_if_score_ge).
- Works for all colors via hue mapping, not a static blacklist.
- User can still use single threshold mode; color sanity operates independently.

