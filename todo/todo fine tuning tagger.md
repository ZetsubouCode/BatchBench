# CODEX INSTRUCTION — Simplify Offline Tagger Parameters (Beginner-Friendly + Safe Advanced)
Semua instruksi yang ada disini bersifat suggestive, jadi jika codex memiliki solusi yang lebih efisien dan optimal maka prioritaskan solusi milik codex.

## Goal
Refactor **Offline Tagger** UI + backend so:
1) **All WD color-correction knobs are removed** from UI and become **mandatory internal policy** (permanent).
2) Remove **overly-complex / uncommon** parameters (not typically seen in online/offline taggers).
3) Keep **beginner-friendly controls** only.
4) Still provide a small set of **advanced enhancements** used by serious trainers, BUT:
   - default state = **does not change results** (or minimal impact)
   - only affects results when explicitly enabled or set
   - does not make the workflow complex.

---

## 0) Define the new parameter tiers (single-source spec)

### Tier A — Beginner (always visible)
These are the only knobs a casual/beginner user should need.

**Input**
- `input_dir` (folder picker)
- `recursive` (checkbox)
- `limit` (int, optional)

**Model / Performance**
- `batch_size` (int)  ← explain: "bigger = faster, needs more VRAM"

**Tag Output**
- `write_mode` (radio): `overwrite | append | skip_if_exists`
- `min_general` (float, default 0.35)
- `min_character` (float, default 0.75)
- `threshold_mode` (radio): `fixed | mcut` (default mcut)
- `include_rating` (checkbox, default off)
- `include_character` (checkbox, default on)
- `trigger_tag` (text, optional)
- `exclude_tags` (text, optional, comma-separated)

**Preview**
- `preview_only` (checkbox)
- `preview_limit` (int, default 20)

> NOTE: Hide “model_id” from beginner tier if you always use WD family. If you still want it, keep it under Advanced.

---

### Tier B — Advanced (collapsed panel; default off / no-effect)
Only include advanced knobs that:
- are actually used by advanced trainers,
- are safe,
- are “enhancement-only” unless turned on.

**Advanced 1 — Tag Budget (enhancement)**
- `max_general_tags` (int, default 0 meaning unlimited / no cap)
- `max_character_tags` (int, default 0)
- Explanation: “cap tag count to reduce noise; default 0 = unchanged”.

**Advanced 2 — Tag Merging behavior (no-effect by default)**
- `dedupe` (checkbox, default on; safe, no behavior surprise)
- `sort_tags` (checkbox, default on; safe)
- `keep_existing_tags` (checkbox, default on if write_mode=append; otherwise ignored)

**Advanced 3 — Character priority (optional enhancement)**
- `character_topk` (int, default 0, no-effect)
  - when > 0: only keep top-k character tags by score.

**Advanced 4 — Output formatting (safe)**
- `newline_end` (checkbox, default on)
- `strip_whitespace` (checkbox, default on)

**Optional advanced: model selection**
- `model_id` (text/select) — only if you realistically switch WD variants.
  - Default = your WD tagger v3
  - Keep under Advanced so beginner doesn’t touch it.

---

### Tier C — Hardcoded Policy (no UI, cannot be overridden)
These must be enforced internally (the new “policy layer”).

**Mandatory for WD family (color correction)**
- Always apply RGB→BGR swap when `_is_wd_family(model_id)` is true.
- Remove any UI/pipeline param related to:
  - `input_color_order`
  - any “color correction strength” or “channel fix” knobs

**Other hardcoded decisions (uncommon / too complex)**
- category override IDs (general/character/rating ids)
- obscure normalizer bridge parameters (unless made a separate tool)
- backend selection toggles if you always run same backend
- AMP toggles unless you move them to “Auto” policy

---

## 1) Backend refactor: add policy layer (future-proof)

### File: `services/offline_tagger.py`
Create an internal policy object/dict (constant) at top-level:
- `TAGGER_POLICY = {...}`

Include:
- `force_wd_bgr_fix = True`
- defaults for min thresholds + threshold_mode
- defaults for write_mode behavior
- safe output formatting defaults
- batch default

Add helper:
- `_is_wd_family(model_id: str) -> bool` (robust substring checks)

Add helper:
- `_effective_opts(form_opts, policy) -> TaggerOptions`
  - parse only Tier A+B fields
  - everything else filled from policy
  - ignore unknown legacy keys gracefully

**Hard requirement**
- if `_is_wd_family(opts.model_id)` and policy.force_wd_bgr_fix:
  - enforce inference swap
  - log “WD color fix ON (RGB→BGR)”

**Backward compatibility**
- If pipeline JSON still contains removed keys:
  - ignore them silently
  - BUT log once: “Ignored deprecated options: input_color_order, ...”

---

## 2) UI refactor: simple form + collapsed advanced

### File: `templates/offline_tagger.html`
Restructure form into:
- Section: “Basic”
- Section: “Advanced” (collapsed by default, using `<details>`)

**Remove completely**
- any color correction fields
- any channel-order dropdown
- any rare/complex knobs (category overrides, etc.)

**Add small hints**
- near `threshold_mode`: “MCUT recommended for WD tagger”
- near `min_general/min_character`: show typical recommended numbers

**Advanced panel behavior**
- default values should be “no effect”:
  - numeric defaults = 0 (means disabled)
  - checkboxes default safe (dedupe/sort ON)

---

## 3) Pipeline UI + defaults
### File: `templates/pipeline.html`
- Update offline_tagger step defaults to include only Tier A + Tier B.
- Remove any removed field references from step editor.
- Keep “Advanced” collapsed the same way (if pipeline editor supports it).
- Ensure pipeline load does not crash when config contains deprecated keys.

---

## 4) Pipeline runner mapping
### File: `services/pipeline.py`
- In offline_tagger step mapping, only pass Tier A+B keys.
- Do not pass any removed or policy keys.
- Ensure legacy keys in step_cfg are ignored (no crash).

---

## 5) Logging requirements (clear + trustable)
When run starts, log a short summary:
- Model: `<model_id>`
- WD color fix: `ON/OFF`
- Threshold mode: `<fixed/mcut>`
- Min thresholds: general / character
- Output mode: overwrite/append/skip
- Advanced enabled summary:
  - “Tag caps: general=X, character=Y” (only if >0)
  - “Character top-k: K” (only if >0)

---

## 6) Quick testing checklist
### Basic (Beginner)
1. Run with defaults on a small folder.
2. Confirm output tags generated, no blue color false positives.
3. Confirm no advanced options touched; results stable.

### Advanced (Enhancement-only)
1. Set `max_general_tags=25` and run.
   - Expect: fewer tags, no errors.
2. Set `character_topk=1`.
   - Expect: only top character retained.

### Legacy config test
1. Load pipeline JSON that includes `input_color_order`.
2. Run pipeline.
   - Expect: no crash, field ignored, log “Ignored deprecated option”.

---

## Definition of Done
- Beginner UI shows only Tier A fields.
- Advanced panel exists but defaults are no-effect enhancements.
- WD color shift fix is permanent (no UI/pipeline override).
- Deprecated fields are ignored safely.
- Logs clearly show applied policy + advanced enhancements when active.

## Extra note for Codex (implementation preference)
When removing options, prefer:
- **ignore legacy keys** over breaking configs
- add a small list `DEPRECATED_KEYS = {...}` and filter them out
- keep the code path “single policy source” so future model changes don’t reintroduce UI complexity
