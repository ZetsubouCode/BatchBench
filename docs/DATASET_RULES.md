# Dataset Rules and Invariants

This document defines the dataset behaviors that future BatchBench changes should preserve.

## 1. Canonical Caption Format

Final sidecar captions are comma-separated tag lists.

Canonical written form:

```text
triggerword, long hair, blue eyes, white shirt, looking at viewer
```

Rules:

- trim leading/trailing whitespace,
- normalize repeated whitespace inside a tag,
- keep one logical tag per comma-separated item,
- avoid duplicate logical tags unless a feature explicitly models duplicates before final serialization,
- write human-facing caption tags with **spaces rather than underscores**,
- preserve meaningful tag order when order has semantic/workflow value,
- protect an explicitly configured trigger tag from automatic removal.

### Danbooru Boundary

BatchBench uses Danbooru-derived vocabulary, but Danbooru APIs/catalogs commonly use underscore identifiers such as `long_hair`.

Internal lookup/matching code may use underscore-form identifiers when necessary. The conversion boundary must be explicit:

```text
Danbooru/catalog form: long_hair
BatchBench caption form: long hair
```

Do not solve the new caption requirement by blindly replacing every underscore in every internal identifier. Session IDs, configuration keys, API identifiers, policy identifiers, file names, and catalog records may legitimately require underscores.

## 2. Image and Caption Pairing

For a captioned training dataset, an image and caption pair share the same stem:

```text
0001.png
0001.txt
```

Pair-sensitive operations must treat the image and `.txt` file as one logical unit whenever a caption exists or is required.

For move/rename/delete/copy operations:

- update both members of the pair when applicable,
- detect missing captions,
- avoid silently overwriting an unrelated pair,
- use deterministic conflict handling,
- report skipped/conflicting/error items in the log,
- roll back partial pair moves when practical.

A final dataset audit should clearly report orphaned images, orphaned captions, and missing sidecars.

## 3. Dataset Tag Editor Project Layout

The following layout is canonical **for the Dataset Tag Editor project workflow**, not necessarily for every BatchBench tool:

```text
<project root>/
├─ database/
├─ dataset/
│  └─ _temp/
└─ prompt.txt
```

Meaning:

- `database/` is the retained source/reference image area created by Tag Editor initialization.
- `dataset/` is the editable working dataset.
- `dataset/_temp/` holds Tag Editor staging/session data and must not be treated as final training output.
- `prompt.txt` is the project trigger/cheat-sheet source used by tagging flows.

Other independent tools may accept ordinary folders and do not need to force this layout.

## 4. Initialization Safety Model

Tag Editor initialization currently creates dataset safety through separation:

1. root source images are copied into `dataset/`,
2. missing sidecar captions are created,
3. the original root images are moved into `database/` after the dataset copy succeeds.

The intended model is therefore:

```text
source/reference -> database/
editable copy    -> dataset/
```

Normal caption editing should operate on `dataset/`, not mutate `database/` as part of routine tagging.

Do not introduce a second universal recovery mechanism unless there is a concrete failure mode that the existing copy/staging model cannot cover.

## 5. Destructive Operation Policy

A global `.bak` requirement is **not** a BatchBench invariant.

For new or changed destructive operations, prefer the lightest safety mechanism that fits the operation:

1. separate output/copy,
2. isolated workflow working directory,
3. staging/temp area,
4. preview/dry-run before execution,
5. transactional rollback for multi-file operations.

Existing feature-specific `.bak` behavior may remain when it already serves a useful overwrite workflow. Do not automatically spread that pattern to unrelated tools.

When an overwrite option exists, the UI must make it explicit.

## 6. Manual Tagging Authority

Automated systems may provide:

- tag suggestions,
- model predictions,
- policy filtering,
- glossary recommendations,
- normalization previews.

They must not be treated as inherently more correct than manual review.

Auto Tag Assist should be designed as assistance. Preview and transparent filtering information are more important than maximizing automatic write volume.

## 7. Session and Temporary Data

Guided Tagging Flow session data lives under `dataset/_temp/` and should support safe resume across app restarts and reasonable dataset changes.

Current-session decisions should not be discarded merely because images were added later. Missing historical items may remain as history but must not block completion of the current dataset.

Temporary/session files must be excluded from final training exports.

## 8. Normalization Rules

Normalization should be explicit and inspectable.

A normalization operation should report at least:

- files scanned,
- files changed,
- tags added/removed/replaced when practical,
- missing caption files,
- skipped files,
- errors.

Do not silently reinterpret large classes of tags without a documented rule or preset.

The canonical caption-space migration must be applied consistently at caption read/write boundaries rather than through unsafe global string replacement.

## 9. Final Export Integrity

Final export should prefer clean training artifacts only.

Unless explicitly requested, final output should not contain:

- `_temp` session files,
- pipeline job state,
- runtime caches,
- patch/backup files,
- unrelated project metadata.

For captioned workflows, final audit should be able to confirm:

- expected image count,
- expected `.txt` count,
- image/caption pairing,
- no accidental empty captions when captions are required,
- no obvious duplicate tag serialization,
- token warnings when token checking is enabled.

## 10. Logging Contract

Batch operations must produce useful logs.

A good result log answers:

- What did the tool scan?
- What did it change?
- How many files were affected?
- What was skipped?
- Were files renamed due to conflicts?
- What failed and why?
- Where is the output?

Prefer summary counts plus actionable per-file errors over noisy success output for every trivial operation.
