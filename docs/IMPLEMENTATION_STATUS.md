# BatchBench Implementation Status

Last reviewed against the `main` working tree on 2026-09-12 (base commit `ec7bb875faa35c9bb3cc89551d508ec108531fb7`).

This document describes the repository as it exists now and separates implemented behavior from intended direction.

## Current Product State

BatchBench already functions as a broad local dataset-preparation application rather than a small collection of disconnected scripts.

The current navigation groups tools into image preparation, dataset assembly, tagging, workflow, and reference/settings areas.

## Implemented

### Core local application

- Flask local web application.
- Shared Jinja/Bootstrap UI shell.
- Tab-based tool navigation.
- Shared local folder selection/path-oriented workflow.
- Runtime logs/status for batch operations.

### Dataset Tag Editor

- project initialization around `database/`, `dataset/`, `dataset/_temp/`, and `prompt.txt`,
- source image copy/move preparation,
- automatic creation of missing sidecar captions during initialization,
- manual caption editing,
- bulk tag operations,
- Guided Tagging Flow,
- prompt/cheat-sheet integration,
- local suggestion/glossary integration,
- persistent tagging session state,
- session reconciliation when images or guided segments change,
- preservation of missing historical session items without assigning them to a different image.

### Tag reference and suggestion system

- local Danbooru tag catalog,
- CSV/SQLite catalog data,
- local autocomplete/suggestion usage during typing,
- explicit catalog synchronization/import behavior,
- glossary/wiki support,
- Danbooru reference lookup.

### Auto Tag Assist

- curated CAFormer S36 dbv4 and WD SwinV2 v3 model profiles,
- explicit AnimeTimm RGB and SmilingWolf BGR preprocessing adapters,
- CAFormer ONNX Runtime CPU inference and per-tag optimized thresholds,
- writable-data model manager with background download and local-folder installation,
- reusable Jio7 semantic classification/filtering,
- fixed/MCUT threshold behavior,
- semantic tag policy support,
- character identity omission policy,
- custom keep/block rules,
- preview/review logging,
- replacement behavior,
- policy leak auditing,
- color-sanity filtering,
- local/cached model paths where configured.

### Guided Flow Context Suggestions

- explicit **Inspect Dataset** background job with progress and cancellation,
- CAFormer recommended by default while WD remains selectable,
- compact segment-routed Detected chips that require explicit user selection,
- Jio7 defaults that keep action, expression, object, setting, and other while hiding appearance/outfit classes,
- persistent resumable cache under `dataset/_temp/tag_suggestions/`,
- cache invalidation by image fingerprint, model/revision, sensitivity/threshold strategy, classifier version, and suggestion-policy version,
- inspection does not modify caption `.txt` files.

### Dataset normalization and review

- normalization presets/rules,
- tag modification workflows,
- CLIP token checking,
- pair/audit behavior in the dataset workflow.

### Dataset Workflow

Workflow presets currently cover concepts such as:

- raw images -> tagged dataset,
- existing captioned dataset -> final export,
- image preparation only.

The pipeline supports isolated working copies, reorderable/compiled steps, manual-review pauses, resume/status state, audit, and final export.

### Image and dataset utilities

The repository includes tools such as:

- Image -> PNG conversion,
- Photo Adjust,
- Brush Blur,
- Color Brush,
- Manga Palette Helper,
- EPUB extraction,
- Webtoon Panel Splitter,
- Stitch Groups,
- Flatten & Renumber,
- Combine Dataset.

### Windows runtime

- Windows launcher/package support,
- packaged executable build scripts,
- user-data/resource path separation,
- system tray integration,
- shared writable data-root support through `BATCHBENCH_DATA_DIR`.

### Optional integrations

- Discord Rich Presence is optional and designed not to expose dataset contents, tags, prompts, paths, trigger words, or filenames.

### Tests

The repository has meaningful regression coverage across services and APIs, including areas such as:

- tag editor initialization/transactions,
- guided session reconciliation,
- Auto Tag Assist,
- Danbooru APIs,
- runtime paths,
- folder picking,
- image operations,
- webtoon splitting,
- Color Brush,
- Discord presence,
- workflow UI/guide behavior.

## Caption Representation Migration

### Space-separated caption boundary implemented

BatchBench writes caption tags such as:

```text
long hair
```

instead of:

```text
long_hair
```

The implementation now distinguishes between:

- Danbooru/catalog/query identifiers (`long_hair`), and
- BatchBench caption/display form (`long hair`).

Covered surfaces include:

- caption parsing/serialization helpers,
- Dataset Tag Editor reads/writes,
- Guided Tagging Flow selections,
- normalization,
- Auto Tag Assist final writes,
- trigger/protected-tag comparison,
- prompt/cheat-sheet generation,
- glossary/suggestion insertion,
- pipeline audit/export,
- README/examples,
- regression tests.

Legacy underscore captions and saved session values remain accepted and are normalized in memory. Files are rewritten only by an explicit save, tagging operation, normalizer apply, or final export. Configured trigger literals remain exact.

## Known Technical Debt

### Large application entry point

`app.py` contains substantial route/API wiring and has continued to grow.

Direction: keep it working, but move new cohesive behavior into `services/` instead of expanding the monolith when practical.

### Large Dataset Tag Editor template

`templates/dataset_labeling.html` is a very large interactive surface.

Direction: extract cohesive UI/JavaScript pieces when a touched feature would otherwise make the file materially harder to maintain. Do not rewrite the frontend wholesale.

### Mixed safety mechanisms

The repository currently includes working-copy/staging behavior plus some feature-specific `.bak` options.

Direction: do not standardize on `.bak`. Prefer isolation/output separation for new work while retaining existing backups where removing them would reduce current safety.

## Platform Status

- Windows: primary target; packaged runtime exists.
- Linux: intended to remain runnable, but not the primary test target.
- macOS: no hard compatibility promise.

## Offline Status

Most manual/local workflows can operate offline.

Internet may still be required for:

- first-time remote model download,
- Danbooru catalog synchronization,
- uncached Danbooru wiki/reference fetches,
- Discord Rich Presence.

Normal local tagging and local catalog usage should not become network-dependent.

## Recommended Next Engineering Sequence

1. Implement the caption-space representation boundary and migrate affected tag read/write paths.
2. Add focused regression tests for underscore catalog form vs space caption form.
3. Update README/examples after behavior is correct.
4. Continue incremental extraction from `app.py` / Dataset Tag Editor only when touched by real feature work.
5. Keep improving pair-integrity/audit feedback where real workflow failures reveal gaps.

Avoid pausing product work for a broad repository rewrite.
