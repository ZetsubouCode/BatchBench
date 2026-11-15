# BatchBench Implementation Status

Last reviewed against the local working tree on 2026-10-03. Existing unrelated working-tree edits were preserved.

## 2026-10-03 Guided Tagging mobile workspace

- Phone portrait and short landscape layouts use one modal scroll area. Only the contained image preview is sticky; a slim green current-image progress bar (`45/60`) and action rows scroll normally. The preview contracts while a tag input is active.
- Previous/next segment and image actions stay visible in separate full-width rows. The optional segment picker, image zoom controls, and session actions expand inline under More.
- Image filenames and autosave metadata stay available on desktop; mobile save errors appear beside Save & Next. Mobile tagging controls have larger touch targets, and Close stays available in the action row.
- The mobile Tools drawer hides the Tag Editor dock while open and keeps keyboard focus within the drawer.
- Desktop Guided Tagging gives the image more of the modal height. The complete caption stays on one line by default and expands with Show all tags.

## 2026-09-25 UI/workflow implementation

- Home now opens first, with direct Tag Editor and Dataset Health navigation, a collapsible desktop rail, and a searchable mobile drawer. Heavy Tag Editor, Workflow, Settings, and Glossary panes are deferred until opened. Bootstrap and icons are vendored locally.
- Shared path context separates project, dataset, source, output, and working roles. Tag Editor may derive an existing `<project>/dataset`; unrelated source/output inputs are not changed by project selection.
- Auto Tag Assist generates reviewable proposals before writing. Selected proposals use one-time tokens and source fingerprints; the packaged external Python worker can return proposals. Existing caption backup behavior remains on Apply.
- Dataset Normalization Apply is tied to a one-time fingerprinted preview and rejects changed options/files. Preview reports all scanned captions, changed/unchanged totals, and changed-only examples.
- Dataset Health reuses Workflow audit checks and supports guarded caption inspection/editing with paired image preview and stale-file detection.
- Guided Tagging is the first Tag Editor action, shows complete caption tags, distinguishes session autosave from `.txt` writes, and keeps Save & Next reachable in the mobile footer. Selected pair moves show scope and validate a fresh preview when used from the UI.
- Workflow keeps a compact normal path and moves uncommon rule controls under Advanced. Manual review carries project context back to Tag Editor and exposes Return to Workflow.
- Combine Dataset uses a source folder list. Move mode requires a fresh pair-count and collision preview. Webtoon Splitter, Stitch Groups, and Flatten & Renumber now put tuning controls under Advanced. Image → PNG avoids overwriting existing output names and reports collision renames.
- CLIP Token Check groups over-limit, near-limit, and longest captions with links into Dataset Health. Settings shows Guided Flow unsaved state and warns on navigation.
- A shared operation result component renders structured counts/issues/output before technical logs for normal form tools, while legacy service handlers continue to work.

Current scope: some specialized batch tools still use their existing form/result layouts, and legacy direct Tag Editor move APIs remain callable for compatibility. These are not yet fully migrated to the new preview/result pattern.

This document describes the repository as it exists now and separates implemented behavior from intended direction.

## Current Product State

BatchBench already functions as a broad local dataset-preparation application rather than a small collection of disconnected scripts.

The current navigation groups tools into image preparation, dataset assembly, tagging, workflow, and reference/settings areas.

## Implemented

### Tag Editor corrective desktop refinement (2026-11-15)

- Restored one main workspace; removed the unsuccessful launcher-only Tag Images / Manage Dataset split.
- Guided Tagging retains the original neo panel hierarchy and category colors, visible recommendations and glossary, direct segment access, grouped manual entry, and persistent image inspection.
- Filename and caption/selection filters operate on loaded images. Browser cards use compact caption summaries and keep full preview access. One contextual staging action replaces duplicated controls.
- Batch edits expose inspected/changed/unchanged/skipped/failed and backup counts; file-operation results remain visible after refresh. Folder maintenance is secondary to staging.
- Draft writes are serialized; close/unload guards preserve pending work. Save & Next prevents repeated writes and reports caption success separately from session-save failure.
- Segment navigation preserves pending manual text in its original segment. Image shortcuts no longer intercept typing in free mode.
- Selection updates preserve gallery thumbnails and focus. Caption writes defer dataset-wide gallery/tag refresh until the editor closes.
- Existing pair previews, conflict handling, backups, file return, and pipeline review wiring remain. See [Tag Editor workflow and verification](TAG_EDITOR_REDESIGN.md).

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
- preservation of missing historical session items without assigning them to a different image,
- Dataset Tag Editor-only mobile layout and touch controls for Guided Tagging, Dataset Browser, Bulk Tag CRUD, and Dataset Workspace, including a compact sticky Guided Tagging image preview, scrolling Save & Next action, keyboard-safe tag inputs/autosuggest, and touch-native Tagging Flow mapping controls,
- configurable Project Root browse base with remembered-project precedence,
- safe browser download of generated dataset ZIP files for remote/mobile use.

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

## Tag Editor further refinement (task 002, 2026-11-15)

Implemented connected sequential step navigation; optional split browsers with saved
visibility and keyboard/pointer resizing; a staged browser with search, sort, folder
filter, selection, preview, incremental loading, and paired return actions; full-panel
internal drops; staged-only Tag Explorer with incremental chips and explicit info
actions; manual-input-safe auto-sync; full-width composer with read-only preview;
separate dataset maintenance and export sections; and state-aware LED status labels.
The shared pair move service now preserves nested staging paths and handles
case-insensitive image/caption collisions and rollback. Explicit dataset flattening
remains separate. Existing uncommitted draft-save/reporting changes were retained.

Task-002 validation: 80 focused Python tests and 10 subtests passed. The Chromium workflow suite passed at 1920, 1440, 1024, 768, and 375px. See `docs/TAG_EDITOR_FURTHER_REDESIGN.md` for scope, files, screenshots, and limits.
