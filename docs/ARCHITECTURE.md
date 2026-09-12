# BatchBench Architecture

## 1. Architecture Summary

BatchBench is a local Flask application with server-rendered Jinja templates and browser-side JavaScript for interactive tools.

The current architecture is intentionally lightweight:

```text
Browser
  -> Flask routes / JSON APIs (app.py)
      -> service modules (services/)
          -> shared utilities (utils/)
          -> local files / project folders / caches / presets
```

Windows packaging adds a launcher/system-tray layer but does not change the core application model.

## 2. Main Code Areas

### `app.py`

Application entry point and route wiring.

Responsibilities should trend toward:

- request parsing,
- validation at the HTTP boundary,
- selecting/orchestrating service functions,
- returning templates or JSON responses,
- application-level integration wiring.

Do not keep adding large blocks of reusable business logic directly to `app.py` when the logic can live in a cohesive service module.

### `services/`

Own tool/domain behavior.

Examples include:

- tag editing,
- normalization,
- Auto Tag Assist,
- dataset workflow/pipeline,
- Danbooru/catalog logic,
- image transforms,
- dataset assembly,
- Discord presence,
- packaged runtime path handling.

For ordinary form-based tools, use the existing `services/registry.py` pattern when it fits.

### `services/registry.py`

Registry for straightforward tool handlers submitted through the shared tool flow.

A normal new tab-based tool usually needs:

1. `services/<tool>.py`,
2. import + registration in `services/registry.py`,
3. `templates/<tool>.html`,
4. tab/menu wiring in `templates/index.html`,
5. focused tests when the behavior is non-trivial.

Do not force complex stateful/API subsystems through the registry when that makes the design worse.

### `utils/`

Small reusable helpers that are not specific to one tool.

Examples include:

- image operations,
- parsing,
- text I/O,
- dataset helpers,
- common tool-result structures.

Do not create generic abstractions until at least two real call sites benefit from them.

### `templates/`

Jinja templates for the shared shell and individual tabs/tools.

`templates/index.html` currently composes the top-level tab experience from per-tool templates.

Large interactive templates may contain substantial JavaScript today. When a file becomes difficult to reason about, extract coherent JavaScript/template pieces incrementally rather than rewriting the frontend framework.

### `presets/`, `settings/`, `_config/`, `data/`

Local configuration, presets, user-editable settings, and catalog/cache data.

These are not interchangeable:

- presets describe reusable operation settings,
- settings/config describe application behavior,
- `data/` may contain mutable local catalog/reference data,
- user datasets remain outside application source folders unless a tool explicitly creates local work state.

### `_work/`

Runtime work area for transient application state such as pipeline jobs.

Pipeline state currently uses paths such as:

```text
_work/pipeline_jobs/<job_id>/state.json
```

Do not treat `_work/` as durable user dataset output.

## 3. Dataset Tag Editor Architecture

Dataset Tag Editor is the central workflow area and has its own project convention:

```text
project root/
├─ database/
├─ dataset/
│  └─ _temp/
└─ prompt.txt
```

This layout is scoped to the Tag Editor project workflow.

Guided Tagging Flow session state is stored under `dataset/_temp/`. Current reconciliation logic preserves progress when dataset contents change and tracks missing historical items rather than transferring decisions to unrelated files.

See `docs/DATASET_RULES.md`.

## 4. Dataset Workflow / Pipeline

The workflow layer builds repeatable preparation plans such as:

- raw images -> tagged dataset,
- existing captioned dataset -> final export,
- image preparation only.

The workflow should operate through an isolated working copy where possible before mutating files.

The pipeline is a BatchBench-specific dataset workflow system, not a general workflow engine. Keep workflow steps grounded in existing dataset preparation operations.

## 5. Tag Representation Boundary

Current code historically normalizes many tags to underscore identifiers because Danbooru uses names such as `long_hair`.

The product requirement has changed: final/user-facing caption tags must use spaces (`long hair`).

Architecture should therefore distinguish:

- **catalog/policy/query form** when an underscore identifier is required,
- **caption/display form** for BatchBench `.txt` output.

Create explicit conversion helpers and reuse them. Do not globally replace underscores because internal IDs, session keys, policy names, configuration keys, and Danbooru API values may require them.

The shared boundary is implemented in `utils/tags.py`. Caption-facing services normalize legacy and new input to spaces, while catalog, API, policy, regex, and identifier paths retain underscore keys where required.

## 6. Local-First and Network Boundaries

Core manual dataset work should not require the network.

Network-dependent features currently include or may include:

- Danbooru catalog synchronization,
- first-time Danbooru wiki/reference fetches,
- Hugging Face model downloads when a requested model is not local/cached,
- Discord Rich Presence.

Rules:

- make these features optional,
- prefer local cache/catalog reads during normal interaction,
- fail clearly when offline rather than breaking unrelated features,
- never require Discord for application startup,
- allow local model paths/caches where supported.

## 7. Writable Data and Packaged Windows Builds

`services/paths.py` separates packaged resources from mutable user data.

`BATCHBENCH_DATA_DIR` can point source and packaged launches at the same writable BatchBench data root.

When adding mutable application data, use the established user-data path abstraction rather than assuming files beside `__file__` are always writable in frozen builds.

## 8. Platform Strategy

Windows is the primary tested target.

Linux compatibility should be preserved when the implementation cost is reasonable.

Prefer `pathlib.Path` and portable Python behavior in core services. Windows-specific integrations should remain isolated behind optional paths/modules.

Do not claim macOS compatibility as a hard requirement without testing it.

## 9. Dependency Strategy

Keep setup reasonably simple.

Before adding a dependency, ask:

1. Does it solve a real product problem?
2. Is the benefit significant compared with a small local implementation?
3. Does it add large binaries, services, or installation complexity?
4. Does it degrade offline use?
5. Does it complicate the packaged Windows build?

Heavy ML dependencies are already justified by Auto Tag Assist. Do not use that as a reason to add unrelated heavy dependencies casually.

## 10. Refactoring Policy

Do not perform a repository-wide "clean architecture" rewrite merely for purity.

Known large surfaces include `app.py` and `templates/dataset_labeling.html`. They are legitimate technical debt, but changes should use **extraction when touched**:

- if a new feature would add another large cohesive block to `app.py`, put the block in a service/module,
- if a tagging UI change would make an already-large script materially harder to maintain, extract the relevant cohesive JavaScript/template fragment,
- preserve behavior while extracting,
- avoid unrelated rewrites in the same task.

## 11. Error Handling and Logging

Service operations should return enough structured information for the UI to show:

- success/failure,
- summary counts,
- warnings,
- actionable errors,
- output paths.

Do not swallow per-file errors in batch operations.

For multi-file pair operations, prefer transaction-like behavior or explicit partial-failure reporting.

## 12. Testing Strategy

Tests should protect important behavior without turning normal changes into long-running CI-style work.

Prefer:

- focused unit tests for parsing/policies/path rules,
- service tests using temporary directories,
- route/API tests for important contracts,
- regression tests for previously broken behavior.

Avoid by default:

- downloading ML models,
- live Danbooru/Hugging Face requests,
- tests requiring external accounts/services,
- time-consuming end-to-end model inference,
- huge fixture datasets.

Use mocks, small fixtures, temporary files, or already-local test doubles instead.

## 13. Curated Tagger and Context Suggestion Architecture

Tagger selection is defined in `services/tagger_models/registry.py`. A profile chooses an explicit adapter; inference code does not infer preprocessing from model-name substrings.

Initial profiles are:

- `caformer_s36_dbv4`: AnimeTimm dbv4 adapter, ONNX Runtime CPU execution, RGB/white-pad/384×384/ImageNet preprocessing, and per-tag `best_threshold` metadata.
- `wd_swinv2_v3`: SmilingWolf adapter, Timm/PyTorch runtime, WD-specific BGR handling, and legacy MCUT/fixed threshold behavior.

Curated model files live in writable user data under `models/taggers/<profile>/`. Download and local-folder installation validate exact profile requirements and never mutate the source folder.

`services/jio7_tags.py` supplies a cached, versioned semantic classification layer shared by Auto Tag Assist and Context Suggestions. Catalog/model identifiers remain underscore-form internally and convert to caption-space form only at the display/write boundary.

`services/context_suggestions.py` runs sequential background inspection and stores resumable JSON cache entries under `dataset/_temp/tag_suggestions/`. Cache identity includes the model profile/revision, threshold strategy and sensitivity, Jio7 version, suggestion-policy version, and image size/mtime fingerprint. Inspection never writes caption sidecars; only an explicit chip selection enters Guided Flow state.
