# BatchBench Product Requirements

## Product Summary

BatchBench is a local-first dataset preparation workspace for image + sidecar caption datasets. Its primary purpose is to make **manual dataset tagging structured, repeatable, and fast enough to use for real LoRA preparation work** without constantly switching between unrelated tools.

The product is built primarily for the repository owner and their personal workflow. The repository is public so other people may benefit from it, but public/general usability must not override the owner's workflow or force a large redesign.

## Product North Star

Make manual dataset preparation feel like one coherent environment:

`collect/clean images -> initialize dataset -> manually tag/review -> normalize/audit -> export`

Tagging is the core concern. Image cleanup, dataset assembly, automated suggestions, token checks, glossary lookup, and workflow automation exist to support that core flow.

## Primary User

The primary user is a technically comfortable creator preparing datasets for LoRA training, primarily for Illustrious/CivitAI workflows.

The application should still be understandable to other users when that can be achieved with small UX improvements, clear labels, sensible defaults, and concise help text. Do not generalize the product at the cost of the owner's established workflow.

## Core Product Principles

### 1. Manual tagging is authoritative

BatchBench should help a human tag a dataset, not replace human judgment.

Auto Tag Assist, Danbooru suggestions, tag policies, glossary data, and other automated features are support systems. They may propose, filter, preview, or prefill tags, but the manual workflow remains the product's main source of truth.

### 2. One workspace for dataset preparation

If a small image or dataset utility is repeatedly needed during preparation, it can belong in BatchBench when adding it avoids unnecessary app switching and does not make the product difficult to maintain.

Examples include:

- image conversion and cleanup,
- brush-based corrections,
- webtoon/EPUB extraction,
- stitching and renumbering,
- dataset combination,
- caption normalization,
- pair integrity checks,
- token-length review.

### 3. Offline-friendly by default

The majority of the application must remain useful without internet access.

Network-dependent features must be optional and clearly scoped. Local caches/catalogs should be preferred where practical.

### 4. Safe dataset manipulation

BatchBench should make it difficult to accidentally destroy the only useful copy of a dataset.

Safety should come primarily from isolated working copies, separate outputs, staging, preview/dry-run behavior, and transactional pair operations rather than a universal `.bak` policy.

### 5. Simple local operation

Fresh installation should remain straightforward. Flask and the current local web application model are acceptable defaults.

A new framework, service, database server, worker system, or other infrastructure is justified only when it provides a clear practical benefit that outweighs setup and maintenance cost.

## Canonical Dataset Output

The main dataset target is:

- image files,
- matching sidecar `.txt` files with the same stem,
- comma-separated Danbooru-derived tags,
- **spaces in human-facing/written caption tags instead of underscores**.

Example canonical caption:

```text
triggerword, long hair, blue eyes, white shirt, looking at viewer
```

Not:

```text
triggerword, long_hair, blue_eyes, white_shirt, looking_at_viewer
```

Danbooru/API/catalog internals may retain underscore-form identifiers where required for matching or remote lookup. Caption serialization is a separate boundary and should follow BatchBench's space-separated tag representation.

See `docs/DATASET_RULES.md` for the detailed invariants.

## Product Areas

### Dataset Tagging

This is the central product area.

It includes:

- Dataset Tag Editor,
- Guided Tagging Flow,
- project initialization,
- manual caption editing,
- reusable prompt/cheat-sheet guidance,
- tag insert/delete/replace/dedup/move operations,
- session resume/reconciliation,
- tag suggestions and glossary lookup,
- dataset normalization and review.

### Dataset Preparation Utilities

Supporting tools may include:

- image format conversion,
- photo adjustment,
- blur/color brush editing,
- manga palette helpers,
- EPUB extraction,
- webtoon splitting,
- stitching,
- flattening/renumbering,
- dataset combining.

These remain secondary to tagging but are valid product features because they keep preparation work inside one environment.

### Assisted Tagging

Auto Tag Assist is a support tool, not the product's tagging authority.

It should favor:

- preview before writes,
- policy-controlled suggestions/results,
- transparent kept/dropped tag logs,
- local model use when available,
- explicit user review.

### Dataset Workflow

The workflow/pipeline layer exists to connect repeatable preparation steps and isolate working data from the original source.

It should not become a general workflow automation platform.

### Reference Tools

Danbooru catalogs, glossary/wiki data, CLIP token checks, and similar reference features should reduce manual lookup work while remaining optional to the core offline workflow.

## Platform Scope

- **Primary platform:** Windows.
- **Secondary compatibility target:** Linux should continue to run when reasonably possible.
- **macOS:** best effort only; it is not a release requirement unless explicitly tested later.

Windows-specific conveniences such as the packaged executable and system tray are allowed. Core service logic should still avoid unnecessary Windows-only assumptions when a portable implementation is simple.

## UX Direction

BatchBench is a utility workspace, not a marketing website.

The UI should be practical, compact, readable, and consistent with the current Bootstrap-based application.

Avoid generic AI-generated visual patterns such as:

- excessive gradients,
- glowing decorative backgrounds,
- marketing-style hero sections,
- small eyebrow text above large headings without a real need,
- excessive cards/boxes,
- cards nested inside cards for simple content,
- decorative elements that reduce information density.

See `docs/UI_UX.md` for detailed rules.

## Non-Goals

BatchBench is not intended to become:

- a hosted SaaS product,
- a multi-user collaboration platform,
- a training orchestration platform,
- a replacement for LoRA training services,
- a fully automatic caption-generation system,
- a generic image editor,
- a general-purpose workflow engine,
- a database-heavy enterprise application.

## Success Criteria

A feature is successful when it materially reduces friction in the real dataset preparation workflow.

For the core dataset flow, completion means the user can:

1. start from source images,
2. work on an isolated editable dataset,
3. tag/review images manually without losing progress,
4. use optional assistance without surrendering control,
5. verify image/caption integrity,
6. see clear logs of what changed,
7. export a clean dataset with canonical captions.

## Documentation Boundary

`README.md` remains the user-facing usage guide.

The files under `docs/` define development direction, architectural boundaries, dataset invariants, UX rules, decisions, and implementation status. When the current implementation or README conflicts with a newer documented product decision, the mismatch should be treated as implementation debt rather than silently changing the product decision.
