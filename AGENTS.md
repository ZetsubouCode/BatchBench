# AGENTS.md

## Scope and source of truth

These instructions apply to the whole BatchBench repository unless a deeper `AGENTS.md` overrides them.

Read, in order when relevant:

1. `docs/DECISIONS.md`
2. `docs/PRODUCT.md`
3. `docs/DATASET_RULES.md`
4. `docs/ARCHITECTURE.md`
5. `docs/UI_UX.md`
6. `docs/IMPLEMENTATION_STATUS.md`
7. `README.md` for user-facing usage

If current code/README conflicts with a newer accepted decision, treat it as implementation debt rather than silently reverting the decision.

## Product north star

BatchBench is a local-first dataset preparation app centered on **structured manual tagging**. Image cleanup, dataset assembly, Auto Tag Assist, audit, reference lookup, and workflow automation support that core task.

Hard rules:

- Manual tagging remains authoritative; automation assists.
- Final `.txt` captions are comma-separated tags using spaces: `long hair`, not `long_hair`.
- Danbooru/catalog/API internals may keep underscore identifiers when needed. Convert explicitly at boundaries; never globally replace underscores.
- Treat image + `.txt` as one logical pair when captions are required.
- `database/`, `dataset/`, `dataset/_temp/`, `prompt.txt` is the Dataset Tag Editor project layout, not a requirement for every tool.
- Routine tagging edits belong in `dataset/`; do not casually mutate `database/` source/reference files.
- Core work must remain offline-friendly. Network features are optional.
- Windows is primary; keep Linux compatibility when straightforward.

## Data safety

Do not make `.bak` the default safety pattern for new features. Prefer the lightest suitable mechanism: separate output/working copy, staging/temp, preview/dry-run, or transaction-like rollback.

Existing feature-specific backups may stay unless the task changes them.

For pair operations, handle image/sidecar together when applicable, detect missing sidecars, avoid silent overwrite, use deterministic conflict handling, and report partial failures.

## Architecture

Keep Flask + Jinja + Bootstrap/vanilla JS as the default stack unless another technology has a clear practical payoff.

For a normal new tab/tool, prefer:

- `services/<tool>.py`
- `services/registry.py` when the shared handler model fits
- `templates/<tool>.html`
- `templates/index.html` wiring
- focused tests for non-trivial behavior

Routes should validate/orchestrate. Cohesive domain behavior belongs in `services/`; small genuinely reusable helpers belong in `utils/`.

`app.py` and `templates/dataset_labeling.html` are already large. Do not start a broad cleanup rewrite, but extract cohesive touched logic when a task would materially worsen them.

For packaged builds, use the existing resource/user-data path abstractions rather than assuming source-relative paths are writable.

## UI / UX

BatchBench is a practical desktop utility, not a marketing page.

Use the existing Bootstrap language. Avoid gratuitous gradients/glows, marketing heroes, eyebrow text without purpose, excessive cards/boxes, nested cards for simple content, and decorative whitespace that pushes controls/logs away.

Prefer compact sections, obvious primary actions, useful defaults, concise tooltips, keyboard-friendly tagging, visible progress, and clear logs. Make destructive targets/overwrite behavior explicit.

## Logging

Batch operations should expose, when relevant: scanned, changed, skipped, conflict/rename, warning/error counts, output path, and actionable per-file errors. Never hide errors just to produce a clean success state.

## Offline and dependencies

Do not introduce live-network requirements into manual tagging or local dataset browsing. Danbooru sync/wiki fetch, remote Hugging Face downloads, and Discord presence must fail independently without breaking core local use.

Prefer local cache/catalog/model paths when available.

Before adding a dependency, consider fresh Windows setup, packaged EXE impact, offline use, binary size, and maintenance cost. Do not add React/Vue, a database server, Redis, Celery, or similar infrastructure by default.

## Testing

Testing is targeted, not ceremonial. Add/run focused tests for changed behavior when useful.

Do not create or run tests that normally download models, require live internet/accounts, perform long real-model inference, or process huge datasets. Use temp directories, mocks, small fixtures, and local doubles.

A full `pytest` run is not mandatory for every task. Always report what was actually tested.

## Documentation

Update docs when a change alters an invariant, accepted decision, workflow, architecture boundary, or implementation status. Keep README as the user manual rather than duplicating it in `docs/`.

## Git workflow

Default branch: `main`.

Before editing/committing, inspect `git status`. Preserve unrelated user changes; never reset/clean/stash/drop them unless explicitly asked.

If asked to commit/push:

- use the repository's existing local Git account/identity,
- do not change `user.name`/`user.email`, invent an author, or add artificial `Co-authored-by` lines,
- commit only task-related changes,
- push to `main` unless another branch is explicitly requested.

Never assume a dirty working tree belongs to the current task.

## Execution and done criteria

For non-trivial work: inspect current behavior -> make the smallest coherent change -> preserve unaffected workflow -> implement service/backend logic -> wire UI as needed -> run focused tests -> verify logs/error paths -> update affected docs.

Do not broaden the task into unrelated refactors.

A feature is done when it works from the intended UI/workflow, clearly reports what changed, and produces a safe/clean dataset result without unnecessary setup.
