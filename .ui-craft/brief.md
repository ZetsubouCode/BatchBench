## 1. Product purpose

BatchBench prepares local image/caption datasets through authoritative structured manual tagging and safe file operations.

## 2. Primary user

A desktop dataset editor tagging hundreds of images in repeated sessions, primarily on Windows.

## 3. Principles

1. Preserve the original neo Guided Tagging panel structure, category context, and direct-access controls when refining the UI.
2. Compact means useful information and actions grouped closely; it does not mean empty pages or hidden primary tools.
3. Keep the launcher, browser, staging workspace, and batch actions together on one main Tag Editor page.
4. Distinguish session drafts from caption writes, and preserve image/caption pairs and explicit operation targets.
5. Remove repeated metadata and redundant actions before reducing useful functionality.

## 4. Success metric for the surface

Users can repeatedly choose recommendations, enter/remove tags, revisit resolved segments, advance through unresolved steps, save captions, and continue while the image stays visible. Filename/caption filters, multi-selection, staging, and batch results work without a second workspace tab or a scan for each filter change.

## 5. Out of scope

Global visual identity changes, frontend framework changes, automatic tag application, directory-semantic changes, and replacing accepted caption-format rules.

## 6. Learned constraints

- 2026-11-15, corrective brief: the user rejected the Tag Images / Manage Dataset split and the nearly empty launcher page. Keep one coherent workspace; no replacement tabs.
- 2026-11-15, corrective brief: the user prefers the information-rich Guided Tagging design in `0805926`, also present at `1c4afc6`. Retain recognizable panels and access to completed segments. Do not flatten them into an ungrouped form or hide the glossary to create whitespace.
- 2026-11-15, corrective brief: manual entry and its Add/Clear actions belong together. Recommendation source metadata should be contextual rather than repeated on every chip.

- 2026-11-15, task 002 supersedes free segment navigation: use connected numbered steps, lock future unresolved steps, and preserve choices when revisiting completed steps. Keep optional split browsers, staged-only batch edits, dense explorer chips, and export below the workspace.
