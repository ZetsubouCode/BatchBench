# Tag Editor corrective refinement

## Baseline and scope

The preferred visual baseline is `0805926` (Guided Tagging mobile workspace and
navigation). `1c4afc6` retains that UI and changes glossary metadata only. The first
redesign was uncommitted when this corrective pass began. Its template and CSS/JS
were copied under `_work/tag-editor-corrective/` before editing; Git was not reset.

The correction restores the original neo information-rich panel character and one
main workspace. The launcher-only Tag Images page and Manage Dataset split were
removed. The durable user constraints are in `.ui-craft/brief.md`.

| Rejected iteration | Corrected behavior | Preserved original strength |
| --- | --- | --- |
| Separate launcher and management contexts | Launcher, browser, staging, batch operations, and export on one page | Shared project/workflow context |
| Flattened selected/recommendation/manual areas | Distinct panels, category colors, visible selected tags | Recognizable neo hierarchy |
| Glossary behind disclosure | Glossary and its controls directly accessible | Rich vocabulary access |
| Manual Add/Clear detached from the input | Actions adjacent to the textarea | Repeated multi-tag entry |
| Repeated source labels on chips | Concise chips; source remains in tooltip | Source information without repetition |
| Segment and image navigation mixed in footer | Direct segment buttons and sequential segment controls together; Previous image and Save & Next in footer | Both navigation dimensions remain explicit |

The original dark theme, Bootstrap/vanilla JavaScript, font stack, API routes,
single/multiple choice rules, segment mapping, optional Autosuggest, separate
Autotagger, glossary, backups, and file semantics remain. Accepted D002 continues
to govern space-separated caption tags; catalog identifiers retain their existing
explicit conversion boundaries.

## Interaction and data changes

- One state-aware Start Tagging / Resume Tagging / Review Captions action replaces
  redundant primary-looking launcher actions. Session state and written caption
  counts are summarized without long filename banners.
- The main context uses project name and relative dataset/staging labels. Full
  paths remain inspectable; project/session mismatch refers to session ownership,
  not the legitimate distinction between project root and dataset root.
- Filename, caption availability, and selected-image filters operate on loaded
  image metadata. Counts explicitly identify loaded/total scope while filtering.
  Filters do not issue directory scans. Select All applies to visible results;
  staging acts on the displayed total selected count, including hidden selections.
- Cards prioritize thumbnail, filename, caption state, and a two-line tag summary.
  The remaining count uses `tag_count`, including tags beyond the API preview cap.
  Existing image/caption preview stays available. Truncated captions load their
  complete text through the existing read-only Dataset Health caption endpoint.
- Auto fit adapts the grid to available width; explicit 2/3/4-column choices and
  saved preferences remain. Desktop zoom keeps the staging sidebar beside the
  browser instead of placing it after a long gallery. The existing mobile layout
  remains authoritative at small widths.
- The browser toolbar owns Stage Selected. The duplicate sidebar control remains
  a hidden compatibility action for existing handlers. Drop targets and click
  actions remain; folder maintenance is secondary to staging/return.
- File operation results persist independently of refresh/loading messages,
  including moved/returned image counts, failures, destination, and per-file errors.
- Normal batch tag edits return structured inspected, changed, unchanged, skipped,
  failed, created-caption, and actual backup-write counts. Partial failures retain
  counts and actionable logs. Existing move/restore behavior is preserved.
- Tag scans do not overwrite a newer manual file selection. Selection changes do
  not rebuild thumbnails. Setup reparenting preserves focus, and an unchanged
  verified path is not checked again merely because its input loses focus.

The valid functional fixes from the preceding iteration remain: serialized draft
writes, stale-response protection, segment ownership of pending manual text,
typing-safe shortcuts, in-flight caption-save protection, failed-draft close/unload
guards, separate caption/session persistence results, project validation before
caption writes, keyboard-accessible removal, modal/drawer focus handling, paired
move previews/conflict handling, and pipeline review wiring.

## Verification and artifacts

Run the focused tests from the repository:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tag_editor_reporting.py tests/test_tagging_caption_save.py tests/test_tagging_session_reconcile.py tests/test_tag_editor_transactions.py tests/test_app_tag_editor_file_api.py tests/test_tag_representation.py tests/test_tag_editor_initialize.py tests/test_tag_editor_limits.py tests/test_ui_workflow_foundations.py tests/test_workflow_guide_readme.py -q
```

Run `tests/tag_editor_workspace.browser.cjs` against a local server on port 5091
(override with `BATCHBENCH_URL`). Set `NODE_PATH` to an available Playwright install
and `PYTHON` if fixture generation uses Python outside `.venv/Scripts/python.exe`.
No application dependency was added. The test creates/removes only its own temporary
project and Chromium profile. Its temporary local extension uses `chrome.tabs.setZoom`
to test real browser zoom, not CSS zoom or device pixel scaling.

The browser checks cover lazy loading from Home, the unified workspace, filename,
caption and selection filters, recommendation keyboard selection, tag removal,
direct/non-adjacent segment access, segment draft ownership, draft versus caption
writes, save failures, repeated saves, previous-image/free-tag navigation, focus
return, zoom/fit, long captions/filenames, 165 images, pair staging/return, all four
batch modes, backups, persistent operation counts, invalid/empty projects, and a
390×844 mobile smoke test. No models, accounts, or live inference are required.

Artifacts under `_work/tag-editor-corrective/`:

- `A-original/`: the original template rendered from `0805926` against temporary fixtures.
- `B-unsuccessful/`: screenshots retained before this corrective pass.
- `C-corrected/`: corrected screenshots and `layout-results.json`.
- `comparison.html`: standalone original/rejected/corrected visual comparison.

Desktop captures include 1366×768, 1440×900, and 1920×1080, plus constrained
1093×614 and 911×512 viewports. Real Chromium browser zoom is additionally checked
at 125% and 150% with a 1366×768 window. The image and Save & Next remain visible;
the right editor can scroll while direct segment buttons wrap rather than becoming
a dropdown. Wrapped labels and validation alignment are checked at narrower widths.

Final focused run: **74 tests and 10 subtests passed**. The Chromium browser workflow
suite passed after the final layout adjustment, including actual browser zoom.

Like-for-like file-path detector scans report **139 → 139** template findings and
**5 → 5** scoped stylesheet findings. Earlier inline template scans reported 136;
file-path scanning also detects three unchanged legacy typography/CSS findings.
Retained findings include mixed CSS units, the existing type ladder and duplicate
declaration, and static rules that cannot trace Bootstrap focus management or
handler-based confirmations. Runtime tests cover these flows; the detector is not
a WCAG certification or a claim that all legacy findings were eliminated.

Not covered: Firefox/Safari, packaged EXE behavior, OS file-picker interaction,
real-model downloads/inference, or the full repository test suite.
