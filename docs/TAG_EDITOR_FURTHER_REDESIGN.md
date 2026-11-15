# Tag Editor task 002 implementation

## Audit and baseline

The working tree was already dirty before this task. Existing changes in README,
UI/status docs, `services/tag_editor.py`, the Tag Editor template, workspace JS/CSS,
browser tests, draft-save tests, reporting tests, design brief, and redesign notes
were inspected and retained. `tag_editor_glossary.json` was already modified and
was not edited for this task. No reset, checkout, stash, commit, or push was used.

The checked-out branch was `neo`. Read-only inspection of its committed template
and the working-tree diff confirmed the original image/selected/recommended/manual/
glossary panel hierarchy and sequential Previous/Next/Skip actions. Both committed
`neo` and the prior uncommitted direct segment buttons permitted arbitrary jumps;
this task adds the requested progression gates while retaining the newer draft-save,
caption-write, project validation, and modal/focus protections.

| Existing behavior | Implemented behavior | Reason |
| --- | --- | --- |
| Freely navigable segment buttons | Connected numbered nodes, resolved checks, active circle, locked future steps | Make progression explicit while preserving completed-step revisits |
| Narrow permanent staging sidebar | Optional saved split layout with pointer/keyboard resizing; stacked below 1100px | Give normal browsing full width and staging useful space |
| Flat interactive pair moves | Mirror dataset-relative folders into staging and strip `_temp/` on return | Make temporary staging a predictable round trip |
| Dataset scan plus duplicate staging scan | One explicit staged-caption scan | Match the batch-edit target and remove redundant glow |
| Manual input overwritten by later tag selection | Manual edits pause auto-sync until explicit Sync to Tags | Preserve user input |
| Narrow explorer and composer | Full-width batch section with incremental multi-column chips | Keep hundreds of tags practical |
| Ordinary text status / cramped export | LED labels tied to real save/operation state; separate export section and exclusion warning | Make status and archive scope inspectable |

## Interaction details

- Step state derives from saved tag selections, manual tags, and explicit Not Applicable
  decisions. Required steps that disallow Not Applicable need tags. All navigation
  entry points share the gate, including keyboard, mobile controls, and wheel actions.
  Revisiting does not change later choices; progress counts resolved steps, not the
  active index. Save & Next checks every guided step. Free tagging remains available
  for completed captions. The stepper uses the actual session segments, including
  the ten-segment fixture tested here; it does not replace user step configuration.
- The image pane, zoom, preview, recommendations, glossary, autosuggestions, session
  persistence, and caption-write safeguards remain. Not Applicable sits with desktop
  step navigation and remains in the mobile action sheet.
- Browser panels accept app MIME payloads, not plain-text paths. A source upload zone
  still handles external file uploads separately. Both directions have button alternatives.
  Failed moves retain selections; successful moves update source, staging, counts,
  folders, and explorer once per batch. Processing disables repeat operations.
- Temporary browsing provides filename search, sorting, folder filters, selection,
  previews, caption indicators, tag summaries, and Load More. Filters operate on the
  explicitly counted loaded items; files outside the current load need Load More.
- Tag Explorer starts at 40 chips, supports Show More, search, frequency, sorting,
  Selected Only, selected chips/removal, clear/sync, and separate Danbooru info controls.
  Frequency means caption-file presence. Manual composer text is preserved across
  selection and scans. Selected tags still drive the existing gallery match logic.
- Preview uses the same four edit modes but writes neither captions nor backups.
  It is a snapshot/dry run, not rollback or a reservation of the later Apply result.
  Existing optional `.txt.bak` behavior remains for Apply. Batch targets remain staging.
- Dataset maintenance retains folder creation, empty-folder removal, confirmed folder
  deletion, and explicitly labeled flatten-to-root. Staged files can return to matching
  folders, move to the selected dataset folder, or move into a new timestamped folder.
- Confirmation actions wait for the dialog to finish opening and resolve after it closes, preventing a stuck backdrop on fast clicks. Image previews open/close immediately for repeated inspection.
- Export still downloads ZIP directly and excludes all of `_temp/`. Other files under
  dataset remain included, matching existing export behavior. Export never moves staging.

## Backend boundary

`app.py` delegates the existing sidecar-move helper to `services/pair_moves.py`.
The shared service plans matching nested destinations for preview and apply, detects
case-insensitive image/caption/backup conflicts, reserves missing-caption filenames,
uses deterministic suffixes, and rolls back partial pair transfers. An application
lock serializes pair moves. External filesystem changes can still invalidate a preview;
the existing preview fingerprint rejects stale plans. An unrelated process changing
files during the actual transfer is outside that lock. Conflict renames remain renamed
when returned; no filename provenance is invented for legacy staged files.

## Files changed for this task

- `templates/dataset_labeling.html`: shared navigation gates, progress, browser/explorer
  integration, batch preview, move feedback, and operation guards.
- `static/js/dataset_tag_editor_workspace.js`: layout/divider, staged browser, connected
  stepper, status labels, explorer controls, and maintenance/export composition.
- `static/css/dataset_tag_editor_workspace.css`: scoped desktop/responsive styling.
- `static/js/dataset_tag_editor_mobile.js`: locked segment options, Not Applicable state,
  and opening the optional workspace from the mobile dock.
- `templates/base.html`, `static/js/batchbench_context.js`: left Tools trigger, consistent
  path buttons, responsive paths, and Change-project focus behavior.
- `services/pair_moves.py`, `app.py`: shared transactional pair move and preserved paths;
  explicit all-subfolder flattening remains available.
- `services/tag_editor.py`: read-only batch dry runs using existing edit logic.
- `tests/test_tag_editor_further_redesign.py`, `tests/test_app_tag_editor_file_api.py`:
  pair/path/preview regressions and updated intentional nested-return expectation.
- `tests/tag_editor_further_redesign.browser.cjs`, `tests/tag_editor_workspace.browser.cjs`:
  new workflow coverage and adaptation of the earlier regression suite to current UI.
- README, dataset rules, UI guidelines, implementation status, this report, and
  `.ui-craft/brief.md` / `.ui-craft/tokens.md`: workflow and design documentation.

## Verification

Focused Python run: **80 passed and 10 subtests passed**, covering the existing editor,
caption-save, session-reconcile, transaction, file-API, representation, initialization,
limits, reporting, workflow foundations, README, and new task-002 regression tests.
The task-002 Chromium workflow suite passed with screenshots at **1920, 1440, 1024,
768, and 375px**, including ten connected steps, locking, revisit/progress, save/resume,
resizing, nested staging/return, explorer with 310 tags, manual sync pause/resume,
all four edit modes, backup creation, preview, and export.

The earlier Chromium regression suite also passed against the final code, including
165 images, long filenames/captions, keyboard removal, draft ownership, caption/draft
save failures, repeated Save, image reuse, previous-image/free tagging, close guards,
focus restoration, real browser zoom at 125%/150%, and mobile smoke checks. Its
screenshots are in `_work/tag-editor-further/regression/`.

Artifacts: `_work/tag-editor-further/`, including workspace/guided PNGs and
`layout-results.json`. JavaScript syntax checks and `git diff --check` passed.
The scoped CSS detector reports **24 findings: 23 mixed-unit warnings and one type-scale
finding**, with no critical findings. These are not a claim of WCAG certification.

Not verified: packaged EXE, OS drag from Explorer/file picker, Firefox/Safari, real-model
inspection or downloads, live Danbooru requests, assistive-technology hardware, or the
full repository test suite. All functional data checks used temporary projects.
