# BatchBench Decision Log

This file records product/engineering decisions that should remain stable until explicitly changed.

## D001 - Manual tagging is the primary workflow

**Status:** Accepted

BatchBench is primarily a system for structured manual dataset tagging. Auto Tag Assist and other automated tag sources are helpers, not the final authority.

**Consequence:** UX and architecture should optimize review speed, resume safety, and manual control before maximizing automation.

---

## D002 - Caption tags use spaces, not underscores

**Status:** Accepted - migration required

Final/user-facing BatchBench caption tags must use forms such as:

```text
long hair, blue eyes, looking at viewer
```

rather than:

```text
long_hair, blue_eyes, looking_at_viewer
```

Danbooru catalog/API matching may retain underscore-form identifiers internally.

**Consequence:** Implement an explicit catalog/query <-> caption conversion boundary. Do not globally replace underscores in arbitrary strings.

---

## D003 - Tag Editor project layout is scoped to Tag Editor

**Status:** Accepted

`database/`, `dataset/`, `dataset/_temp/`, and `prompt.txt` form the canonical Dataset Tag Editor project layout.

Other tools may continue to operate on normal source/output folders without adopting this project structure.

---

## D004 - Safety comes from separation before backup files

**Status:** Accepted

A universal `.bak` policy is not desired.

Prefer working copies, staging, separate outputs, preview/dry-run, and transactional pair operations.

Existing feature-specific `.bak` behavior may remain when useful, but new tools should not add backups automatically just because other tools have them.

---

## D005 - Offline-friendly is a product requirement

**Status:** Accepted

Core manual dataset preparation should work without internet access.

Danbooru synchronization/wiki fetches, remote model downloads, and Discord presence are optional integrations.

---

## D006 - Windows is primary; Linux remains a compatibility target

**Status:** Accepted

Windows is the main tested/optimized platform and may have packaged-runtime conveniences.

Linux should continue to run when reasonable. macOS is best effort and should not be advertised as a hard compatibility guarantee without testing.

---

## D007 - Flask/local web architecture remains the default, not a dogma

**Status:** Accepted

The current Flask + Jinja + Bootstrap/JavaScript stack is simple enough for the product and should remain the default.

A different technology may be introduced when it gives a significant practical improvement and does not create disproportionate fresh-install or maintenance burden.

---

## D008 - Refactor by extraction when touched

**Status:** Accepted

Do not perform a broad architecture rewrite only to make the repository look cleaner.

When a feature would materially worsen a large file such as `app.py` or `templates/dataset_labeling.html`, extract the cohesive logic needed by that feature while preserving existing behavior.

---

## D009 - Testing is pragmatic and targeted

**Status:** Accepted

Tests should cover important behavior and regressions, but routine development should not require slow model downloads, live network requests, or long inference runs.

Run focused tests appropriate to the touched area. A full suite is useful when the change is broad and the suite can run without expensive external work, but it is not a ritual requirement for every small change.

---

## D010 - UI is a utility workspace, not a marketing page

**Status:** Accepted

Avoid gratuitous gradients, decorative hero patterns, eyebrow text, excessive boxes/cards, and nested containers that resemble generic AI-generated dashboards.

Optimize for practical density and clear workflow.

---

## D011 - Public repository, owner-first product

**Status:** Accepted

The repository is public so others can use it, but the product primarily exists to improve the owner's personal dataset preparation workflow.

General usability improvements are welcome when they do not force major workflow compromises or unnecessary complexity.

---

## D012 - Repository guidance files are tracked

**Status:** Accepted

`AGENTS.md` and development docs may be committed to the repository.

The default development branch is `main` unless a task explicitly says otherwise.

When Codex is asked to commit/push from the user's local environment, it must use the repository's existing Git identity/account configuration. It must not rewrite Git identity, invent another author, or include unrelated working-tree changes.
