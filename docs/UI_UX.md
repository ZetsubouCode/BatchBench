# UI / UX Guidelines

BatchBench is a local productivity tool. Optimize for speed, clarity, and low friction rather than visual novelty.

## 1. Visual Direction

Use the existing Bootstrap-based visual language and keep the interface utility-first.

Prefer:

- clear hierarchy,
- compact spacing,
- readable forms,
- useful grouping,
- obvious primary actions,
- restrained borders/background separation,
- icons when they improve scanning,
- concise helper text.

Avoid "AI slop" patterns:

- gratuitous gradients,
- glowing blobs or decorative light effects,
- marketing-style hero sections,
- eyebrow/kicker text above every heading,
- excessive cards,
- nested cards for simple content,
- oversized empty spacing,
- decorative badges with no functional meaning,
- repeated explanatory copy that pushes the actual tool below the fold.

A plain, dense, understandable utility screen is better than a polished-looking but slower workflow.

## 2. Information Architecture

Keep the current high-level categories coherent:

- Image Tools,
- Dataset Assembly,
- Tag Tools,
- Dataset Workflow,
- Reference/Settings.

Tagging-related features should remain easy to reach because tagging is the product's primary concern.

Do not add a new top-level category for a single small feature when it fits an existing category.

## 3. Tool Page Structure

A typical tool should expose, in roughly this order:

1. source/input,
2. output/target,
3. the small set of important options,
4. optional advanced controls,
5. preview/dry-run when meaningful,
6. primary action,
7. result summary/log.

Avoid putting every input inside its own card.

Use sections, spacing, simple borders, accordions, or fieldsets before creating additional nested containers.

## 4. Defaults

Defaults should optimize the owner's normal workflow and reduce repetitive setup.

Examples:

- derive an output folder when the destination is obvious,
- remember the active tab or recent safe settings when already supported,
- prefer offline/local resources when available,
- default potentially risky automation toward preview/review.

Do not invent "smart" behavior that changes files based on guesses.

## 5. Manual Tagging UX

Dataset Tag Editor and Guided Tagging Flow should minimize repetitive interaction.

Prioritize:

- keyboard shortcuts,
- fast next/previous navigation,
- visible progress,
- clear selected/current tags,
- quick add/remove actions,
- useful autocomplete,
- cheatsheet/glossary access without leaving the flow,
- resume behavior that preserves work,
- predictable single-choice/multi-choice semantics.

Automated recommendations should look like suggestions, not authoritative answers.

The UI must not imply that an autotagger result is inherently correct.

## 6. Tag Display

Human-facing caption tags should use the canonical BatchBench display form with spaces:

```text
long hair
looking at viewer
white shirt
```

If a feature must expose the underlying Danbooru identifier (`long_hair`) for debugging or reference, distinguish it from the caption representation instead of mixing both silently.

## 7. Destructive Actions

When an action overwrites, moves, or deletes data:

- label the behavior clearly,
- show the target path/scope,
- expose preview/dry-run where useful,
- do not hide overwrite behavior inside generic wording such as `Save`,
- show completion/error logs.

Do not add confirmation dialogs to every harmless action. Reserve confirmation friction for genuinely destructive operations that cannot be easily inspected beforehand.

## 8. Logs and Feedback

Every batch action should end with a trustworthy result state.

Prefer a compact summary such as:

```text
Scanned: 120
Changed: 84
Skipped: 34
Renamed conflicts: 1
Errors: 1
Output: D:\datasets\project\output
```

Then show actionable details for warnings/errors.

Long-running tasks should expose progress/status rather than looking frozen.

## 9. Offline and Network UX

Network-dependent behavior must be obvious.

Good examples:

- `Sync Danbooru catalog`,
- `Fetch wiki`,
- `Download/load model` when a remote model is selected.

Do not make ordinary typing, browsing local datasets, or manual tagging depend on live network calls.

When cached/local data is available, use it without requiring internet access.

## 10. Responsive Scope

The application is desktop-first because dataset preparation is primarily a desktop workflow.

Responsive behavior should prevent broken layouts on narrower windows, but mobile-first redesign is not a product requirement.

## 11. Accessibility and Clarity

Use proper labels, button text, focus behavior, and semantic controls where practical.

Tooltips should be short and explain non-obvious behavior, not duplicate visible labels.

Do not rely on color alone for error/success/destructive states.

## 12. New UI Dependencies

Do not add a frontend framework merely to build one new tool.

Continue with Jinja, Bootstrap, and focused vanilla JavaScript unless a new technology solves a concrete problem significantly better and the setup/maintenance cost is justified.
