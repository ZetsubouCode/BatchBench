# Tag Editor tokens

Use the existing Bootstrap/system font stack and shared `--bb-*` variables in
`templates/base.html`. This refinement does not replace the app theme.

| Purpose | Token / existing value |
| --- | --- |
| Page / surface | `--bb-bg` / `--bb-card` |
| Text / supporting text | `--bb-text` / `--bb-muted` |
| Borders / inputs | `--bb-border` / `--bb-input-bd` |
| Primary actions / focus | `--bb-primary` / `--bb-focus` |
| Links | `--bb-link` |
| Ready / resolved step | scoped `--te-success: #a5dfb6` |
| Staging / unsaved / warning | scoped `--te-warning: #ffd18a` |
| Error | scoped `--te-error: #ffb4b4` |
| Active step / valid source drop | scoped `--te-accent: var(--bb-focus)` |
| Control height / spacing | `--te-control-height: 2.25rem` / `--te-gap: .75rem` |

Status LEDs always accompany text. Step state also uses numbers, checkmarks, lock
icons, and accessible labels. Preserve existing category colors in Guided Tagging.
