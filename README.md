# BatchBench - Local Batch Image and Dataset Toolkit

BatchBench is a small Flask app that runs on your own machine at `localhost`.
It wraps common image, caption, and LoRA dataset preparation utilities behind a
browser UI.

Current feature set:

- **Workflow Guide** generated from this README.
- **Image Tools**: Image to PNG Converter, Photo Adjust, Brush Blur, Color Brush, Manga Palette Helper.
- **Dataset Assembly**: EPUB Image Extractor, Webtoon Panel Splitter, Stitch Groups, Flatten and Renumber, Combine Dataset.
- **Tag Tools**: Dataset Tag Editor, Dataset Normalization, Auto Tag Assist (WD v3), CLIP Token Check.
- **Dataset Workflow** with reorderable step cards and pause/resume controls.
- **Tag Glossary Wiki** with reusable glossary categories and Danbooru reference lookup.
- **Settings** for Guided Tagging Flow, local Danbooru tag suggestions, and tag catalog sync/import.

Runs on Windows, Linux, and macOS.

---

## 0) Requirements

- Install **Python 3.11+**: <https://www.python.org/downloads/>
- Make sure Python runs from a terminal with `python --version` or `python3 --version`.
- For the Offline Tagger, install a PyTorch build that matches your CPU or CUDA setup.

---

## 1) Extract the Project Folder

Example locations:

- Windows: `C:\BatchBench`
- Linux/macOS: `~/BatchBench`

---

## 2) Install Dependencies

Windows:

- Run `setup.bat` by double-clicking it or launching it from Command Prompt.

Linux/macOS:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

The setup script will:

- create `.venv`
- install packages from `requirements.txt`
- create `.env` from `.env.example` if `.env` is missing

---

## 3) Start the App

Windows:

- Run `run.bat`

Linux/macOS:

```bash
./run.sh
```

Open this URL in your browser:

- <http://127.0.0.1:5000/>

The launch script opens the app in your browser, watches app source files, and
reloads the local server after code changes. In debug mode, the browser page
also refreshes itself when app, template, or static files change.

---

## Discord Rich Presence

Discord Rich Presence is optional. BatchBench works normally without Discord.
No bot token, client secret, OAuth configuration, or public server is needed.
The activity only reports broad tool categories, never dataset contents, tags,
paths, prompts, trigger words, or filenames.

Manual setup:

1. Open Discord Developer Portal.
2. Create an application named `BatchBench`.
3. In the application's General Information page, upload an application icon.
   Discord uses this icon in compact activity surfaces; Rich Presence art
   assets are used inside the expanded profile activity card.
4. Copy its Application ID.
5. Run:

```bash
python scripts/export_discord_asset.py
```

6. In the application's Rich Presence / Art Assets section, upload:

```text
static/icons/discord_presence.png
```

7. Set the uploaded asset key exactly to:

```text
batchbench
```

8. Add this to `.env`:

```env
DISCORD_RICH_PRESENCE_ENABLED=true
DISCORD_APPLICATION_ID=your_application_id_here
DISCORD_PRESENCE_ASSET_KEY=batchbench
```

9. Restart BatchBench.
10. Keep Discord desktop open and make sure activity sharing is enabled.

The Discord asset must be uploaded manually once because Discord hosts Rich
Presence art assets.

Packaged Windows builds read writable BatchBench data from the same project
folder when launched from `dist\BatchBench\BatchBench.exe`. This includes the
Danbooru CSV/SQLite catalog, tag suggestion settings, Guided Tagging Flow
settings, and Tag Glossary Wiki data. If you move the EXE elsewhere, set
`BATCHBENCH_DATA_DIR` in `.env` to the BatchBench source/data folder you want
both source and EXE launches to share.

When launched through the Windows EXE, BatchBench also creates a system tray
icon. Use **Open BatchBench** to reopen the browser tab and **Exit BatchBench**
to stop the local server cleanly.

Windows build:

Double-click `compile_exe.bat`. It creates/uses `.venv`, installs build
requirements, and writes the executable to:

```text
dist\BatchBench\BatchBench.exe
```

Command-line equivalent:

```bat
compile_exe.bat
```

---

## 4) MD5 Checksums for Project Files

Run with the virtual environment Python:

Windows:

```bat
.venv\Scripts\python.exe md5sum.py
```

Linux/macOS:

```bash
.venv/bin/python md5sum.py
```

This prints MD5 values and writes `checksums.md`.

---

## Offline Tagger Models

The Offline Tagger accepts either a Hugging Face repo ID such as `org/model` or
a local model folder path.

Requirements:

- The model must work with `AutoModelForImageClassification` for multi-label image tagging.
- Safetensors weights are required because the app loads models with `use_safetensors=True`.
- A tag list file should exist as `selected_tags.csv` or `tags.csv`; otherwise the app falls back to `id2label`.

How to switch models:

1. Open **Offline Tagger (WD v3)**.
2. Set **Model ID or local path** to another model repo or local folder.
3. Run the tagger. The app downloads required files into the Hugging Face cache unless **Local files only** is checked.

Tip: if you keep models inside this repo, store them under `models/`; that
folder is ignored by git.

---

## 5) Tool Guide

This section is also used by the in-app **Workflow Guide**.

### Image Tools

#### Image -> PNG Converter
How to use:
- Fill **Source Folder** with a folder of images; **Output Folder** auto-fills as `<source>\output`. Click **Convert**.
Parameters:
- Source Folder: folder containing Pillow-supported image files at the top level.
- Output Folder: destination for generated `.png` files; defaults to `<source>\output` and is created if missing.
Watch out:
- This tool is not recursive.
- Every supported source image is converted to `.png`.

#### Photo Adjust (preset)
How to use:
- Choose Source Folder and Output Folder, select a preset, adjust suffix or limit, then click **Process**.
Parameters:
- Source Folder: top-level image folder for `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`.
- Output Folder: destination for adjusted photos.
- Preset: JSON preset in the project root, such as `preset_keep_warm_balanced.json` or `preset_greyscale.json`.
- Suffix: text added to output file names; default is `_adj`.
- Max files: `0` processes every file.
- Preset values map to `utils/image_ops.py`: `exposure_ev` uses `-4..4`; `brightness`, `contrast`, `highlights`, `shadows`, `saturation`, `warmth`, `tint`, and `sharpness` use `-1..1`; `vignette` uses values `>= 0`.
Watch out:
- Output is always JPEG at quality 95 with a `.jpg` extension.
- This tool is not recursive.
- Re-saving images can remove EXIF metadata.

#### Brush Blur
How to use:
- Fill Folder, click **Load Images**, select an image, then click **Open**.
- Choose Gaussian Blur, Mosaic, or Box Blur; adjust Brush Size, Strength, and Feather.
- Paint the mask, click **Preview Result**, then click **Apply & Save** when the preview is correct.
Parameters:
- Output: `overwrite` replaces the source image; `copy` writes a separate file.
- Create `.bak`: creates a backup before overwrite.
- Paint, Erase, Undo, Redo, and Clear Mask control the manual mask.
Watch out:
- Preview before saving.
- This tool edits one image at a time and is best for local touch-up work.

#### Color Brush
How to use:
- Fill Folder, click **Load Images**, select an image, then paint directly on the image.
- Choose Brush type, Color, Brush size, Opacity, and Paint/Erase/Pick Color tools.
- Use zoom, pan, Undo, Redo, and Clear Paint while reviewing the edit, then click **Apply & Save**.
Parameters:
- Folder: image folder scanned recursively by the Color Brush image list.
- Image: selected source image to paint.
- Brush type: Hard Round, Soft Round, Marker, or Airbrush.
- Color: synced color picker, hex input, and swatch.
- Brush size: size in source-image pixels.
- Opacity: paint layer opacity from `1%` to `100%`.
- Output: `Copy as _paint` writes a separate file; `Overwrite selected image` replaces the source image.
- Create `.bak` before overwriting: creates a backup only when overwrite mode is used.
Watch out:
- Edits stay in the browser until **Apply & Save**.
- Copy mode avoids changing the source image and never creates a duplicate backup.
- Very large images can take longer to save because the final paint layer is applied at original resolution.

#### Manga Palette Helper
How to use:
- Pick a base color from the color wheel, hex input, color picker, EyeDropper, or generated SVG sampler.
- Use the Triadic Scheme to choose a main color, eye accent, and contrast detail.
- Load a black-and-white manga image to create an SVG pixel sampler, then click a dominant color or SVG cell to sample it.
Parameters:
- Color/Gray wheel: Color shows a hue wheel; Gray shows a grayscale value map at the same hue position.
- Hex and Picker stay synced.
- Sat and Light adjust saturation and lightness for the triadic palette.
- SVG pixels controls raster sampling resolution before SVG cells are generated.
- Cell and Gap control the visual SVG cell size and spacing.
- Levels controls color quantization.
- Ignore white paper excludes white paper areas from dominant-color sampling.
Watch out:
- The tool does not write files unless you download the generated SVG.
- Very large images are downsampled in the browser to keep the UI responsive.

### Dataset Assembly

#### EPUB Image Extractor
How to use:
- Extract images from one EPUB or a folder of EPUB files into normal folders.
- Choose an output folder, keep reading order enabled, run a dry run first, then extract.
- Continue with PNG Converter, Flatten and Renumber, or Webtoon Panel Splitter if needed.
Best for:
- Preparing ebook or comic source images before renaming, converting, splitting, or tagging.
Parameters:
- EPUB file or folder: one `.epub` file or a folder containing multiple EPUBs.
- Output folder: each EPUB gets a subfolder at `<output>/<epub_stem>/`.
- Recursive: scans subfolders only when enabled.
- Use reading order: follows the OPF spine when possible.
- Extract cover image: places a clearly marked cover first.
- Extract SVG: off by default because SVG pages are usually not useful for dataset preparation.
- Rename mode: `sequential` writes names like `001.jpg`; `original` keeps internal file names.
- Dry run, Overwrite, and Create report JSON preview work, control replacement, and write `extract_report.json`.
Watch out:
- DRM-protected EPUB files cannot usually be extracted.
- Some EPUB files have messy internal order; reading-order mode is best effort.

#### Webtoon Panel Splitter
How to use:
- Fill Source folder, tune stripe detection, then click **Split panels**.
Parameters:
- Source folder: chapter folder or root folder containing chapter subfolders.
- Output folder: default is `_panels` per folder; if set, output goes under `<output>/<chapter>/_panels`.
- Filename glob and Extensions filter page files.
- Width alignment: `match-width` resizes to the largest width; `none` pads only.
- Stripe detection: Stripe colors, Color threshold, Row coverage, Tolerance inside stripe, and Min panel height tune panel cuts.
- Options: save strip, overwrite, dry run.
Watch out:
- Pages are stacked vertically and split on selected-color stripes, such as white gutters, black gutters, or both.
- If noise causes bad splits, lower Row coverage or Color threshold.

#### Stitch Groups
How to use:
- Fill Source folder, choose merge options, then click **Merge**.
Parameters:
- Source folder: contains files named like `<prefix>_<number>.<ext>`.
- Filename glob filters candidate files.
- Extensions to include controls accepted image extensions.
- Stack direction: Vertical or Horizontal.
- Resize rule: `auto` matches the stacking direction; `none` pads only; explicit rules force width or height matching.
- Gap and Background control spacing and canvas color.
- Skip single, Reverse order, Overwrite, and Dry run control output behavior.
Watch out:
- Only files matching `<prefix>_<number>` are grouped.
- Output is PNG and goes to `<source>/combined` when Output is empty.

#### Flatten & Renumber
How to use:
- Fill Root Folder, tune numbering and ordering, then click **Run Renamer**.
Parameters:
- Root Folder: contains images and first-level subfolders.
- Output Folder: default is `<root>/_renamed`.
- Image extensions: comma-separated extensions such as `.png,.jpg`.
- Start at, Prefix pad, Suffix pad, and Separator control generated names such as `001.png` and `002_1.png`.
- Ordering controls top-level, folder, and in-folder sort order by natural name, creation time, or modified time.
- Rename matching `.txt`: copies or renames paired captions.
- Move instead of copy: moves files out of the source.
- Dry run previews the plan without writing files.
Watch out:
- Only top-level files and first-level subfolders are processed.
- Name collisions are resolved with `-1`, `-2`, and so on.

#### Combine Dataset
How to use:
- Fill Source folders with one dataset folder per line, fill Output Folder, then click **Combine**.
Parameters:
- Source folders: at least two dataset folders.
- Suffix for non-base sets: added to file names from non-base datasets to avoid collisions; with more than two datasets, suffixes auto-increment as `_B`, `_C`, `_D`.
- Image ext: extensions treated as image and caption pairs.
- Move (not copy): moves files instead of copying them.
Watch out:
- Only top-level image and `.txt` pairs are processed.
- The base set is the source with the most pairs; ties follow input order.
- Name collisions get `_1`, `_2`, and so on.

### Tag Tools

#### Dataset Tag Editor
How to use:
- Fill **Project Root**, the parent folder that contains `database/`, `dataset/`, `dataset/_temp`, and `prompt.txt`.
- If Project Setup appears, run **Initialize Project** first.
- Use **Guided Tagging Flow** for structured image-by-image review.
- Use **Bulk Tag CRUD** for direct add, remove, rename, and duplicate cleanup operations.
Parameters:
- Project Root: dataset project root. The tool resolves `database/`, `dataset/`, and `dataset/_temp` automatically.
- Image extensions: scanned image extensions such as `.png,.jpg`.
- Tags / mapping: used by Bulk Tag CRUD for add/remove or rename mappings such as `old->new; old2->new2`.
- Create `.bak` backups: creates backups before bulk edits.
Bulk Tag CRUD:
- Add Tags appends tags that are not already present.
- Remove Tags deletes matching tags.
- Rename Tags replaces tags using `old->new; old2->new2` mappings.
- Clean Duplicates removes duplicate tags.
Guided Tagging Flow:
- The flow stores session state in `dataset/_temp/tagging_session.json`.
- Final output remains normal sidecar `.txt` caption files.
- Recommendation chips can repeat when the same tag comes from different mapped cheat-sheet groups, but selecting any copy adds one caption tag and hides the matching copies for that image.
- Manual tagging supports caption chips, autocomplete, glossary quick pick, keyboard shortcuts, and optional image preview inside the cheat-sheet overlay.
- Danbooru-aware autocomplete uses the local catalog under `data/tag_catalog/`; typing does not call Danbooru.
- Suggestions can show source, category, post count, validation state, alias mapping, glossary source, recent usage, and segment relevance when the local data supports it.
- Alias entries insert their canonical tag. Unknown tags are allowed and can be added to the project custom whitelist so they stop repeating as warnings.
- Segment ranking favors the active step, preferred tags, active tag packs, project history, recent tags, and glossary categories. **Show all tags** opens broader catalog results.
- Tag packs are project-local reusable groups stored in `database/tagging_assist.json`; applying a pack appends and deduplicates tags without overwriting existing tags.
- Pinned packs can be applied with `Alt+1` through `Alt+9`.
- **Uncertain / revisit later** stores uncertainty and notes in review metadata, not captions. The **Uncertain only** queue helps return to those images.
- Sibling propagation uses explicit groups and a preview. The default mode is append-only; replace modes must be selected deliberately by API callers.
- **Caption Lint** is advisory and never edits captions. It can report unknown tags, aliases, deprecated tags when known locally, duplicates, malformed tags, missing triggers, rare one-off tags, simple conflicts, grayscale/color advisories, and related quality-control issues.
- Optional machine suggestions are stored separately per image with model and threshold metadata. They do not change captions or completion status until the user adds a suggestion.
Watch out:
- Bulk Tag CRUD works in `dataset/_temp`.
- **Undo** restores files from `_temp` to `dataset/`.
- **Dataset zip** zips `dataset/` without `_temp`.
- **Move all from _temp** moves staged files into a timestamped folder.
- Add Tags can create new `.txt` files for images without captions after UI confirmation.
- The Tag Editor treats captions as simple tag lists. Use Dataset Normalization for files containing `#optional:` or `#warning:` blocks.

#### Dataset Normalization
How to use:
- Fill Dataset folder, choose preset type and preset file, then click **Scan** or **Dry-run preview**.
- Review the preview, then click **Apply** when the plan is correct.
Parameters:
- Dataset folder: folder containing images and `.txt` captions.
- Image extensions: image extensions matched with `.txt` captions.
- Include subfolders: scans recursively.
- Create missing `.txt`: writes empty captions for images without a caption.
- Create backups (`.bak`): backs up captions before overwrite.
- Normalize order: sorts tags according to the selected preset.
- Preset type/file: JSON preset under `presets/<type>/`.
- Extra remove, keep, and identity tags: comma-separated or newline-separated overrides.
- Move unknown backgrounds to `#optional:`: moves tags not listed in `allow_general` to the optional block.
- Background frequency threshold: overrides preset `frequency_threshold`; blank uses the preset value.
- Preview limit: number of files shown in preview.
Supported caption formats:
- `tag1, tag2, tag3`
- `#optional: tagA, tagB`
- `#warning: note`
Preset rules:
- Presets live under `presets/<type>/`, for example `presets/anime/normalize_v1.json`.
- Rule order is trim, replace, remove, deduplicate, optional handling, background policy, then sort.
- `remove_tags` removes exact matches unless the tag is kept.
- `remove_regex` uses case-insensitive Python regex.
- `replace_map` replaces exact tags.
- `keep_tags`, extra keep tags, and identity tags are preserved.
- `optional_handling` moves configured tags to `#optional:`.
- `sort` can prioritize groups before alphabetic ordering.
- `background_policy` can keep broad background tags while moving specific or rare background tags to optional.
Watch out:
- Run Scan or Dry-run before Apply.
- Invalid regex rules are ignored.
- Add new presets under `presets/<type>/` and click Reload in the UI.

#### Auto Tag Assist
How to use:
- Fill Dataset Folder, choose a WD model or local model path, configure the trigger tag and policy, then run **Preview** first.
- Review kept tags, removed tags, policy drops, and final leak counts in the log.
- Run tagging only after the preview looks correct.
Parameters:
- Dataset Folder: folder containing images and optional `.txt` captions.
- Model ID or local path: Hugging Face repo ID or local folder compatible with `AutoModelForImageClassification`.
- Trigger tag: inserted first and protected from filtering.
- Character policy: default is `Character - omit identity`, which removes recurring identity traits while keeping promptable expression, pose, outfit, and scene tags.
- Threshold mode: `fixed` uses configured confidence thresholds; `mcut` estimates per-image thresholds from score gaps.
- Replacement mode: replaces selected existing captions after creating a timestamped backup; preview writes nothing.
- Color sanity: can drop weak color-attribute tags when the image does not support them.
Watch out:
- The model can only return tags from its known vocabulary.
- Review preview samples before writing captions, especially after changing thresholds or policy overrides.
- A normal strict character-policy run should report `Final policy leaks: 0`.

#### CLIP Token Check
How to use:
- Fill Dataset Folder, choose mode, then click **Scan Tokens**.
- Review captions with the highest token counts or captions above Token Limit.
Parameters:
- Image Extensions: image and `.txt` pairs to scan.
- Token Limit: warning threshold; default is `77`.
- Top N: number of longest captions to show.
- Mode `estimate`: lightweight scan without an extra tokenizer.
- Mode `exact`: uses a CLIP tokenizer from `transformers`.
- Recursive scan: includes subfolders.
- Include `_temp`: scans staging captions when needed.
Watch out:
- Use this as a review signal, not an automatic deletion rule.
- Run it again after caption cleanup to confirm token counts are reasonable.

### Workflow

#### Dataset Workflow
How to use:
- Fill Dataset Source, Working Directory, and Output Directory.
- Configure Image extensions, Include subfolders, and Working copy mode.
- Add or remove step cards, drag them into order, then click **Run Pipeline**.
Parameters:
- Dataset Source Folder: original dataset copied into the working folder.
- Working Directory: job workspace at `<working>/jobs/<job_id>/`.
- Output Directory: destination for final artifacts such as zip files.
- Working copy mode: `copy`, `hardlink`, or `incremental`.
- Run controls: Run Pipeline, Pause, Resume, Stop, and Open working folder.
- Step cards can run Offline Tagger, Tag Editor, Normalization, Zip result, dedup tags, converter, photo adjust, combine, renumber, stitch, and webtoon split steps.
Workflow:
- Prepare working copy, run step cards in order, then update artifacts and logs.
Watch out:
- The pipeline does not modify the original dataset.
- Put image-only steps before steps that edit captions.
- `Zip result` should usually be the last step.
- The Dataset Tag Editor pipeline step uses the `_temp` staging workflow for non-manual modes.
- Pipeline state is stored in `_work/pipeline_jobs/<job_id>/state.json`.

### Reference

#### Tag Glossary Wiki
How to use:
- Add and categorize reusable tags.
- Select a tag collection to load wiki text, short guidance, related tags, and a reference image from Danbooru.
- Use linked tags to compare choices, then add useful tags to the glossary.
Parameters:
- Glossary data is stored in `tag_editor_glossary.json`.
- Danbooru article and image lookup happens only when requested or when a tag view needs it.
Watch out:
- Internet access is required for the first Danbooru wiki or reference-image fetch.
- The glossary syncs with the Dataset Tag Editor quick picker in the same browser session.

#### Settings
How to use:
- Configure **Guided Tagging Flow** step cards. Each card is one review question such as Body Composition or Camera Angle.
- Choose `Single choice` when one answer should replace other tags in the same step.
- Choose `Multi choice` when several tags may remain together.
- Choose `Manual tagging` for full-caption editing with autocomplete and glossary quick pick.
- Configure Danbooru Tag Suggestions by syncing from Danbooru, importing CSV, exporting CSV, rebuilding the local index, and choosing category filters.
- Save the config after reviewing changes.
Parameters:
- Step Label and ID define the user-facing name and stable metadata key.
- Mode controls single-choice, multi-choice, or manual-tagging behavior.
- Default queue controls the initial filter: Missing only, All images, Conflict only, Not reviewed, or other supported queue modes.
- Required, Auto advance, Segment-only autosuggest, Danbooru catalog autosuggest, and Allow Not Applicable tune step behavior.
- Tags define answer options for single and multi-choice steps.
- Global Behavior controls default target area, image fit, thumbnail preload count, keyboard shortcuts, auto-save, current tags, and progress display.
- Catalog settings control local Danbooru suggestions, deprecated tag inclusion, minimum post count, maximum suggestions, and allowed Danbooru categories.
- Create steps can build editable draft steps from a `prompt.txt` cheat sheet.
- The Tagging Flow Setup preview keeps duplicate recommendation chips only when they come from different mapped cheat-sheet groups; defaults and final captions still count each normalized tag once.
Watch out:
- Danbooru catalog suggestions are local while typing. The app contacts Danbooru only when you manually sync or fetch wiki/reference data.
- Export JSON before large flow changes if you want a backup.
- Change step IDs carefully after review has started because IDs are used by metadata.
- Cheat sheets are vocabulary references; Guided Tagging Flow is the workflow. Imported steps are not saved until you click Save config.

---

## Offline Autotagger Reference

### What the autotagger does

The Offline Autotagger uses the selected WD model to produce probabilities for known Danbooru tags. It does not generate arbitrary natural-language captions, and it cannot invent tags outside the model vocabulary. Accepted tags are written as comma-separated sidecar `.txt` captions with spaces (`long hair`). Danbooru lookups and policy matching continue to use underscore keys internally (`long_hair`).

### Processing order

```text
WD inference
-> semantic policy
-> color sanity
-> WD category split
-> MCUT/fixed threshold
-> tag limits
-> custom exclusions
-> final policy safety filter
-> trigger insertion
-> leak audit
-> .txt write
```

The semantic policy runs before MCUT and tag limits so blocked identity tags do not consume the strongest scores or fill the tag cap before useful controllable tags are considered.

### Character identity policy

`Character - omit identity` is the default policy for new runs.

| Group               | Examples                                  | Default  |
| ------------------- | ----------------------------------------- | -------- |
| Character names     | named WD character tags                   | Removed  |
| Demographic         | `1girl`, `solo`                           | Removed  |
| Appearance identity | `blue hair`, `long hair`, `blue eyes`     | Removed  |
| Body identity       | `large breasts`, `wide hips`, `dark skin` | Removed  |
| Permanent marks     | `tattoo`, `scar`, `mole`                  | Optional |
| Expression          | `smile`, `closed eyes`                    | Kept     |
| Pose/framing        | `sitting`, `cowboy shot`                  | Kept     |
| Outfit/accessory    | `red dress`, `hair ornament`              | Kept     |
| Scene               | `indoors`, `window`, `sunset`             | Kept     |

Omitted recurring identity is expected to bind more strongly to the trigger, while captioned variable details remain promptable. Custom keep and block overrides are available in Advanced settings.

### Threshold modes

`fixed` uses static confidence thresholds for general and character tags. `mcut` estimates a per-image threshold from score gaps, with relax values that can lower the threshold for each category. Minimum-tag settings can force MCUT to keep weaker tags; after identity groups are removed this can introduce low-confidence filler, so `Policy MCUT minimum general tags` defaults to `0` for strict character-policy output.

### Existing caption behavior

Replacement is the new default. When replacement is enabled, existing `.txt` captions selected for processing are copied to:

```text
.batchbench_backup/offline_tagger_YYYYMMDD_HHMMSS/
```

Then the generated main caption is written atomically. Turning replacement off skips non-empty existing captions and only creates missing or empty captions. Preview mode writes nothing and creates no backup. Legacy `write_mode=append` is still accepted for saved configurations and direct callers, but it is not exposed as the normal default UI behavior.

### Trigger tag

The trigger tag is always pinned first. It is exempt from semantic policy filtering, exact exclusions, regex exclusions, sorting, and the final leak audit.

### Color sanity

Color sanity checks color-attribute tags such as hair, eye, and skin colors against the image. Low-presence color tags can be dropped unless their WD score passes the high-score override. Its ratio, saturation, value, and override parameters are available in Advanced settings.

### Preview and logs

Use Preview only to verify kept and removed tags before writing. Logs show the selected policy, blocked groups, replacement mode, backup path, sample kept tags, sample policy removals, aggregate policy drops, skipped existing captions, and final leak count. A normal character-policy run should report `Final policy leaks: 0`.

### Pipeline behavior

The Dataset Workflow Offline Tagger uses the same `services/offline_tagger.py` and `services/tag_policy.py` implementation as the standalone tool. New workflow configurations default to `Character - omit identity` and replacement with backups, while legacy saved `write_mode` values are still parsed.

### Limitations

- Semantic groups are deterministic rules, not perfect visual understanding.
- WD can only return tags from its known vocabulary.
- Low-confidence or visually ambiguous content may still require manual correction.
- Policy leak audit verifies tags, but you should still review samples.
- Outfit details that are also permanent character design elements may need custom keep/block overrides depending on the dataset.

### Examples

Example A: identity removed

Input WD candidates:

```text
1girl, solo, blue hair, long hair, blue eyes, smile,
looking at viewer, red dress, sitting, indoors, window
```

Output:

```text
mytrigger, smile, looking at viewer, red dress, sitting, indoors, window
```

Example B: expression exception

Input:

```text
1girl, white hair, closed eyes, hair ornament, smile, upper body
```

Output:

```text
mytrigger, closed eyes, hair ornament, smile, upper body
```

---

## Tips

- If a tool says **folder not found**, paste the full path, for example:

  ```text
  D:\_Training DATA\MySet
  ```

- You can drag a folder into many path boxes to paste its path.
- Back up important data before bulk operations.
