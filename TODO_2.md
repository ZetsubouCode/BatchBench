# TODO: Tab Baru "A-to-Z Pipeline" (Semi-Otomatis) untuk Dataset Preparation
## Status singkat
- Tab Pipeline sudah menyediakan form input, toggle run CivitAI autotag, pause manual wajib, watch download, dan kontrol Run/Pause/Resume/Stop dengan progress + log.
- Backend worker pipeline (job + state machine) jalan di background thread, bisa resume/pause/stop, normalize memakai preset, dan zip final.
- Download detection berbasis watch folder + pattern, dengan opsi auto-detect atau manual resume.

Goal: user cukup load dataset + settings, klik Run, lalu sistem:
- (opsional) prepare zip untuk CivitAI autotag
- (opsional) bantu user upload ke CivitAI (manual assisted)
- auto detect file hasil download dari CivitAI, extract
- normalisasi tags (preset)
- PAUSE untuk manual retagging (wajib)
- dedup + cleanup final
- zip output siap upload training

Catatan Realitas:
- CivitAI public REST API saat ini tidak menyediakan endpoint upload/autotag publik (read-only). Jadi flow upload/autotag harus "manual assisted" atau gunakan browser automation (risk/ToS). Default: manual assisted.

---

## 1) UI/UX: Tab "A-to-Z Pipeline"
### 1.1 Section: Input
- [ ] Dataset Source Folder picker (root folder, images)
- [ ] Working Directory picker (temp workspace)
- [ ] Output Directory picker (zip final)
- [ ] Toggle recursive include subfolder
- [ ] Dropdown preset normalisasi (type: anime/manga/manhwa/etc, file json)
- [ ] Checkbox: "Run CivitAI Autotag Step" (default ON)
- [ ] Checkbox: "Pause for Manual Retagging" (default ON, tidak bisa dimatikan kalau pipeline include tagging manual)
- [ ] Preview limit untuk diff (default 30)

### 1.2 Section: CivitAI Autotag (Assisted)
- [ ] Field: "Downloads Watch Folder" (default: OS downloads)
- [ ] Field: "Expected download filename pattern" (regex / contains string)
- [ ] Button: "Open CivitAI Autotag Upload Page" (open new tab)
- [ ] Button: "Copy zip path" / "Open folder containing zip" (mempermudah upload manual)
- [ ] Toggle: "Auto-detect downloaded zip from watch folder" ON/OFF
- [ ] Timeout / polling interval config (misal 2 detik)

### 1.3 Section: Run Control
- [ ] Button "Run Pipeline"
- [ ] Button "Pause/Resume"
- [ ] Button "Stop"
- [ ] Progress bar + log panel
- [ ] Current step indicator (state machine)
- [ ] "Open working folder" button

---

## 2) Pipeline State Machine (Wajib supaya bisa on-hold & resume)
Buat struktur job + step deterministic.

### 2.1 Data Model (in-memory + persisted to JSON)
- [ ] `PipelineJob`:
  - id (uuid)
  - created_at
  - status: QUEUED | RUNNING | WAITING_MANUAL | WAITING_DOWNLOAD | COMPLETED | FAILED | STOPPED
  - current_step
  - config snapshot (input settings)
  - log list (timestamp, level, message)
  - artifacts:
    - source_zip_path
    - civitai_result_zip_path
    - extracted_folder
    - final_zip_path

- [ ] Persist file: `working_dir/.pipeline_job.json` (agar bisa resume setelah restart flask)

### 2.2 Step Interface
- [ ] `PipelineStep`:
  - name
  - kind: AUTO | MANUAL_WAIT | EXTERNAL_WAIT
  - run(job) -> StepResult
  - StepResult: SUCCESS | WAIT(reason) | FAIL(error)

### 2.3 Worker
- [ ] Background worker thread/queue:
  - FIFO job processing
  - allow pause/stop
  - safe shutdown: persist state

---

## 3) Step List (A to Z)
### Step A: Validate & Index Dataset
- [ ] Validate folder exists
- [ ] Collect image files (png/jpg/webp) based on allowed ext
- [ ] If no images -> FAIL
- [ ] Create working subfolders:
  - `work/raw_zip/`
  - `work/civitai_result/`
  - `work/normalized/`
  - `work/final/`

### Step B: Build Zip for CivitAI Autotag (AUTO)
- [ ] Create zip of images only (exclude .txt)
- [ ] Name zip deterministic:
  - `{dataset_name}__for_autotag__{YYYYMMDD-HHMMSS}.zip`
- [ ] Save to `work/raw_zip/`
- [ ] Artifact: source_zip_path

### Step C: Upload to CivitAI (MANUAL_ASSISTED)
Default: MANUAL WAIT (karena belum ada API upload publik).
- [ ] Show UI instruction:
  - "Klik Open CivitAI page"
  - "Upload file: {source_zip_path}"
- [ ] Button "Open CivitAI upload page"
- [ ] Button "Open file explorer to zip location"
- [ ] Job status -> WAITING_MANUAL sampai user klik "I have started upload"

> Optional advanced (config hidden):
> - Browser automation via Playwright untuk upload form.
> - Mark as EXPERIMENTAL + warning (bisa break, bisa bertentangan dengan aturan situs).
> - Default OFF.

### Step D: Wait for Download Result (EXTERNAL_WAIT)
- [ ] Monitor Downloads Watch Folder:
  - detect new .zip created after job start
  - match pattern (regex/contains)
- [ ] Once found:
  - Artifact: civitai_result_zip_path
  - move/copy ke `work/civitai_result/`
- [ ] If timeout: FAIL dengan pesan "No downloaded zip detected"

### Step E: Extract Result Zip (AUTO)
- [ ] Extract ke `work/civitai_result/extracted/`
- [ ] Pastikan struktur berisi images + .txt
- [ ] Artifact: extracted_folder

### Step F: Normalize Tags (AUTO)
- [ ] Jalankan engine normalisasi (yang kamu rancang di tab Normalization):
  - remove_tags + replace_map + dedup + optional policy
  - backup .txt ke `.bak` di folder kerja
- [ ] Output tetap di folder extracted (atau copy ke `work/normalized/`)

### Step G: PAUSE untuk Manual Retagging (MANUAL_WAIT)
Ini step wajib karena judgement karakter/outfit/POV/lighting/body composition.
- [ ] UI:
  - tombol "Open Tag Editor tab" (ke tab lain)
  - tombol "Open normalized folder"
  - tombol "Resume pipeline"
- [ ] Mode WAITING_MANUAL sampai user klik Resume
- [ ] Saat Resume:
  - optionally scan perubahan timestamp `.txt` untuk memastikan ada edit (jika tidak ada, tampilkan warning saja, jangan block).

### Step H: Final Cleanup / Dedup (AUTO)
- [ ] Dedup final:
  - dedup tag dalam main tags
  - dedup tag dalam optional
  - remove empty tags
  - normalize comma+space
- [ ] Optional: enforce ordering groups (identity -> composition -> lighting -> env -> outfit)
- [ ] Sanity checks:
  - file tanpa .txt -> create empty .txt atau FAIL (config)
  - file .txt kosong -> allow (config)

### Step I: Build Final Zip (AUTO)
- [ ] Zip images + .txt (final dataset)
- [ ] Name zip:
  - `{dataset_name}__final__{YYYYMMDD-HHMMSS}.zip`
- [ ] Save ke output directory
- [ ] Artifact: final_zip_path

### Step J: Done (AUTO)
- [ ] status COMPLETED
- [ ] Show:
  - path zip final
  - button "Open output folder"
  - button "Copy path"

---

## 4) API Routes (Flask) untuk Tab Pipeline
- [ ] GET `/api/pipeline/presets` -> list preset types/files
- [ ] POST `/api/pipeline/start`
  - payload: dataset_path, working_dir, output_dir, run_autotag, preset, downloads_watch, pattern, etc
  - returns: job_id
- [ ] POST `/api/pipeline/pause` {job_id}
- [ ] POST `/api/pipeline/resume` {job_id}
- [ ] POST `/api/pipeline/stop` {job_id}
- [ ] GET `/api/pipeline/status` {job_id} -> status + current_step + logs + artifacts
- [ ] POST `/api/pipeline/open-path` (optional) -> server trigger open folder (Windows `explorer.exe`) untuk convenience

---

## 5) "Background Tagging" Ideal (yang boleh dilakukan sistem)
Sistem normalisasi tidak boleh "mengarang" background tags dari nol.
Yang boleh:
- [ ] remove tag sampah
- [ ] replace varian/tag typo ke standar
- [ ] move background terlalu spesifik -> `#optional`
- [ ] frequency policy:
  - jika tag background sangat jarang muncul & bukan protected -> optional
- [ ] keep tag general (indoors/outdoors/day/night/city/nature/etc)

Yang tidak boleh:
- [ ] menambah tag background baru yang tidak ada (itu judgement).

---

## 6) Acceptance Criteria
- [ ] Run pipeline tanpa autotag: normalize -> pause manual -> final zip bekerja
- [ ] Run pipeline dengan autotag: zip -> manual upload -> auto detect download -> normalize -> pause -> final zip bekerja
- [ ] Resume pipeline setelah pause tetap lanjut tanpa restart
- [ ] Semua overwrite `.txt` punya backup
- [ ] UI jelas step mana yang menunggu user (manual) vs menunggu file (download)
