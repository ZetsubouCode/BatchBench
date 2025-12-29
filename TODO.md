# TODO / Rundown Implementasi: Tab “Dataset Normalization” + Fondasi Tab “Pipeline”

## Progress (sudah diimplementasi)
- Tab Dataset Normalization sudah ada (scan, dry-run preview, apply + backup, preset picker, optional block).
- Tab Pipeline (beta) ditambahkan untuk fondasi config chaining + export JSON.
- Backend normalizer baru: parser #optional/#warning, preset schema, background policy, backup, stats.
- Preset contoh: presets/anime/normalize_v1.json dan presets/realistic/normalize_basic.json.
Target: nambah 2 tab baru di BatchBench (Flask local web) untuk mempercepat workflow retagging LoRA.
Fokus utama:
- Tab 1: Normalisasi dataset (hapus tag sampah + rapihin konsistensi) secara cepat & aman.
- Tab 2: Pipeline (nanti detail lanjutan dari user) — saat ini cukup siapkan “fondasi” chaining step dan export config.

---

## 0) Prinsip desain (biar workflow makin cepat)
1. Semua aksi harus punya **dry-run preview** (lihat perubahan sebelum apply).
2. Semua apply harus aman: buat **backup** `.txt` sebelum overwrite.
3. Normalisasi harus bisa jalan tanpa ngerusak “judgement tags” manual:
   - jangan sentuh tags karakter/kostum/POV/lighting/body composition yang user set manual
   - normalisasi fokus ke: tag sampah, duplikat, typo umum, tag background terlalu spesifik, metadata/artist/rating, dll
4. Output tagging tetap Danbooru-style: `tag1, tag2, tag3` dan opsional blok:
   - `#optional: ...`
   - `#warning: ...` (untuk tag salah kaprah / kemungkinan salah)

---

## 1) Tab Baru: “Dataset Normalization”
### 1.1 UI Layout (Field yang harus ada)
A. Dataset Picker
- [ ] Dropdown / input path untuk memilih **dataset folder** (root) yang isinya gambar + `.txt` tags
- [ ] Toggle include subfolder (recursive) ON/OFF
- [ ] Filter file: hanya pasangan yang punya `.txt`, atau auto-create `.txt` kosong jika tidak ada (opsional)

B. Preset Picker (JSON Preset Directory)
- [ ] Dropdown “Preset Type” (contoh: anime, manga, manhwa, realistic, misc) — ini mapping ke sebuah folder berisi preset json
- [ ] Dropdown “Preset File” yang load list file `.json` dari folder preset tersebut
- [ ] Button “Reload presets” untuk refresh list tanpa restart server
- [ ] Panel “Preset Summary” (read-only) menampilkan ringkas isi preset (berapa remove tags, replace map, dll)

C. Custom Rules (Override user)
- [ ] Textarea: “Extra remove tags” (dipisah koma/enter) → di-merge ke remove list preset
- [ ] Textarea: “Extra keep tags” (opsional) → tag di sini tidak boleh terhapus walau masuk remove list (untuk safety)
- [ ] Toggle: “Move unknown/specific background tags to #optional” (ON/OFF)
- [ ] Slider/number: “Background tag frequency threshold” (opsional)
  - contoh: tag background yang muncul < X% → pindah ke optional (untuk netral)
- [ ] Toggle: “Normalize order” (urutkan tag dengan prioritas tertentu) ON/OFF

D. Preview & Apply
- [ ] Button “Scan dataset” (ambil sample & statistik)
- [ ] Button “Dry-run preview” (tampilkan perubahan untuk N file: before/after diff)
- [ ] Input number: “Preview limit” (default 30)
- [ ] Button “Apply normalization” (eksekusi semua perubahan)
- [ ] Checkbox: “Create backups (.bak)” default ON
- [ ] Output log panel + summary:
  - total file diproses
  - file berubah berapa
  - total tag dihapus/replace/dedup/move optional

---

## 2) Spesifikasi Preset JSON (wajib dibuat standar)
Buat schema yang simple tapi kuat.

### 2.1 JSON Schema Minimal
- name: string
- description: string
- version: string
- rules:
  - remove_tags: [string]                 # hapus tag persis match
  - remove_regex: [string]                # hapus tag yg match regex (optional)
  - replace_map: { "old_tag": "new_tag" } # ganti tag (untuk normalisasi konsisten)
  - dedup: boolean                        # hapus duplikat
  - trim: boolean                         # rapihin spasi, koma
  - sort:
    - enabled: boolean
    - priority_groups: [[string]]         # list group tags yg diprioritaskan ordernya
  - optional_handling:
    - move_to_optional_tags: [string]     # tag tertentu dipindah ke #optional
    - move_to_optional_regex: [string]
  - keep_tags: [string]                   # tag yg wajib dipertahankan
  - background_policy:
    - enabled: boolean
    - allow_general: [string]             # contoh: indoors, outdoors, city, nature, sky
    - block_specific: [string]            # contoh: classroom, bedroom, kitchen (opsional)
    - move_specific_to_optional: boolean  # true = pindahkan ke optional, false = hapus
    - frequency_threshold: float|null     # misal 0.05 (5%) → tag jarang jadi optional

> Catatan: `keep_tags` harus menang melawan remove.

### 2.2 Contoh Preset JSON (anime)
Buat 1–2 contoh file preset sebagai referensi.

Contoh file: `presets/anime/normalize_v1.json`
{
  "name": "anime-normalize-v1",
  "description": "Remove junk tags, stabilize common tags, keep dataset neutral.",
  "version": "1.0.0",
  "rules": {
    "dedup": true,
    "trim": true,
    "remove_tags": [
      "watermark", "signature", "artist name", "copyright name",
      "jpeg artifacts", "lowres", "worst quality"
    ],
    "remove_regex": [
      "^artist:.*$",
      "^rating:.*$"
    ],
    "replace_map": {
      "from back": "from behind",
      "looking back": "looking back",
      "upperbody": "upper body"
    },
    "keep_tags": [
      "solo"
    ],
    "optional_handling": {
      "move_to_optional_tags": [
        "classroom", "bedroom", "kitchen"
      ],
      "move_to_optional_regex": []
    },
    "sort": {
      "enabled": true,
      "priority_groups": [
        ["character_name", "series_name"],
        ["solo"],
        ["upper body", "cowboy shot", "full body", "close-up"],
        ["from above", "from below", "from behind"],
        ["day", "night", "indoors", "outdoors"]
      ]
    },
    "background_policy": {
      "enabled": true,
      "allow_general": ["indoors", "outdoors", "city", "nature", "sky", "sea", "forest"],
      "block_specific": ["classroom", "bedroom", "kitchen", "train interior"],
      "move_specific_to_optional": true,
      "frequency_threshold": 0.05
    }
  }
}

> Placeholder `character_name`/`series_name` di priority_groups: implementasi boleh treat ini sebagai “tag apa pun yang user tandai sebagai identity tags” (lihat section 4).

---

## 3) Engine: Parser & Writer Tag File
### 3.1 Format tag file yang didukung
- Default: satu baris `tag1, tag2, tag3`
- Dukung blok tambahan:
  - `#optional: tagA, tagB`
  - `#warning: text...` (boleh plain text, tidak harus tags)

### 3.2 Implementasi parsing
- [ ] Parse file `.txt`:
  - ambil main tags (comma-separated)
  - ambil optional tags jika ada
  - ambil warning text jika ada
- [ ] Simpan sebagai object:
  - main_tags: set/list
  - optional_tags: set/list
  - warning: string|null
- [ ] Writer harus:
  - output main tags baris pertama (koma+spasi)
  - jika optional_tags non-empty → tulis `#optional: ...`
  - jika warning non-empty → tulis `#warning: ...`

### 3.3 Normalization pipeline (per file)
Urutan yang direkomendasikan:
1) trim & normalize delimiter
2) apply replace_map (main + optional)
3) remove by remove_tags + remove_regex (kecuali keep_tags)
4) dedup
5) background_policy:
   - jika tag masuk block_specific:
     - jika move_specific_to_optional true → pindah ke optional
     - else → hapus
   - jika frequency_threshold aktif:
     - hitung frekuensi tag across dataset (butuh scan step)
     - tag yang jarang & bukan “protected groups” → optional
6) sort if enabled (priority_groups)
7) write back (dengan backup)

---

## 4) Konsep “Protected Tags” (agar judgement manual aman)
Tambahkan pengaturan global/app config:
- [ ] Identity tags list: misal user bisa set di UI atau config:
  - `identity_tags`: ["<character_name>", "<series_name>"] (atau pattern)
- [ ] Protected categories (opsional):
  - tags yang tidak boleh disentuh normalizer: “character/outfit/pose/lighting/body composition”
  - implement paling aman: user bisa input list “never remove tags” di UI (Extra keep tags), cukup.

Minimal wajib:
- keep_tags dari preset + Extra keep tags dari UI harus selalu menang.

---

## 5) Scan Dataset: Statistik untuk Background Policy & Quality Check
Biar tab normalisasi bisa bantu “background tagging” tanpa manual 100%, lakukan scan:
- [ ] Scan semua `.txt` untuk menghitung frekuensi tag:
  - map: tag -> count
  - total files
- [ ] Output summary:
  - top 50 tags
  - tags jarang (< threshold)
  - tags yang masuk block_specific
- [ ] Preview “apa yang akan dipindah ke optional / dihapus” sebelum apply.

---

## 6) Cara Ideal Tagging Background (aturan yang di-embed ke preset)
Tujuan: model netral, background tidak nempel.

Rekomendasi rule (implementable):
- Background general yang aman dipertahankan (main tags):
  - `indoors`, `outdoors`, `day`, `night`, `city`, `nature`, `sky`, `sea`, `forest`, `snow`
- Background spesifik yang sering bikin bias:
  - `classroom`, `bedroom`, `kitchen`, `bathroom`, `train interior`, `office`, `cafe`
  - default: pindah ke `#optional` (bukan hapus) supaya masih ada info kalau diperlukan
- Frequency-based optional:
  - tag background yang muncul sangat jarang → optional
- Jangan pernah auto-add background tag baru dari kosong:
  - normalizer hanya remove/move/replace (judgement tetap di user)

Tambahkan ke preset lewat `background_policy`.

---

## 7) Integrasi ke Flask App (Routes + Frontend)
### 7.1 Routes backend
- [ ] GET `/api/presets?type=anime` → list preset files
- [ ] GET `/api/preset?type=anime&file=normalize_v1.json` → load preset json
- [ ] POST `/api/normalize/scan` payload:
  - dataset_path, recursive, preset_type, preset_file, extra_remove, extra_keep, settings...
  - response: stats + sample preview list
- [ ] POST `/api/normalize/dryrun` payload:
  - same + preview_limit
  - response: array of `{file, before, after, changed, actions[]}`
- [ ] POST `/api/normalize/apply` payload:
  - same + backup_enabled
  - response: summary counts + list changed files (optional)

### 7.2 Frontend (tab + components)
- [ ] Tambah tab “Dataset Normalization”
- [ ] Komponen preview diff:
  - before textarea (read-only)
  - after textarea (read-only)
  - highlight changed tags (opsional)
- [ ] Progress indicator + log
- [ ] Tombol apply disabled sampai scan/dryrun sukses (safety)

---



## 9) Error Handling & Safety
- [ ] Validasi dataset_path ada & readable
- [ ] Skip file yang tidak punya pasangan gambar/txt (sesuai toggle)
- [ ] Backup policy:
  - `.txt.bak` disimpan sekali (atau timestamp)
- [ ] Transaction-ish:
  - saat apply, jika error di tengah:
    - minimal: log file terakhir
    - opsional advanced: rollback dari `.bak` untuk file yg sudah berubah

---

## 10) Acceptance Criteria (ceklist selesai)
- [ ] Preset picker berfungsi: load dari folder per type
- [ ] Extra remove/keep berjalan & override dengan benar
- [ ] Dry-run preview menunjukkan before/after untuk sample file
- [ ] Apply menghasilkan file tags rapi + backup aman
- [ ] Optional/warning format tidak rusak
- [ ] Background policy bisa memindahkan tag spesifik ke #optional
- [ ] Statistik scan membantu user mutusin threshold tanpa coba-coba

---

## 11) Catatan Implementation Detail (biar Codex tidak nyasar)
- Jangan pakai “or” di output tagging.
- Gunakan delimiter standar: `", "` (koma + spasi).
- Semua rule harus deterministic (hasil sama jika dijalankan ulang).
- Pastikan engine tidak menambah tag baru secara “mengarang” — hanya transform/remove/move.
