# BatchBench - Local Batch Image & Dataset Toolkit

A tiny Flask site you can run on your own PC (**localhost**) with simple menus that wrap common batch utilities:

- **Workflow Guide**
- **Image Tools**: Image -> PNG converter, Photo Adjust, Brush Blur
- **Dataset Assembly**: Webtoon Panel Splitter, Stitch Groups, Flatten & Renumber, Combine Dataset
- **Tag Tools**: Dataset Tag Editor, Dataset Normalization, Offline Tagger (WD v3), CLIP Token Check
- **Pipeline (beta)** with reorderable step cards
- **Tag Glossary Wiki**
- **Settings** for Guided Tagging Flow configuration

*Runs on Windows, Linux, and macOS.*

---

## 0) Requirement

- Install **Python 3.11+**: <https://www.python.org/downloads/>
- Make sure Python runs from terminal (`python --version` or `python3 --version`).

---

## 1) Extract the project folder

Example locations:
- Windows: `C:\BatchBench`
- Linux/macOS: `~/BatchBench`

---

## 2) Install dependencies (one-time)

Windows:
- Run `setup.bat` (double-click or from Command Prompt).

Linux/macOS:
- Run:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

`setup` will:
- create `.venv`
- install dependencies from `requirements.txt`
- create `.env` from `.env.example` if missing

---

## 3) Start the app

Windows:
- Run `run.bat`

Linux/macOS:
- Run `./run.sh`

Open in browser:
- <http://127.0.0.1:5000/>

The launch script watches the app source for changes and reloads the local server
automatically. After updating the app, refresh the browser page.

---

## 4) MD5 checksums for project files (local)

Run with venv Python:

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
## Offline Tagger: using other models

The Offline Tagger accepts a Hugging Face repo ID (e.g. `org/model`) or a local folder path.

Requirements:
- The model must work with `AutoModelForImageClassification` (multi-label).
- Safetensors weights are required (the app loads with `use_safetensors=True`).
- A tag list file should exist: `selected_tags.csv` or `tags.csv`. If missing, the app uses `id2label`.

How to switch:
1. Open the **Offline Tagger (WD v3)** tab.
2. Set **Model ID or local path** to another model repo or local folder.
3. Run. The app will download required files into the Hugging Face cache unless **Local files only** is checked.

Tip: if you store models in this repo, put them under `models/` (ignored by git).

---

## 5) Panduan tab (menu)

Penjelasan singkat untuk tiap tab: cara pakai, parameter, dan hal yang perlu diperhatikan.

### Image Tools

#### Image -> PNG Converter
Cara pakai:
- Isi Source Folder (berisi file gambar), isi Output Folder, klik Convert.
Parameter:
- Source Folder: folder berisi file gambar yang didukung Pillow (hanya level paling atas).
- Output Folder: folder hasil `.png` (dibuat otomatis jika belum ada).
Perhatian:
- Tidak recursive (hanya level folder teratas).
- Semua format yang didukung akan dikonversi ke `.png`.

#### Photo Adjust (preset)
Cara pakai:
- Pilih Source Folder dan Output Folder, pilih preset, atur suffix/limit, klik Process.
Parameter:
- Source Folder: folder gambar (`.jpg/.png/.webp/.bmp`), hanya level atas.
- Output Folder: hasil foto disimpan di sini.
- Preset: file JSON preset di root project (contoh: `preset_keep_warm_balanced.json`, `preset_greyscale.json`).
- Suffix: tambahan di nama file output (default `_adj`).
- Max files: 0 berarti semua file di folder.
Preset mengacu ke `utils/image_ops.py` dengan range:
- `exposure_ev` (-4..4).
- `brightness`, `contrast`, `highlights`, `shadows`, `saturation`, `warmth`, `tint`, `sharpness` (-1..1).
- `vignette` (>= 0).
Perhatian:
- Output selalu JPEG dengan quality 95 dan ekstensi `.jpg`.
- Tidak recursive; metadata EXIF bisa hilang karena re-save.

#### Brush Blur
Cara pakai:
- Isi Folder, klik Load Images, pilih gambar, lalu klik Open.
- Pilih Gaussian Blur, Mosaic, atau Box Blur. Atur Brush Size, Strength, dan Feather.
- Paint area yang ingin diproses, klik Preview Result, lalu Apply & Save jika hasilnya sudah benar.
Parameter:
- Output: `overwrite` menimpa file atau `copy` menyimpan salinan.
- Create `.bak`: membuat backup sebelum overwrite.
- Paint / Erase, Undo / Redo, dan Clear Mask: kontrol mask manual.
Perhatian:
- Gunakan Preview Result sebelum menyimpan.
- Tool ini memproses satu gambar per kali dan cocok untuk touch-up area tertentu.

### Dataset Assembly

#### Webtoon Panel Splitter
Cara pakai:
- Isi Source folder, atur stripe detection, klik Split panels.
Parameter:
- Source folder: folder chapter atau root yang berisi subfolder chapter.
- Output folder: default `_panels` per folder. Jika diisi, hasil ke `<output>\<chapter>\_panels`.
- Filename glob + Extensions: filter file halaman.
- Width alignment: `match-width` resize ke lebar terbesar; `none` hanya pad (background putih).
- Stripe detection:
  - Min stripe height: tinggi minimal garis putih untuk jadi pemisah.
  - Row white threshold: 0-255, seberapa putih sebuah pixel.
  - Row coverage: persentase pixel putih dalam satu baris.
  - Tolerance inside stripe: toleransi baris gelap di dalam stripe.
  - Min panel height: potongannya harus setinggi ini.
- Options: save strip, overwrite, dry run.
Perhatian:
- Gambar disusun vertikal tanpa gap, lalu dipotong berdasarkan stripe putih.
- Jika banyak noise, turunkan row coverage atau threshold.

#### Stitch Groups
Cara pakai:
- Isi Source folder, atur opsi, klik Merge.
Parameter:
- Source folder: berisi file dengan pola `<prefix>_<number>.<ext>`.
- Filename glob: filter awal, misalnya `*_*.*`.
- Extensions to include: daftar ekstensi yang diikutkan.
- Stack direction: `Vertical` (atas ke bawah) atau `Horizontal` (kiri ke kanan).
- Resize rule: `auto` menyesuaikan lebar/tinggi sesuai arah; `none` hanya pad; opsi lain memaksa match-width/height.
- Gap (px) dan Background: jarak antar gambar dan warna kanvas.
- Skip single / Reverse order / Overwrite / Dry run: kontrol output.
Perhatian:
- Hanya file yang namanya cocok `<prefix>_<number>` yang diproses; urutan berdasarkan angka.
- Output selalu PNG dan disimpan di `<source>\combined` jika Output kosong.

#### Flatten & Renumber
Cara pakai:
- Isi Root Folder, atur numbering dan ordering, klik Run Renamer.
Parameter:
- Root Folder: berisi gambar dan subfolder level pertama.
- Output Folder: default `<root>\_renamed`.
- Image extensions: daftar ekstensi yang diproses (gunakan format `.png,.jpg`).
- Start at / Prefix pad / Suffix pad / Separator: mengatur format `001.png`, `002_1.png`, dst.
- Ordering: urutan top-level, folder, dan isi folder (name natural, ctime, mtime).
- Rename matching `.txt`: ikut menyalin/rename file `.txt` pasangan.
- Move instead of copy: pindahkan file (hapus dari sumber).
- Dry run: hanya tampilkan rencana, tidak menulis file.
Perhatian:
- Hanya memproses top-level dan subfolder level pertama.
- Nama unik dijamin; jika bentrok, ditambah `-1`, `-2`, dst.

#### Combine Dataset
Cara pakai:
- Isi Source folders (satu folder per baris) dan Output Folder, lalu klik Combine.
Parameter:
- Source folders: daftar semua dataset sumber (minimum 2 folder, satu folder per baris).
- Suffix for non-base sets: ditambahkan ke nama file dari dataset non-base agar tidak bentrok. Jika lebih dari 2 dataset, suffix auto increment (`_B`, `_C`, `_D`).
- Image ext: ekstensi gambar yang dianggap pasangan `.txt` (gunakan format `.png,.jpg`).
- Move (not copy): jika aktif, file dipindah (bukan disalin).
Perhatian:
- Hanya memproses pasangan image+txt di level atas, tidak recursive.
- Base set dipilih dari dataset dengan pasangan terbanyak (tie ikut urutan).
- Jika ada bentrok nama, akan ditambah `_1`, `_2`, dst.

### Tag Tools

#### Dataset Tag Editor
Cara pakai (disarankan):
- Isi **Project Root** (folder induk yang berisi `database/`, `dataset/`, `dataset/_temp`, `prompt.txt`).
- Jika Project Setup muncul, jalankan **Initialize Project** dulu.
- Gunakan **Guided Tagging Flow** sebagai workflow utama untuk menandai gambar satu per satu.
- Gunakan **Bulk Tag CRUD** hanya untuk edit langsung seperti add/remove/rename/clean duplicate tags.
Parameter:
- Project Root: root proyek dataset. Tool otomatis resolve `database/`, `dataset/`, `dataset/_temp`.
- Image extensions: ekstensi gambar yang di-scan (gunakan format `.png,.jpg`).
- Tags / mapping: dipakai di Bulk Tag CRUD untuk add/remove atau rename mapping (`old->new; old2->new2`).
- Create .bak backups: membuat `.bak` untuk mode edit bulk.
Bulk Tag CRUD:
- Add Tags: tambah tag jika belum ada.
- Remove Tags: hapus tag.
- Rename Tags: ganti tag, format `old->new; old2->new2`.
- Clean Duplicates: hapus duplikat tag.
Perhatian:
- Bulk Tag CRUD berjalan di `dataset/_temp`.
- **Undo** ada sebagai tombol terpisah untuk restore file dari `_temp` ke `dataset/`.
- Tersedia tombol **Dataset zip** (zip `dataset/` tanpa `_temp`) dan **Move all from _temp** (pindahkan isi `_temp` ke folder timestamp baru).
- Add Tags bisa membuat `.txt` baru untuk gambar tanpa caption (dengan konfirmasi di UI).
- Guided Tagging Flow menyimpan sesi di `dataset/_temp/tagging_session.json` dan output akhir tetap caption `.txt` biasa.
- Tag editor tidak memahami blok `#optional:` atau `#warning:`; gunakan Dataset Normalization jika file Anda memakai format itu.

#### Dataset Normalization
Cara pakai:
- Isi Dataset folder, pilih preset type dan preset file.
- Klik Scan atau Dry-run preview, cek hasil, lalu Apply jika sudah ok.
Parameter:
- Dataset folder: folder dataset (gambar + `.txt`).
- Image extensions: ekstensi gambar yang dianggap pasangan `.txt`.
- Include subfolders: scan recursive.
- Create missing `.txt`: membuat `.txt` kosong jika ada gambar tanpa tag.
- Create backups (.bak): simpan backup sebelum menimpa.
- Normalize order: aktifkan sorting sesuai preset.
- Preset type/file: file JSON di `presets/<type>/`.
- Extra remove / keep / identity tags: daftar tambahan (comma atau newline).
- Move unknown backgrounds to `#optional:`: pindahkan tag yang tidak ada di `allow_general` ke `#optional:`.
- Background frequency threshold: override `frequency_threshold` pada preset (0..1). Kosong = pakai nilai preset.
- Preview limit: jumlah file yang ditampilkan di preview (1..500).
Format .txt yang didukung:
- `tag1, tag2, tag3`
- `#optional: tagA, tagB`
- `#warning: catatan`
Perhatian:
- Klik Scan atau Dry-run dulu; Apply baru aktif setelah itu.
- `keep_tags` + `extra_keep` + `identity_tags` selalu dipertahankan (tidak dihapus dan tidak dipindah).
- `remove_regex` memakai regex Python, case-insensitive; regex invalid diabaikan.

##### Preset file normalize_v1.json
Preset ada di `presets/anime/normalize_v1.json`. Struktur umumnya:

```json
{
  "name": "anime-normalize-v1",
  "description": "Remove junk tags, stabilize common tags, keep dataset neutral.",
  "version": "1.0.0",
  "rules": {
    "dedup": true,
    "trim": true,
    "remove_tags": ["watermark", "signature"],
    "remove_regex": ["^artist:.*$", "^rating:.*$"],
    "replace_map": {"upperbody": "upper body"},
    "keep_tags": ["solo"],
    "optional_handling": {
      "move_to_optional_tags": ["classroom", "bedroom"],
      "move_to_optional_regex": []
    },
    "sort": {
      "enabled": true,
      "priority_groups": [
        ["character_name", "series_name"],
        ["solo"]
      ]
    },
    "background_policy": {
      "enabled": true,
      "allow_general": ["indoors", "outdoors"],
      "block_specific": ["classroom"],
      "move_specific_to_optional": true,
      "frequency_threshold": 0.05
    }
  }
}
```

Urutan rule: trim -> replace -> remove -> dedup -> optional_handling -> background_policy -> sort.
Keterangan rules:
- remove_tags: tag persis (exact match) yang dihapus jika tidak ada di keep.
- remove_regex: regex Python untuk menghapus tag, case-insensitive.
- replace_map: ganti tag persis `old` menjadi `new`.
- keep_tags: daftar tag yang wajib disimpan di main (menang atas remove/optional/background).
- optional_handling: pindahkan tag tertentu ke blok `#optional:`.
- sort: jika enabled dan Normalize order aktif, urutkan tag dengan priority_groups lalu alfabet.
- background_policy:
  - allow_general: tag umum yang aman tetap di main.
  - block_specific: tag spesifik yang akan dihapus atau dipindah ke optional.
  - move_specific_to_optional: jika true, block_specific dipindah ke optional (bukan dibuang).
  - frequency_threshold: tag dengan frekuensi lebih kecil dari nilai ini (jumlah/total file) akan dipindah ke optional, kecuali ada di allow_general.
- Jika Anda menambah preset baru, simpan di `presets/<type>/` dan klik Reload di UI.

#### Offline Tagger (WD v3)
Cara pakai:
- Isi Dataset folder dan mulai dengan Output profile **Background + pose only (recommended)**.
- Aktifkan Preview only, klik Run tagger, lalu cek ringkasan kept / dropped sebelum menulis file.
- Gunakan `append`, `overwrite`, atau `skip` setelah hasil preview sesuai kebutuhan.
Output profile:
- `Background + pose only (recommended)`: simpan scene, place, object, pose, dan limb action; buang clothing, appearance, character name, rating/meta, serta unknown tag untuk hasil yang lebih bersih.
- `Standard full tags`: perilaku lama dengan output tag luas; Tag focus dan Include character/rating aktif kembali.
- `Custom selective`: pilih bucket sederhana seperti background, objects, pose, appearance, clothing, character names, dan rating/meta.
Parameter utama:
- Device: `auto` memakai CUDA jika tersedia, fallback ke CPU.
- Batch size: semakin besar semakin cepat, tapi butuh memori lebih.
- Trigger tag: selalu dipasang pertama dan tidak dihapus.
- Write mode: `append` menambah ke tag lama, `overwrite` menimpa, `skip` mengabaikan file yang `.txt`-nya sudah berisi.
- Preview only + Preview limit: tampilkan contoh dan ringkasan filter tanpa menulis file.
- Image limit: batasi jumlah gambar (0 = semua).
- Local files only: jangan download model; gagal jika cache kosong.
Advanced:
- Threshold mode `mcut` direkomendasikan untuk WD tagger; `fixed` memakai threshold statis.
- General/Character threshold, MCUT tuning, max tags, non-character regex, dan Danbooru safe-net tersedia untuk tuning lanjutan.
- Danbooru safe-net dapat mengecek unknown selective tags secara online, tetapi lebih lambat dan membutuhkan internet.
Perhatian:
- Tag ditulis ke `.txt` di samping gambar.
- Selective profile mengutamakan output bersih; beberapa fringe tag yang berguna mungkin ikut terlewat.
- Jika ada `#optional:` atau `#warning:` di `.txt`, blok tersebut dipertahankan.
- Urutan file diproses berdasarkan path (sorted).

#### CLIP Token Check
Cara pakai:
- Isi Dataset Folder, pilih mode, lalu klik Scan Tokens.
- Review caption dengan token count tertinggi atau yang melewati Token Limit.
Parameter:
- Image Extensions: hanya pasangan image + `.txt` dengan ekstensi ini yang di-scan.
- Token Limit: batas warning, default `77`.
- Top N: jumlah caption terpanjang yang ditampilkan.
- Mode `estimate`: scan ringan tanpa tokenizer tambahan.
- Mode `exact`: memakai CLIP tokenizer dan membutuhkan `transformers`.
- Recursive scan: include subfolder.
- Include `_temp`: ikut scan staging folder bila diperlukan.
Perhatian:
- Gunakan hasil sebagai sinyal review; jangan menghapus trigger atau identity tag penting secara membabi buta.
- Jalankan lagi setelah cleanup caption untuk memastikan panjang token sudah masuk akal.

### Pipeline (beta)
Cara pakai:
- Isi Dataset Source, Working Directory, Output Directory.
- Atur Image extensions + Include subfolders + Working copy mode.
- Susun step via step cards (drag/reorder), lalu klik Run Pipeline.
Parameter:
- Dataset Source Folder: dataset asli yang akan disalin ke working folder.
- Working Directory: workspace job (`<working>/jobs/<job_id>/`).
- Output Directory: lokasi zip final hasil pipeline.
- Image extensions + Include subfolders: kontrol file yang diikutkan.
- Working copy mode: `copy`, `hardlink`, atau `incremental`.
- Step config diisi per kartu step (Offline Tagger, Tag Editor, Normalization, Zip, dan step utilitas lain).
Alur kerja:
- Prepare working copy -> jalankan urutan step sesuai kartu -> update artifacts + log.
Perhatian:
- Pipeline tidak mengubah dataset asli; semua perubahan di working folder.
- `Zip result` harus di step terakhir.
- Step image-only harus diletakkan sebelum step yang mengubah tag.
- Step **Dataset Tag Editor** di pipeline sekarang sinkron dengan workflow `_temp` (staging -> edit -> restore) untuk mode non-manual.
- File state pipeline disimpan di `_work/pipeline_jobs/<job_id>/state.json`.

### Tag Glossary Wiki
Cara pakai:
- Tambah dan kategorikan tag yang ingin disimpan sebagai referensi reusable.
- Pilih tag collection untuk memuat artikel wiki, guidance singkat, related tag, dan reference image dari Danbooru.
- Gunakan linked tag untuk membandingkan pilihan lalu tambahkan tag yang berguna ke glossary.
Perhatian:
- Lookup artikel dan reference image dimuat dari Danbooru saat diperlukan, jadi koneksi internet dibutuhkan untuk fetch pertama.
- Glossary tersinkron dengan quick picker di Dataset Tag Editor pada sesi browser yang sama.

### Settings
Settings dipakai untuk mengatur **Guided Tagging Flow** tanpa edit JSON manual.
Cara pakai:
- Tambah atau edit step card. Satu step adalah satu pertanyaan review, misalnya Body Composition atau Camera Angle.
- Pilih `Single choice` jika satu jawaban harus mengganti tag lain dalam step yang sama. Pilih `Multi choice` jika beberapa tag boleh hidup bersama. Pilih `Manual tagging` untuk step yang membutuhkan editor caption dan glossary quick picker.
- Atur default queue, Required, Auto advance, Allow Not Applicable, dan daftar tag.
- Flow boleh mencampur step pilihan dan step `Manual tagging`. Gunakan queue `Not reviewed` atau `Missing only` untuk satu pass manual per gambar.
- Saat Guided Tagging Flow terbuka, tekan `Ctrl+Alt+C` untuk menampilkan cheat sheet di atas alur tagging. Pada step `Manual tagging`, overlay menampilkan gambar tagging aktif di sisi kanan; klik tag cheat sheet untuk menambahkannya langsung ke caption pending.
- Input `Manual tagging` mengingat tag yang pernah dipakai dan menampilkan autosuggest dari riwayat serta glossary. Posisi gambar terakhir disimpan per project, area, dan manual step agar review dapat dilanjutkan. Flow meminta konfirmasi sebelum membuang perubahan manual yang belum disimpan.
- Review Global Behavior, pertahankan default target `dataset/_temp`, lalu klik Save config.
- Opsional: klik Create steps untuk membuat draft editable dari section `prompt.txt`.
Perhatian:
- Cheat sheet adalah vocabulary/reference; Guided Tagging Flow adalah workflow. Import dari cheat sheet tidak tersimpan sampai Save config diklik.
- Ubah step ID dengan hati-hati setelah review berjalan karena ID dipakai oleh metadata review.
- Export JSON sebelum perubahan besar jika ingin menyimpan backup konfigurasi.

---

## Tips

- If a tool says **"folder not found,"** copy & paste the full Windows path, e.g.:

  ```
  D:\_Training DATA\MySet
  ```

  You can also **drag a folder** into the text box to paste its path.

- **Back up important data** before bulk operations.
