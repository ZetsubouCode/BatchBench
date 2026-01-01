# BatchBench — Local Batch Image & Dataset Toolkit

A tiny Flask site you can run on your own PC (**localhost**) with simple menus that wrap common batch utilities:

- **WebP → PNG converter**
- **Batch photo adjust** using presets (warmth/tint/brightness/etc.)
- **Dataset tag editor** (insert / delete / replace / move / dedup for `.txt` beside images)
- **Append suffix** to all files in a folder
- **Reorder** paired *(image + .txt)* names across numbered subfolders

*Designed for Windows beginners.*

---

## 0) You’ll need Python

- Download **Python 3.11 or newer** from <https://www.python.org/downloads/windows/>
- During install, **check** “Add python.exe to PATH”.

**(Optional) Verify installer integrity with MD5 on Windows**

1. Put the installer (e.g., `python-3.11.9-amd64.exe`) in your **Downloads** folder.  
2. Open **Command Prompt** and run:

```bat
certutil -hashfile "%USERPROFILE%\Downloads\python-3.11.9-amd64.exe" MD5
```

You will see an MD5 hash. Compare it with the checksum on python.org (if provided).

---

## 1) Unzip this folder somewhere simple, like `C:\BatchBench`

**The folder should contain:**

```
app.py
requirements.txt
.env.example
setup.bat
run.bat
md5sum.py
/templates
/static
/tools
```

---

## 2) (One-time) Install everything with `setup.bat`

- Double-click **`setup.bat`**  
  - Creates a virtual environment: `.venv`  
  - Installs required Python packages  
  - Prepares a `.env` file (you can edit later)

> If `setup.bat` fails because Python is not found, install Python, then run `setup.bat` again.

---

## 3) Start the site with `run.bat`

- Double-click **`run.bat`**
- Your browser should open to: <http://127.0.0.1:5000/>
- If the browser does not open automatically, open it yourself and paste the address.

---

## 4) MD5 checksums for your project files (local)

Generate MD5 for all project files to check integrity after copying:

- Double-click **`md5sum.py`**, or run:

```bat
.venv\Scripts\python md5sum.py
```

It prints MD5 for each file and also creates `checksums.md`.

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

### WebP -> PNG Converter
Cara pakai:
- Isi Source Folder (berisi file `.webp`), isi Output Folder, klik Convert.
Parameter:
- Source Folder: folder berisi file `.webp` (hanya level paling atas).
- Output Folder: folder hasil `.png` (dibuat otomatis jika belum ada).
Perhatian:
- Hanya memproses `.webp`, tidak recursive.
- Nama output memakai stem yang sama; jika sudah ada, bisa tertimpa.

### Photo Adjust (preset)
Cara pakai:
- Pilih Source Folder dan Output Folder, pilih preset, atur suffix/limit, klik Process.
Parameter:
- Source Folder: folder gambar (`.jpg/.png/.webp/.bmp`), hanya level atas.
- Output Folder: hasil foto disimpan di sini.
- Preset: file JSON preset di root project (contoh: `preset_keep_warm_balanced.json`).
- Suffix: tambahan di nama file output (default `_adj`).
- Max files: 0 berarti semua file di folder.
Preset mengacu ke `utils/image_ops.py` dengan range:
- `exposure_ev` (-4..4).
- `brightness`, `contrast`, `highlights`, `shadows`, `saturation`, `warmth`, `tint`, `sharpness` (-1..1).
- `vignette` (>= 0).
Perhatian:
- Output selalu JPEG dengan quality 95 dan ekstensi `.jpg`.
- Tidak recursive; metadata EXIF bisa hilang karena re-save.

### Dataset Tag Editor
Cara pakai (disarankan):
- Isi Folder (dataset utama) dan Temp folder (atau klik Use default).
- Pilih Mode, isi Tags/Mapping, lalu Run.
Parameter:
- Folder: base folder dataset (bukan `_temp`).
- Temp folder: lokasi folder sementara; default `<folder>\_temp`.
- Image extensions: ekstensi gambar yang di-scan (gunakan format `.png,.jpg`).
- Tags / mapping: daftar tag untuk insert/delete/move atau mapping untuk replace.
- Create .bak backups: membuat `.bak` untuk mode edit (insert/delete/replace/dedup).
Mode:
- insert: tambah tag jika belum ada.
- delete: hapus tag.
- replace: ganti tag, format `old->new; old2->new2`.
- move: pindahkan image+txt ke temp jika ada tag yang cocok.
- dedup: hapus duplikat tag.
- undo: kembalikan file dari temp ke folder utama.
Perhatian:
- Mode move berjalan di folder utama; semua mode edit berjalan di Temp (bukan di Folder).
- Undo memindahkan dari Temp ke Folder; jika nama bentrok, akan ditambah `_1`, `_2`, dst.
- Tidak recursive. Hanya memproses image yang punya pasangan `.txt`.
- Tag editor tidak memahami blok `#optional:` atau `#warning:`; gunakan Dataset Normalization jika file Anda memakai format itu.

### Offline Tagger (WD v3)
Cara pakai:
- Isi Dataset folder, atur parameter, klik Run tagger.
Parameter utama:
- Model ID or local path: Hugging Face repo ID atau path lokal model. Wajib ada file `.safetensors`.
- Device: `auto` memakai CUDA jika tersedia, fallback ke CPU.
- Batch size: semakin besar semakin cepat, tapi butuh memori lebih.
- General/Character threshold: semakin kecil, tag makin banyak (lebih noisy).
- Max tags: 0 berarti tidak dibatasi.
- Write mode: `append` menambah ke tag lama, `overwrite` menimpa, `skip` mengabaikan file yang `.txt`-nya sudah berisi.
- Include character/rating: kontrol tag karakter dan `rating:general`.
- Replace underscores: `long_hair` -> `long hair`.
- Preview only + Preview limit: hanya tampilkan contoh di log tanpa menulis file.
- Image limit: batasi jumlah gambar (0 = semua).
- Local files only: jangan download model; gagal jika cache kosong.
Perhatian:
- Tag ditulis ke `.txt` di samping gambar.
- Jika ada `#optional:` atau `#warning:` di .txt, blok tersebut dipertahankan.
- Urutan file diproses berdasarkan path (sorted).

### Dataset Normalization
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

#### Preset file normalize_v1.json
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

### Combine Dataset
Cara pakai:
- Isi Folder A, Folder B, (opsional) Extra folders, Output Folder, klik Combine.
Parameter:
- Extra folders: tambahan dataset (satu folder per baris).
- Suffix for extra sets: ditambahkan ke nama file dari dataset tambahan agar tidak bentrok. Jika lebih dari 2 dataset, suffix auto increment (`_B`, `_C`, `_D`).
- Image ext: ekstensi gambar yang dianggap pasangan `.txt` (gunakan format `.png,.jpg`).
- Move (not copy): jika aktif, file dipindah (bukan disalin).
Perhatian:
- Hanya memproses pasangan image+txt di level atas, tidak recursive.
- Base set dipilih dari dataset dengan pasangan terbanyak (tie ikut urutan).
- Jika ada bentrok nama, akan ditambah `_1`, `_2`, dst.

### Flatten & Renumber
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

### Stitch Groups
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

### Webtoon Panel Splitter
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

### Pipeline (beta)
Cara pakai:
- Isi Dataset Source, Working Directory, Output Directory, pilih preset, klik Run Pipeline.
Parameter:
- Dataset Source Folder: dataset asli yang akan disalin ke working folder.
- Working Directory: lokasi kerja dan file status `.pipeline_job.json`.
- Output Directory: lokasi zip final hasil pipeline.
- Preset type/file: preset normalisasi yang dipakai.
- Image extensions + Include subfolders: kontrol file yang diikutkan.
- Run offline auto-tagging step: aktifkan/nonaktifkan auto tag (WD v3).
- Extra remove / keep / identity tags: tambahan aturan normalisasi.
- Move unknown backgrounds to `#optional:`: pindahkan tag yang tidak ada di `allow_general`.
Alur kerja:
- Salin dataset ke working folder -> auto tag (opsional) -> normalize -> pause manual -> cleanup -> zip final.
Perhatian:
- Pipeline tidak mengubah dataset asli; semua perubahan di working folder.
- Manual retag wajib: edit `.txt` di working folder lalu klik Resume.
- Pipeline memakai setting default Offline Tagger (WD v3); untuk konfigurasi khusus gunakan tab Offline Tagger.
- File status disimpan di `<working>\.pipeline_job.json`.

---

## Tips

- If a tool says **“folder not found,”** copy & paste the full Windows path, e.g.:

  ```
  D:\_Training DATA\MySet
  ```

  You can also **drag a folder** into the text box to paste its path.

- **Back up important data** before bulk operations.
