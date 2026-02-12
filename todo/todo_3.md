# Instruksi Codex — Audit + Optimasi Offline Tagger (Batchbench) + Trigger Tag Field

Kamu adalah **code auditor + implementor** untuk project “Batchbench”. Tugasmu: **analisa code yang ada**, lalu **upgrade modul offline tagger** agar:
- lebih cepat (performanya naik nyata)
- output tag lebih bersih/akurat
- minim tag “ngawur”
- tetap kompatibel dengan pipeline yang ada (backward compatible)
- **tambahkan fitur `trigger_tag`**: sebuah field yang kalau diisi akan selalu di-*insert* ke file `.txt` sebagai **tag pertama** (paling awal)

---

## A) File yang WAJIB dibaca dulu (urut)
1) `offline_tagger.py` (inti fitur offline tagger)
2) `pipeline.py` (pipeline manager & step offline tagger)
3) `normalizer.py` (normalisasi tag & kebijakan remove/dedup/sort)
4) `tag_editor.py` (alur edit/dedup tag)
5) `registry.py` (registrasi step/config jika ada)
6) `app.py` (UI / wiring opsi ke pipeline)
7) `dataset.py` + `io.py` (helper parsing path & write output)

> Kalau struktur folder berbeda, cari file dengan nama serupa / import yang relevan.

---

## B) Buat “peta pipeline” kondisi sekarang (wajib ditulis di laporanmu)
Tuliskan step-by-step alur data:
- input dataset → copy/normalize working dir → step offline tagger generate `.txt` → (manual) tag editor → normalizer apply rules → zip/export final
Sebutkan:
- step otomatis vs manual
- format output `.txt` saat ini (separator `, ` atau newline)
- titik rawan: noise tags, urutan tag berubah, dedup menghapus hal penting, replace underscore, sorting, dsb

Outputkan sebagai:
- Diagram bullet “Input → Process → Output”
- Daftar bottleneck performa + dugaan penyebab (I/O vs inference vs parsing)

---

## C) Target perubahan (ringkas tapi tegas)
Implement 2 cluster besar:
1) **Kualitas tagging**: lebih minim tag ngawur + lebih konsisten
2) **Performa**: lebih cepat secara terukur
Tambahan wajib:
3) **Trigger Tag**: field `trigger_tag` dan selalu jadi tag pertama di `.txt`

---

## D) Upgrade kualitas tagging (wajib implement)
### D1) Category mapping harus robust (jangan hardcode ngawur)
- Jangan hardcode “character/general/rating” tanpa verifikasi metadata CSV.
- Kalau model menyediakan `selected_tags.csv` / `tags.csv`, build mapping dari sana.
- Buat config override di UI/pipeline (mis: `character_category_id`, `general_category_id`, `rating_category_id`).
- Default harus aman (tidak bikin nama karakter “nyelip” ke general).

**Deliverable:** fungsi `resolve_category_ids()` yang:
- mencoba baca CSV metadata → detect kategori → return mapping
- fallback ke default mapping → log warning kalau mismatch

### D2) Filter kategori (toggle)
Tambahkan opsi:
- `include_general` (default ON)
- `include_character` (default OFF atau ON tapi harus jelas)
- `include_rating` (default OFF)
- `include_meta`, `include_copyright`, `include_artist` (default OFF)

Tujuan: user bisa mematikan kategori yang sering bikin noise untuk LoRA.

### D3) Exclude tags & exclude regex (anti-noise)
Tambahkan:
- `exclude_tags` (comma/newline)
- `exclude_regex` (newline)
Dipakai setelah scoring dan sebelum write output.

Opsional tapi bagus:
- `use_normalizer_remove_as_exclude` (default OFF)  
  kalau ON: normalizer punya remove list bisa ikut dipakai sebagai exclude offline tagger.

### D4) Threshold adaptif (mode)
Tambahkan `threshold_mode`:
- `fixed` (pakai `general_threshold` dan `character_threshold`)
- `mcut` (adaptif per-image; tujuan mengurangi “kepanjangan” dan noise)
Sediakan `min_threshold_floor` sebagai safety.

### D5) Tag budget per kategori
Tambahkan:
- `max_general_tags` (default 30)
- `max_character_tags` (default 5)
- `max_meta_tags` (default 10)
Tujuan: output ringkas & konsisten.

### D6) Format tag (underscore vs spasi) harus konsisten
Buat opsi:
- `replace_underscore` (default OFF untuk gaya danbooru)
Pastikan tidak merusak token khusus / emoticon.

---

## E) Optimasi performa (wajib ada peningkatan terukur)
Implement minimal 2:
### E1) Backend ONNX opsional
- Opsi `backend = "transformers" | "onnx"`
- Jika ONNX file tersedia, gunakan onnxruntime.
- Jika tidak ada, fallback transformers + warning log.

### E2) CUDA AMP opsional (kalau GPU)
- Opsi `use_amp = true/false` (default false)
- Pastikan output tidak “ngaco”; dokumentasikan jika ada perbedaan minor.

Tambahan opsional:
- Threaded image loading / prefetch untuk mengurangi bottleneck I/O
- Progress/logging yang tidak memperlambat loop

---

## F) FITUR BARU (WAJIB): Trigger Tag field + tag pertama di .txt
### F1) Definisi
Tambahkan field config bernama **`trigger_tag`** (string).
- Jika `trigger_tag` kosong / None → tidak melakukan apa-apa
- Jika diisi → **selalu ditaruh sebagai TAG PERTAMA** pada `.txt` output

### F2) Aturan perilaku trigger_tag (harus dipatuhi)
1) `trigger_tag` harus muncul di `.txt` **sebagai tag paling awal** (posisi index 0).
2) Tidak boleh terhapus oleh:
   - dedup tags
   - sorting tags
   - normalizer cleaning
3) Jika tag lain kosong (mis. semua di-filter) → file `.txt` tetap berisi `trigger_tag` saja.
4) Jika `trigger_tag` juga ada di hasil prediksi model:
   - tetap hanya muncul sekali
   - dan harus tetap di posisi pertama

### F3) Implementasi teknis yang harus kamu lakukan
- Tambah field ke config step offline tagger:
  - UI form (di `app.py` atau file UI terkait)
  - registry/config schema (kalau ada)
  - pipeline step config forwarding (di `pipeline.py`)
- Update fungsi write `.txt`:
  - sebelum join tags, lakukan `tags = [trigger_tag] + [t for t in tags if t != trigger_tag]`
  - lalu apply join (`, `) atau format yang digunakan
- Update normalizer agar “pinned tag” tidak tergeser:
  - Jika normalizer melakukan sort/dedup, buat aturan: `trigger_tag` selalu dipertahankan sebagai first token
  - Kalau normalizer tidak menyentuh urutan, tetap aman tapi kamu wajib pastikan lewat test
- Tambahkan opsi lanjutan (opsional tapi bagus):
  - `pinned_tags` list (future-proof), tapi untuk sekarang cukup `trigger_tag`

### F4) Update docs/log
- Saat run offline tagging, log:
  - trigger_tag aktif: YES/NO
  - contoh output 1 file (debug mode)

---

## G) Integrasi ke pipeline (wajib)
- `pipeline._step_autotag_offline()` harus meneruskan semua opsi baru:
  - backend, threshold_mode, include_* toggles, exclude_tags/regex, budgets, replace_underscore, category ids, trigger_tag
- Pastikan job config lama tetap jalan (default values aman).

---

## H) Acceptance criteria (wajib)
### H1) Kualitas
- Tag “ngawur” turun:
  - kategori yang dimatikan tidak muncul
  - exclude list/regex efektif
- Output tidak “kepanjangan”:
  - rata-rata tag count turun ke range wajar (mis 15–40 tergantung budget)
- Trigger tag:
  - selalu tag pertama
  - tidak hilang walau normalizer dipakai

### H2) Performa
Buat benchmark minimal (laporkan hasil):
- Dataset sample N=200 gambar (atau yang tersedia)
- Bandingkan:
  - transformers+fixed vs transformers+mcut
  - (kalau ada) onnx+fixed vs transformers+fixed
Catat:
- total waktu
- gambar/detik
- penggunaan VRAM/CPU (perkiraan boleh)

---

## I) Testing minimal (wajib ada)
Buat unit tests yang tidak butuh download model:
1) Test build tags dari dummy `probs/labels/categories`:
   - kategori filtering jalan
   - exclude tags/regex jalan
   - budget per kategori jalan
2) Test trigger_tag rules:
   - trigger_tag jadi index 0
   - tidak duplikat
   - tetap index 0 walau sorting/dedup dinyalakan
3) Test normalizer interop:
   - kalau normalizer sort/dedup, pastikan trigger_tag tetap di depan

---

## J) Deliverables (yang harus kamu kirim sebagai output kerja)
1) Patch/commit yang jelas:
   - “Fix category mapping”
   - “Add anti-noise filtering”
   - “Add threshold_mode fixed/mcut”
   - “Add backend onnx option”
   - “Add trigger_tag pinned as first tag”
   - “Wire options through pipeline/UI”
   - “Add tests”
2) Catatan migrasi singkat:
   - field baru apa saja
   - default values
   - contoh config minimal yang kompatibel dengan config lama

---

## K) Cara kamu harus melapor (format wajib)
1) Ringkasan temuan audit (3–10 bullet)
2) Peta pipeline (Input → Process → Output)
3) Daftar perubahan + alasan
4) Benchmark sebelum vs sesudah
5) Checklist acceptance criteria (pass/fail)
6) Link/daftar file yang berubah

> Mulai dengan membaca semua file di bagian A sebelum coding. Jangan “mengarang” fitur yang tidak ada — kalau butuh buat field baru, jelaskan wiring-nya.
