# Codex Instruction (Saran) — Fix Dataset Tag Editor: DnD file, Insert tidak masuk, _temp permanent, Single Log Window

> Catatan: instruksi ini **hanya saran implementasi**. Silakan kamu (Codex) adaptasi sesuai struktur & konteks project yang kamu lihat di repo, karena hanya kamu yang “lihat full context” (routing, template wiring, dan state UI).

## Target masalah yang harus dibereskan
1) **Preview window (gallery) tidak bisa drag & drop file** ke folder lain (harus bisa move file via UI).
2) **Mode `insert` tidak benar-benar masuk ke dataset** (sementara `delete/replace` jalan).
3) **Workdir `_temp` wajib jadi permanent logic**:
   - User tetap pilih **folder root** saja.
   - Workdir selalu `root/_temp` (bukan knob/setting).
   - Jika `undo`, data di workdir dikembalikan ke root folder.
4) **Log window hanya 1**:
   - Log milik offline tagger (global log dari redirect) **jangan tampil** di tab Dataset Tag Editor yang sudah punya log sendiri.

---

## 0) Repro cepat (wajib dilakukan Codex dulu)
- Buka tab **Dataset Tag Editor**
  - Coba `insert` dengan beberapa tag → pastikan sekarang tag benar-benar tertulis pada `.txt`.
  - Coba drag salah satu item preview (gallery) ke folder tree → harus move file (+ pasangan `.txt`) ke folder target.
- Buka tab **Offline Tagger**
  - Jalankan tagging (POST non-AJAX, redirect).
  - Pindah tab **Dataset Tag Editor** → pastikan **log offline tagger tidak ikut terlihat**.

---

## 1) Backend — enforce `_temp` sebagai workdir permanen + fix insert (root cause)
### A. Edit `services/tag_editor.py`
**Root cause insert “tidak masuk”:**
- Di `handle()`, untuk `edit_target == "recursive"`, staging (`do_stage`) cuma aktif untuk `delete/replace`.
- Akibatnya `insert` (tanpa stage) akan mengedit **hanya file yang sudah ada di `_temp`** → user merasa “tidak terjadi apa-apa” pada dataset utama.

**Perubahan yang disarankan:**
1) **Hilangkan knob `temp_dir`**:
   - Jangan baca `temp_dir_input` dari form.
   - Selalu derive: `temp_folder = base_folder / "_temp"`.

2) **Workdir permanen**:
   - Abaikan `edit_target` dari form (atau paksa internal ke `"recursive"`).
   - Untuk semua `EDIT_MODES` (`insert/delete/replace/dedup`):
     - Base folder wajib ada
     - Temp folder dibuat jika belum ada
     - Jalankan staging sesuai mode (supaya edit selalu terjadi di `_temp`)

3) **Staging untuk `insert` dan `dedup` juga**:
   - Ubah `do_stage = stage_only or mode in {"delete","replace"}` menjadi:
     - `do_stage = True` untuk semua `mode in EDIT_MODES` (atau minimal tambah `insert` + `dedup`)
   - Tambahkan kriteria staging untuk `dedup`:
     - `should_move = (len(taglist) != len(set(taglist)))`

4) **(Nice) Exclude `_temp` dari scan/list ketika scanning root secara recursive**
   - Di `scan_tags()` dan `list_images_with_tags()`:
     - Jika folder root dan `recursive=True`, pass `exclude_dir = folder / "_temp"` jika exist.
   - Tujuannya: Tag Explorer & scan tidak “kotor” oleh isi workdir.

**Patch arah (pseudo-diff)**
- Di awal `handle()`:
  - Hapus `temp_dir_input`
  - Paksa:
    - `base_folder = raw_folder.parent if raw_folder.name.lower() == "_temp" else raw_folder`
    - `temp_folder = base_folder / "_temp"`
    - `edit_target = "recursive"` (optional tapi recommended)

- Di branch `elif mode in EDIT_MODES:`:
  - jangan lagi bedakan `base/temp/recursive` dari form
  - selalu:
    - `temp_folder.mkdir(..., exist_ok=True)`
    - lakukan staging block untuk `insert/delete/replace/dedup` (dengan skip rules jika input kosong)
    - jika `stage_only`: return log
    - `scan_folder = temp_folder`

---

## 2) Backend — API untuk move file (DnD gallery → folder)
Saat ini API `/api/tags/move` hanya untuk **folder** (dir). DnD file butuh endpoint baru.

### A. Edit `app.py`: tambah endpoint baru `/api/tags/move-file`
**Kontrak endpoint (saran):**
- Method: `POST /api/tags/move-file`
- Payload JSON:
  - `folder`: root dataset (hasil pilih user)
  - `src`: path rel file sumber (boleh termasuk `_temp/...`)
  - `dst`: path rel folder tujuan ("" berarti root)
- Behavior:
  - Validasi path traversal (pakai `_bad_rel`, `_safe_child`, `_safe_child_or_root`)
  - Pastikan `src` itu **file**, bukan folder
  - Pindahkan:
    - file image
    - pasangan `.txt` jika ada
    - pasangan `.txt.bak` atau `.bak` jika kamu memang generate backup (opsional tapi nice)
  - Jika nama bentrok di tujuan:
    - bump nama `stem_1`, `stem_2`, dst… dan rename pasangan `.txt` mengikuti nama final

**Saran minimal bump logic:**
- Kalau `dest_img` exists → bump stem
- Move image + sidecar mengikuti stem yang sama

### B. (Opsional tapi bagus) Hide `_temp` dari folder tree listing
- Di `/api/tags/dirs`, skip folder bernama `_temp` by default:
  - Tambah payload flag `hide_temp` default True
  - Jika True → `if d.name.lower() == "_temp": continue`
- Ini mencegah user “mindahin file balik ke _temp” secara tidak sengaja.

---

## 3) Frontend — Dataset Tag Editor: enforce `_temp`, DnD file, dan UX sinkron
### A. Edit `templates/dataset_labeling.html`
#### 1) Hilangkan knob “Edit target”
- UI: remove `<select id="edit-target"...>`
- Replace dengan info statis:
  - “Workdir is fixed: `<root>/_temp`”
- Kalau masih perlu kirim value ke backend:
  - pakai `<input type="hidden" name="edit_target" value="recursive">`
  - tapi backend tetap mengabaikan dan memaksa rule sendiri (future proof)

#### 2) Gallery harus mempreview workdir `_temp` (bukan root)
- Ubah `getGalleryRoot()`:
  - dari `return stripTemp(folder)`
  - menjadi `return defaultTempFor(stripTemp(folder))`
- Pastikan `loadGallery()` memakai folder tersebut.
- Implikasi: item `rel` dari gallery akan relatif terhadap `_temp`. Untuk move-file API, kamu bisa kirim `src = "_temp/" + rel`.

#### 3) Workdir tree = folder tree tujuan (root dataset), bukan `_temp`
- Ubah `getWorkdirRoot()`:
  - return `stripTemp(folder)` (base root)
- Saat fetch `/api/tags/dirs`, kirim `hide_temp: true`.

#### 4) Implement Drag & Drop untuk **file** (gallery card → folder row/root drop)
Saat ini DnD handler di tree fokus untuk folder→folder.
Tambahkan:
- Gallery card dibuat draggable:
  - `card.draggable = true`
  - `dragstart`: set data transfer khusus (disarankan):
    - `e.dataTransfer.setData('application/x-bb-file', item.rel)`
    - juga set fallback `text/plain` misalnya `file:${item.rel}`
- Folder row & root drop menerima drop:
  - Deteksi apakah drop berisi file atau folder:
    - fileRel = `getData('application/x-bb-file')` atau parse prefix `file:`
    - dirRel  = `getData('application/x-bb-dir')` atau parse prefix `dir:`
  - Jika fileRel ada:
    - panggil `moveFileToFolder(fileRel, dstDirRel)` → fetch `/api/tags/move-file`
  - Else jika dirRel ada:
    - jalankan logic lama `moveWorkdirFolder(...)`

**Saran: jangan pakai `text/plain` generik untuk dua jenis objek tanpa prefix, biar tidak ketuker.**

#### 5) Setelah move sukses
- Refresh:
  - `loadGallery()` (karena file keluar dari `_temp`)
  - `loadWorkdirRoot()` (opsional, kalau kamu mau update count/tree)
- Tulis log ringkas ke log panel dataset tag editor:
  - contoh: `Moved _temp/abc.png -> good/abc.png`

---

## 4) Single Log Window — log offline tagger jangan nongol di tab tags
Masalahnya: `index.html` punya block global:
```jinja2
{% if log %}
  <div class="card p-3 mt-3"> ... </div>
{% endif %}
````

Block ini berada **di luar tab-pane**, sehingga setelah offline tagger jalan (redirect + log), log card tetap tampil walau user pindah tab.

### Fix minimal yang future-proof

Edit `templates/index.html`:

1. **Hapus** block global log di bawah tab-content.
2. Render log card **di dalam tab-pane** masing-masing (atau minimal di pane offline tagger & tools lain yang pakai redirect).

   * Untuk Dataset Tag Editor (`tab=='tags'`), jangan render log global (karena sudah punya log sendiri).
3. Pattern rekomendasi:

```jinja2
<div class="tab-pane ... " id="tab-offline-tagger">
  {% include "offline_tagger.html" %}
  {% if log and tab=='offline_tagger' %}
    <div class="card p-3 mt-3">
      <h5>Log</h5>
      <pre class="monospace">{{ log }}</pre>
    </div>
  {% endif %}
</div>
```

Dengan ini:

* Log offline tagger hanya terlihat saat tab offline tagger aktif.
* Saat user pindah tab tags, card-nya hidden karena berada di pane yang tidak aktif.

---

## 5) Edge cases yang wajib ditangani

* Move file harus ikut pindahkan `.txt` pasangan (kalau ada).
* Kalau `.txt` tidak ada → tetap boleh move image (log warning).
* Bentrok nama di folder target:

  * bump name dan pastikan `.txt` ikut rename sesuai bump.
* Jangan izinkan move ke path invalid (.., absolute).
* `_temp` tidak ada:

  * auto-create saat diperlukan (edit/stage), tapi hindari membuat `_temp` ketika cuma browse folder tree kalau tidak perlu.

---

## 6) Checklist testing manual (cepat)

### A) Insert fix

1. Pilih folder root dataset (yang punya images + txt).
2. Mode `insert`, isi tag baru.
3. Klik **Run** (tanpa Stage).
   ✅ Expected:

* Log menunjukkan staging + edit terjadi.
* File yang butuh tag masuk ke `_temp`.
* `.txt` di `_temp` benar-benar ketambahan tag.

### B) DnD move file

1. Pastikan gallery menampilkan isi `_temp`.
2. Drag satu card image dari gallery → drop ke folder tree target.
   ✅ Expected:

* File image dan `.txt` berpindah ke folder target.
* Item hilang dari gallery (karena keluar dari `_temp`).

### C) Undo

1. Pastikan masih ada file di `_temp`.
2. Klik **Undo Move**.
   ✅ Expected:

* Semua pasangan (image+txt) di `_temp` pindah balik ke root folder.

### D) Single log window

1. Jalankan Offline Tagger sampai muncul log.
2. Pindah tab Dataset Tag Editor.
   ✅ Expected:

* Log offline tagger tidak terlihat di tab tags (hanya log internal tag editor yang ada).

---

## Output yang harus kamu hasilkan (Codex)

* Patch minimal di:

  * `services/tag_editor.py`
  * `app.py`
  * `templates/dataset_labeling.html`
  * `templates/index.html`
* Jangan tambah dependency berat.
* Pastikan log jelas: “moved X files”, “edited X files”, dan error per-file jika gagal.