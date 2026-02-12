Berikut **instruksi Codex (saran)** untuk adjust fitur **Dataset Tag Editor** supaya:

1. **Klik tag di Tag Explorer → gambar yang punya tag itu di Preview Images (gallery) ter-highlight & naik ke urutan teratas**
2. **Di preview modal (saat gambar diklik), tag pada gambar bisa dihapus lewat tombol “X” saat hover**

**File referensi yang bakal disentuh**

* UI (Tag Editor): `templates/dataset_labeling.html` 
* Backend API: `app.py` 
* Core tag ops: `services/tag_editor.py` 
* (Opsional konteks pipeline, kalau perlu ngecek konsistensi): `pipeline.html` 

---

## 1) Jalur terbaik (1 opsi saja)

Implement **client-side focus + sort** berdasarkan `tagVizState.selected` (tag yang diklik di Tag Explorer) untuk **galleryState.items**:

* **Matched images** (punya minimal 1 selected tag) → tampil **paling atas**
* Urutan matched → **semakin banyak tag yang match semakin atas**
* Tambahkan class CSS `is-match` pada `gallery-card` untuk highlight

Lalu buat fitur **hapus tag per-image** dari preview modal:

* Render tag chip di modal sebagai `gallery-tag is-removable`
* Munculkan tombol **X saat hover**
* Klik X → call API baru `/api/tags/tag-remove` (atau nama serupa) yang:

  * cari `.txt` image tersebut
  * backup `.txt.bak` (konsisten dengan batch edit)
  * remove tag
  * return tags terbaru
* UI update state lokal + re-render modal tags + re-render gallery (agar highlight/sort ikut update)

---

## 2) Rencana implementasi (UI → backend → format data → edge case)

### A) UI: highlight + reorder gallery saat tag explorer diklik

**Target behavior**

* Saat user klik tag di Tag Explorer:

  * gallery akan **re-render** dan images yang mengandung tag tersebut:

    * punya border/outline highlight
    * pindah ke top list

**Langkah**

1. Di `dataset_labeling.html`, tambahkan helper:

   * `computeMatchScore(itemTags, selectedSet)` → integer
   * `getSortedGalleryItems()` → return array item yang sama (bukan clone object baru), tapi di-sort berdasarkan:

     1. `matchScore > 0` dulu
     2. `matchScore` desc
     3. fallback ke index awal (butuh `_idx`)

2. Saat load gallery (setelah fetch `/api/tags/images`):

   * set `item._idx = i` untuk menjaga urutan default

3. Ubah `renderGalleryItems()`:

   * gunakan `const items = getSortedGalleryItems()` daripada `galleryState.items`
   * ketika bikin `gallery-card`, set:

     * `card.classList.toggle('is-match', matchScore > 0)`

4. Di handler klik tag Tag Explorer (di `renderTagViz()` atau handler yang toggle `tagVizState.selected`):

   * setelah update selection + sync ke input, panggil:

     * `renderGalleryItems()` (cukup ini, jangan refetch)

**CSS**

* Tambahkan style:

  * `.gallery-card.is-match { outline:2px solid #2d4f86; box-shadow:0 0 0 1px rgba(...); }`
  * (opsional) ` .gallery-card.is-match .gallery-name { color: #e6f0ff; }`

> Catatan: kalau multi-select sudah ada, sorting pakai “ANY match” + score jumlah match itu enak banget dipakai (lebih berguna dari single-tag doang), dan tetap sesuai request.

---

### B) UI: hapus tag dari preview modal dengan tombol X saat hover

**Target behavior**

* Klik thumbnail image → modal preview terbuka
* Tag list di modal:

  * setiap tag punya tombol **X** muncul saat hover
  * klik X → tag hilang dari `.txt` dan UI langsung update

**Langkah**

1. Upgrade fungsi `fillGalleryTags(...)`

   * dari: `fillGalleryTags(container, tags, hasTxt, maxTags)`
   * menjadi: `fillGalleryTags(container, tags, hasTxt, maxTags, ctx=null)`
   * `ctx` berisi:

     * `ctx.removable` boolean
     * `ctx.folder` (root yang dipakai API)
     * `ctx.rel` (path relatif image)
     * `ctx.onRemoved(updatedTags)` callback (opsional)

2. Di `openGalleryPreview(item)`:

   * panggil `fillGalleryTags(previewTagsEl, item.tags, item.has_txt, 30, { removable:true, folder: galleryRoot, rel: item.rel })`

3. Render removable chip:

   * buat elemen `span.gallery-tag.is-removable`
   * di dalamnya:

     * text tag
     * `button.tag-x` (isi “×”) hidden default, show on hover via CSS
   * klik `button.tag-x`:

     * `e.stopPropagation()`
     * call `apiRemoveTag(folder, rel, tag)`
     * kalau sukses:

       * update `item.tags = updatedTags`
       * re-render modal tags (panggil fillGalleryTags lagi)
       * re-render gallery items (biar highlight/sort ikut berubah)
       * update Tag Explorer count (lihat bagian C)

**CSS untuk X**

* `.gallery-tag.is-removable { position:relative; padding-right:1.1rem; cursor:default; }`
* `.gallery-tag .tag-x { display:none; position:absolute; right:.2rem; top:50%; transform:translateY(-50%); border:0; background:transparent; color:inherit; opacity:.7; }`
* `.gallery-tag.is-removable:hover .tag-x { display:inline; opacity:1; }`

---

### C) UI: update Tag Explorer counts tanpa rescan berat

Setelah remove tag dari 1 gambar:

* cari item tag itu di `tagVizState.items`
* `count -= 1`
* kalau `count <= 0` → remove dari list
* re-render Tag Explorer (fungsi render yang sudah ada)

Ini bikin UI langsung konsisten tanpa scan ulang folder.

---

## 3) Backend API: endpoint remove-tag (minimal & aman)

Tambahkan endpoint baru di `app.py` (dekat block `/api/tags/...`) 

**Contract**

* `POST /api/tags/tag-remove`
* JSON:

  ```json
  { "folder": "D:\\dataset", "rel": "sub/a.png", "tag": "blue_hair", "backup": true }
  ```
* Response sukses:

  ```json
  { "ok": true, "rel": "...", "tag": "...", "removed": true, "tags": ["..."] }
  ```

**Rules**

* Validasi:

  * folder wajib & exists
  * rel tidak boleh absolute / traversal (pakai `_safe_child`)
  * resolve ke file image (atau `.txt` kalau kamu izinkan rel `.txt`)
* Tentukan txt:

  * kalau rel adalah image → `txt = image.with_suffix(".txt")`
  * kalau `txt` tidak ada → return `{ok:false, error:"Missing .txt"}`
* Call helper di `services/tag_editor.py`

---

## 4) Core: helper remove_tag di `services/tag_editor.py`

Tambahkan fungsi baru (standalone) 

**Perilaku**

* Read src
* `tags = split_tags(src)`
* Remove semua occurrence `tag` (match exact)
* Backup: `txt.with_suffix(txt.suffix + ".bak")` (konsisten dengan batch mode)
* Write: `join_tags(newtags)`
* Return updated tags list

> Keep it simple & konsisten dengan existing insert/delete/replace/dedup yang sudah pakai `split_tags/join_tags` dan `.bak`.

---

## 5) Edge cases yang wajib ditangani

* **`.txt` missing** → UI tampilkan error di status modal / gallery status (jangan crash)
* **tag tidak ada di file** → return ok true, removed false, tags tetap (biar UI tetap sinkron)
* **case mismatch** → kalau project-mu sudah normalize ke lower, aman; kalau belum:

  * server bisa pakai exact match (paling aman, predictable)
* **duplicate tag** → remove semua duplicates (filter)
* **file permission/encoding error** → return ok false error string

---

## 6) Quick testing checklist (manual + expected)

1. Load folder, gallery tampil.
2. Klik tag “X” di Tag Explorer
   ✅ gallery re-order: matched naik ke atas
   ✅ matched punya outline/highlight
3. Klik tag lain (multi-select)
   ✅ matched = ANY selected tags
   ✅ sorting lebih atas untuk image dengan match lebih banyak
4. Klik salah satu image → modal terbuka
   ✅ tag list muncul, tombol **X** muncul saat hover
5. Klik **X** pada salah satu tag
   ✅ tag hilang dari list modal
   ✅ `.txt.bak` tercipta
   ✅ gallery highlight/sort update (kalau tag itu mempengaruhi match)
   ✅ Tag Explorer count untuk tag itu berkurang
6. Coba hapus tag yang tidak ada
   ✅ tidak error, UI tetap stabil
