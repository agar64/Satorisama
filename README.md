# Satorisama (悟り様)

> **Real-time optical Japanese popup dictionary for games, visual novels, and manga.**  
> Like Yomitan, but for your entire screen.

**Satorisama** (*Satori* = 悟り "comprehension / sudden awakening", also named after the mythical creature that reads minds) is a lightweight Windows desktop overlay that brings Yomitan-style hover lookups to untyped Japanese text anywhere on your screen.

---

## Features

- **Instant Hover Lookup:** Hold `Shift` while hovering your mouse over any Japanese text in a game, VN, video, or comic to see definitions immediately.
- **Sub-character Accuracy:** Uses geometric bounding box hit-testing to pinpoint the exact character under your cursor, eliminating false lookups from surrounding text.
- **Yomitan Dictionary Compatibility:** Powered by a fast local SQLite database compiled directly from standard Yomitan dictionary archives (JMdict, KANJIDIC, JMnedict).
- **Longest-Match Search:** Automatically scans compound phrases and words starting from the hovered character.
- **Fast & Local:** Powered by RapidOCR (PP-OCRv6 + ONNX Runtime) running locally on your CPU/GPU without cloud APIs.

---

## How It Works

```
[Screen Region] ──> [RapidOCR Detection] ──> [Cursor Hit-Test] ──> [SQLite DictDB] ──> [Floating Popup]
 (mss capture)       (PP-OCRv6 glyph boxes)    (find hovered char)   (longest match)     (PySide6 UI)
```

1. **Screen Capture:** While you hold `Shift`, `capture.py` captures a small region around your mouse cursor via `mss`, clamping bounds across multi-monitor virtual desktops.
2. **Character-Level OCR:** `ocr.py` feeds the ROI into RapidOCR to detect text lines and per-glyph coordinates.
3. **Hover Hit-Testing:** `pick_hover()` tests the cursor coordinates against the glyph bounding boxes to identify the exact character you are pointing at and slices the remaining text line.
4. **Dictionary Lookup:** `tools/dict_db.py` queries `data/jp_dict.sqlite` using a longest-match strategy (checking vocabulary terms, names, and single kanji entries).
5. **Overlay Popup:** `ui_popup.py` renders a non-intrusive floating dark-mode popup beside your cursor with readings, part-of-speech tags, and definitions.

---

## Prerequisites & Installation

- **OS:** Windows 10 / 11
- **Python:** 3.10+

1. Clone or download this repository:
   ```bash
   git clone https://github.com/agar64/Satorisama.git
   cd Satorisama
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Setting Up the Dictionary Data

Dictionary files are not bundled due to copyright and size. You can build your own local database from free Yomitan-format dictionaries:

1. Download the Yomitan-format zip archives for:
   - **JMdict** (English vocabulary)
   - **KANJIDIC** (Kanji definitions)
   - **JMnedict** (Japanese names, optional)
   *(e.g., from [Yomitan Dictionary Index](https://foosoft.net/projects/yomichan/#dictionaries) or community repositories).*

2. Place the `.zip` files in the `data/` directory.

3. Run the extraction builder:
   ```bash
   python tools/build_dict_db_yomitan.py ^
     --zip data/JMdict_english.zip ^
     --zip data/KANJIDIC_english.zip ^
     --names-zip data/JMnedict.zip ^
     --out data/jp_dict.sqlite
   ```

This generates `data/jp_dict.sqlite`, which `app.py` loads on startup.

---

## Usage

Run the application:
```bash
python app.py
```

- **Hover & Hold `Shift`:** Scans text under the mouse cursor and displays the lookup popup.
- **Release `Shift`:** Hides the popup and stops scanning.
- **Exit:** Close the terminal or right-click / terminate the task.

### Debugging
To inspect OCR bounding boxes and detection crops, run with:
```bash
set OCRJP_DEBUG=1
python app.py
```
Debug crops and overlays will be saved to the `debug_ocr/` folder.
