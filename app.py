from __future__ import annotations
import os
import sys
import time
import threading
import zlib
import numpy as np

from PySide6 import QtCore, QtWidgets

from utils_win import get_cursor_pos
from capture import CaptureRegion, capture_bgr, clamp_region_to_virtual_screen
from ocr import RapidOcrBackend, pick_hover
from ui_popup import PopupWindow

from debug_utils import save_debug_image, draw_detection

from pathlib import Path
from tools.dict_db import DictDB

BASE_DIR = Path(__file__).resolve().parent
DB = DictDB(str(BASE_DIR / "data" / "jp_dict.sqlite"))


#DICT = build_toy_dict()

# Optional: silence model source check noise for some libs
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

# ---- Config ----
HOTKEY = "<shift>"
ROI_W = 640  # wide: lookup phrases extend to the right of the cursor
ROI_H = 140
# --------------


class OcrWorker(QtCore.QObject):
    # IMPORTANT: emit only simple Qt-safe types
    finished = QtCore.Signal(object, int, int)  # payload(dict), cursor_x, cursor_y

    def __init__(self) -> None:
        super().__init__()
        self.ocr = RapidOcrBackend()
        self._lock = threading.Lock()

    @QtCore.Slot(object, int, int, int, int)
    def run(self, bgr_obj: object, cx: int, cy: int, roi_cx: int, roi_cy: int) -> None:
        """bgr_obj: raw screenshot ROI; (roi_cx, roi_cy): cursor position inside it."""
        assert isinstance(bgr_obj, np.ndarray)
        bgr = bgr_obj

        hit = None
        ocr_text = ""
        norm = ""
        hover_i = 0
        lines_out: list[str] = ["(no text)"]
        hover = None

        t0 = time.perf_counter()
        try:
            with self._lock:
                text_lines = self.ocr.detect(bgr)
                if not text_lines:
                    # Nothing found at all: retry with the slower, noise-tolerant pass.
                    text_lines = self.ocr.detect(bgr, slow=True)
                hover = pick_hover(text_lines, roi_cx, roi_cy)
                save_debug_image("02_detection", draw_detection(bgr, text_lines, hover, roi_cx, roi_cy))

            if hover is not None:
                # Dictionary matching starts AT the hovered character and shrinks
                # from the right: longest match first.
                ocr_text = hover.line.text
                norm = hover.tail
                hover_i = hover.index
                hit = DB.longest_match_from_many(norm, 0, max_len=24, limit=5)

                if hit:
                    surface, hits = hit
                    head = hits[0]
                    lines_out = [surface]

                    tag = "(name)" if head.kind == "name" else "(kanji)" if head.kind == "kanji" else ""
                    if tag:
                        lines_out.append(tag)
                    if head.reading:
                        lines_out.append(head.reading)

                    glosses = []
                    for h in hits:
                        if h.gloss and h.gloss not in glosses:
                            glosses.append(h.gloss)
                        if len(glosses) >= 3:
                            break
                    for i, g in enumerate(glosses, 1):
                        lines_out.append(f"{i}) {g}")
                else:
                    lines_out = ["(no match)", f"OCR: {norm}"]
        except Exception as e:
            lines_out = [f"(OCR error: {type(e).__name__}: {e})"]
        finally:
            t1 = time.perf_counter()

        payload = {
            "popup_text": "\n".join([ln for ln in lines_out if ln]),
            "ocr_raw": ocr_text,
            "ocr_norm": norm,
            "hover_i": hover_i,
            "latency_ms": int((t1 - t0) * 1000),
            "pred_surface": None,
            "pred_kind": None,
            "pred_reading": None,
            "pred_glosses": [],
        }
        if hit:
            surface, hits = hit
            payload.update({
                "pred_surface": surface,
                "pred_kind": hits[0].kind,
                "pred_reading": hits[0].reading,
                "pred_glosses": [h.gloss for h in hits if h.gloss][:5],
            })

        self.finished.emit(payload, cx, cy)

        print(f"[{payload['latency_ms']} ms] line={ocr_text!r} hover_i={hover_i} "
              f"tail={norm[:20]!r} match={payload['pred_surface']!r}", flush=True)


class AppController(QtCore.QObject):
    # Route hotkey actions into Qt thread safely
    request_capture = QtCore.Signal()
    request_ocr = QtCore.Signal(object, int, int, int, int)  # bgr, cx, cy, roi_cx, roi_cy
    start_scan = QtCore.Signal()
    stop_scan = QtCore.Signal()

    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__()
        self.app = app
        self.popup = PopupWindow()

        self.shift_held = False

        self._last_good_popup_text = None

        self._prev_text = None
        self._stable_count = 0

        self._no_match_start = None
        self._no_match_timeout = 1.2  # seconds

        self._last_ocr_bgr = None  # numpy array fed to OCR
        self._last_result = None  # dict from worker

        # continuous scan timer (runs on Qt thread)
        self.scan_timer = QtCore.QTimer(self)
        self.scan_timer.setInterval(80)
        self.scan_timer.timeout.connect(self._on_capture_requested)

        self.start_scan.connect(self._start_scan, QtCore.Qt.ConnectionType.QueuedConnection)
        self.stop_scan.connect(self._stop_scan, QtCore.Qt.ConnectionType.QueuedConnection)

        # simple change detection
        self._last_cursor = None  # (x,y)
        self._last_hash = None

        self.ocr_busy = False

        # Worker thread
        self.thread = QtCore.QThread()
        self.worker = OcrWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

        self.request_capture.connect(self._on_capture_requested, QtCore.Qt.ConnectionType.QueuedConnection)
        self.request_ocr.connect(self.worker.run, QtCore.Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.on_ocr_finished, QtCore.Qt.ConnectionType.QueuedConnection)

        # Hotkey listener (pynput thread)
        self.listener = None
        self._setup_hotkey()

    @QtCore.Slot()
    def _start_scan(self) -> None:
        # Reset guards so the first tick always runs
        self._last_cursor = None
        self._last_hash = None

        print("SCAN START")

        # Do an immediate capture, then start periodic scanning
        self._on_capture_requested()
        if not self.scan_timer.isActive():
            self.scan_timer.start()

    @QtCore.Slot()
    def _stop_scan(self) -> None:
        if self.scan_timer.isActive():
            self.scan_timer.stop()
        self.ocr_busy = False  # safety
        self._last_cursor = None
        self._last_hash = None
        self.popup.hide()
        print("SCAN STOP")

    def _setup_hotkey(self) -> None:
        from pynput import keyboard

        def on_press(key):
            try:
                if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    if not self.shift_held:
                        self.shift_held = True
                        self.start_scan.emit()
                elif key == keyboard.Key.f8:
                    # Save evaluation sample (go to Qt thread via invokeMethod)
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_save_sample_qt",
                        QtCore.Qt.ConnectionType.QueuedConnection
                    )
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    if self.shift_held:
                        self.shift_held = False
                        self.stop_scan.emit()
            except Exception:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def _hotkey_callback(self) -> None:
        # We are NOT on Qt thread here. Just emit a Qt signal.
        self.request_capture.emit()

    @QtCore.Slot()
    def _on_capture_requested(self) -> None:
        # If an OCR job is currently running, skip this tick.
        if self.ocr_busy:
            return

        cx, cy = get_cursor_pos()

        # --- Movement guard (cheap) ---
        if self._last_cursor is not None:
            lx, ly = self._last_cursor
            if abs(cx - lx) + abs(cy - ly) < 2:
                return
        self._last_cursor = (cx, cy)

        # Capture ROI centred on the cursor (clamped to the virtual screen).
        region = clamp_region_to_virtual_screen(
            CaptureRegion(left=cx - ROI_W // 2, top=cy - ROI_H // 2, width=ROI_W, height=ROI_H)
        )
        # Cursor position inside the (possibly clamped) capture.
        roi_cx, roi_cy = cx - region.left, cy - region.top

        # --- Temporarily hide popup so it doesn't appear in the screenshot ---
        was_visible = self.popup.isVisible()
        if was_visible:
            self.popup.setWindowOpacity(0.0)
            self.app.processEvents()

        bgr = capture_bgr(region)
        save_debug_image("01_raw_roi", bgr)

        if was_visible:
            self.popup.setWindowOpacity(1.0)
            self.app.processEvents()

        # --- Hash guard: skip if the pixels and cursor are unchanged ---
        h = (zlib.adler32(bgr.tobytes()), roi_cx, roi_cy)
        if self._last_hash == h:
            return
        self._last_hash = h

        self.ocr_busy = True
        self.popup.set_text(["Processing OCR..."])
        self.popup.show_near(cx, cy)

        self._last_ocr_bgr = bgr.copy()
        self.request_ocr.emit(bgr, cx, cy, roi_cx, roi_cy)

    @QtCore.Slot(object, int, int)
    def on_ocr_finished(self, result: object, cx: int, cy: int) -> None:
        # Safety: if something still sends a string, don't crash
        if isinstance(result, str):
            result = {"popup_text": result}
        assert isinstance(result, dict)

        self._last_result = result

        text = result.get("popup_text", "")

        self.ocr_busy = False

        # stability counter
        if text == self._prev_text:
            self._stable_count += 1
        else:
            self._stable_count = 0
        self._prev_text = text

        is_match = "(no match)" not in text and "(no text)" not in text and "(OCR error" not in text

        now = time.time()

        if is_match:
            self._last_good_popup_text = text
            self._no_match_start = None
            self.popup.set_text(text.splitlines())
            self.popup.show_near(cx, cy)
            return

        # no match path
        if self._no_match_start is None:
            self._no_match_start = now

        # If stable twice, accept the no-match (prevents flicker between two different misses)
        if self._stable_count >= 1:
            if (now - self._no_match_start) >= self._no_match_timeout:
                self._last_good_popup_text = None
                self.popup.set_text(["..."])
            else:
                if self._last_good_popup_text:
                    self.popup.set_text(self._last_good_popup_text.splitlines())
                else:
                    self.popup.set_text(["..."])
        else:
            # not stable yet: keep last good if available
            if self._last_good_popup_text:
                self.popup.set_text(self._last_good_popup_text.splitlines())
            else:
                self.popup.set_text(["..."])

        self.popup.show_near(cx, cy)

    def shutdown(self) -> None:
        try:
            if self.listener:
                self.listener.stop()
        except Exception:
            pass
        self.thread.quit()
        self.thread.wait(1500)

    def on_save_sample(self):
        # must be executed on Qt thread
        QtCore.QMetaObject.invokeMethod(self, "_save_sample_qt", QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def _save_sample_qt(self):
        import time, json, uuid
        from pathlib import Path
        from PySide6 import QtWidgets
        import cv2

        def imwrite_unicode(path: Path, bgr) -> bool:
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                return False
            path.write_bytes(buf.tobytes())
            return True

        if self._last_ocr_bgr is None or self._last_result is None:
            print("No OCR result to save yet.", flush=True)
            return

        base = Path(__file__).resolve().parent
        out_dir = base / "evals" / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        sample_id = f"{ts}_{uuid.uuid4().hex[:8]}"

        pred = self._last_result.get("pred_surface") or "NONE"
        popup_text = self._last_result.get("popup_text") or ""

        img_path = out_dir / f"{sample_id}.png"
        jsonl_path = base / "evals" / "samples.jsonl"

        # Ask GT
        gt, ok = QtWidgets.QInputDialog.getText(
            None,
            "Ground Truth",
            f"Predicted: {pred}\nEnter correct surface form (leave empty if none):"
        )
        gt = gt.strip() if ok else ""

        # Save image (Unicode-safe)
        if not imwrite_unicode(img_path, self._last_ocr_bgr):
            print(f"[EVAL] FAILED to write image: {img_path}", flush=True)
            return

        rec = dict(self._last_result)
        rec.update({
            "timestamp": ts,
            "sample_id": sample_id,
            "image_path": str(img_path).replace("\\", "/"),
            "gt_surface": gt,
        })

        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Saved sample: {img_path}  GT={gt!r}", flush=True)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # <-- ADD THIS
    ctrl = AppController(app)
    app.aboutToQuit.connect(ctrl.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
