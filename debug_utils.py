from __future__ import annotations
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Debug images are only written when OCRJP_DEBUG=1 (otherwise every scan would fill the disk).
DEBUG = os.environ.get("OCRJP_DEBUG", "") == "1"
_DEBUG_DIR = Path(__file__).resolve().parent / "debug_ocr"

_counter = 0


def save_debug_image(tag: str, bgr: np.ndarray) -> None:
    global _counter
    if not DEBUG or bgr is None:
        return
    _DEBUG_DIR.mkdir(exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = _DEBUG_DIR / f"{_counter:04d}_{ts}_{tag}.png"

    img = bgr
    if img.dtype != np.uint8:
        img = img.clip(0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", img)  # unicode-safe write
    if ok:
        path.write_bytes(buf.tobytes())
    _counter += 1


def draw_detection(bgr: np.ndarray, lines, hover, cx: int, cy: int) -> np.ndarray:
    """Overlay detected glyph boxes (green), the hovered line (blue) and glyph (red)."""
    out = bgr.copy()
    for ln in lines:
        for g in ln.glyphs:
            cv2.rectangle(out, (int(g.x0), int(g.y0)), (int(g.x1), int(g.y1)), (0, 200, 0), 1)
    if hover is not None:
        x0, y0, x1, y1 = hover.line.bounds
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 120, 0), 2)
        g = hover.line.glyphs[hover.index]
        cv2.rectangle(out, (int(g.x0), int(g.y0)), (int(g.x1), int(g.y1)), (0, 0, 255), 2)
    cv2.drawMarker(out, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
    return out
