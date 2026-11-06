"""OCR backend: RapidOCR (PP-OCRv6 detector + Japanese recognizer, ONNX Runtime).

Unlike the previous MangaOCR approach (recognition only, which needed heuristic
thresholding/cropping to isolate the hovered line), this backend *detects* text
lines itself and returns per-character boxes. That lets us decide exactly which
line and which character the cursor is on from geometry, so no other line or
neighbouring character can leak into the lookup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Glyph:
    ch: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class TextLine:
    text: str
    glyphs: list[Glyph] = field(default_factory=list)
    score: float = 0.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(g.x0 for g in self.glyphs), min(g.y0 for g in self.glyphs),
            max(g.x1 for g in self.glyphs), max(g.y1 for g in self.glyphs),
        )


@dataclass(frozen=True)
class Hover:
    line: TextLine
    index: int  # index into line.glyphs / line.text of the hovered character

    @property
    def tail(self) -> str:
        """Text from the hovered character to the end of its line."""
        return "".join(g.ch for g in self.line.glyphs[self.index:])


def _box_bounds(box) -> tuple[float, float, float, float]:
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _char_weight(ch: str) -> float:
    return 0.55 if ord(ch) < 0x2000 else 1.0  # half-width vs full-width


def _split_word(text: str, box, score: float, out: list[Glyph]) -> None:
    """Split a word box into per-character boxes (used when the engine
    returns a multi-character box)."""
    x0, y0, x1, y1 = _box_bounds(box)
    if len(text) == 1:
        out.append(Glyph(text, x0, y0, x1, y1))
        return
    weights = [_char_weight(c) for c in text]
    total = sum(weights)
    x = x0
    for c, w in zip(text, weights):
        nx = x + (x1 - x0) * w / total
        out.append(Glyph(c, x, y0, nx, y1))
        x = nx


class RapidOcrBackend:
    def __init__(self) -> None:
        from rapidocr import LangRec, OCRVersion, RapidOCR

        base = {
            "Rec.lang_type": LangRec.JAPAN,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Global.log_level": "error",
            "Global.use_cls": False,  # horizontal text only; skips a model pass
        }
        # Fast path: we upscale small captures ourselves, so don't let the
        # detector blow them up to 736px (that alone cost ~1s per scan).
        self._fast = RapidOCR(params={**base, "Det.limit_side_len": 200})
        # Slow path for hard/noisy backgrounds: default detector scaling.
        self._slow = RapidOCR(params=base)
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)

    @staticmethod
    def _parse(res, scale: float) -> list[TextLine]:
        lines: list[TextLine] = []
        if res is None or res.txts is None or res.word_results is None:
            return lines
        for txt, line_score, words in zip(res.txts, res.scores, res.word_results):
            glyphs: list[Glyph] = []
            for w_text, w_score, w_box in words or ():
                if w_text and w_box is not None:
                    _split_word(w_text, w_box, w_score, glyphs)
            if not glyphs:
                continue
            glyphs = [Glyph(g.ch, g.x0 / scale, g.y0 / scale, g.x1 / scale, g.y1 / scale) for g in glyphs]
            lines.append(TextLine("".join(g.ch for g in glyphs), glyphs, float(line_score)))
        return lines

    def detect(self, bgr: np.ndarray, *, upscale: float = 2.0, slow: bool = False) -> list[TextLine]:
        """Return text lines with per-character boxes in `bgr` coordinates."""
        if slow:
            return self._parse(self._slow(bgr, return_word_box=True), 1.0)
        img = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        return self._parse(self._fast(img, return_word_box=True), upscale)


def pick_hover(lines: list[TextLine], cx: float, cy: float) -> Hover | None:
    """Select the single line under (cx, cy) and the character on it.

    - A line is a candidate only if the cursor's y lies within its vertical
      extent (plus a small tolerance); the closest one wins. Lines above/below
      are therefore never considered.
    - The hovered character is the one whose box contains cx; inside a small
      gap we snap to the nearest glyph. Outside the line's horizontal extent
      (by more than half a glyph) there is no hover.
    """
    best: tuple[float, TextLine] | None = None
    for ln in lines:
        x0, y0, x1, y1 = ln.bounds
        h = y1 - y0
        tol = 0.15 * h
        if not (y0 - tol <= cy <= y1 + tol):
            continue
        dy = 0.0 if y0 <= cy <= y1 else min(abs(cy - y0), abs(cy - y1))
        dx = 0.0 if x0 <= cx <= x1 else min(abs(cx - x0), abs(cx - x1))
        cost = dy * 2 + dx
        if best is None or cost < best[0]:
            best = (cost, ln)
    if best is None:
        return None
    ln = best[1]

    x0, _, x1, _ = ln.bounds
    avg_w = (x1 - x0) / max(1, len(ln.glyphs))
    if cx < x0 - 0.5 * avg_w or cx > x1 + 0.5 * avg_w:
        return None
    for i, g in enumerate(ln.glyphs):
        if g.x0 <= cx <= g.x1:
            return Hover(ln, i)
    centers = [(g.x0 + g.x1) / 2 for g in ln.glyphs]
    i = int(np.argmin([abs(c - cx) for c in centers]))
    return Hover(ln, i)
