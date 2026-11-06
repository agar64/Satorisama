from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mss
import cv2

from utils_win import get_virtual_screen_rect

@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

def clamp_region_to_virtual_screen(region: CaptureRegion) -> CaptureRegion:
    vx, vy, vw, vh = get_virtual_screen_rect()
    # Clamp within virtual screen bounds
    left = max(region.left, vx)
    top = max(region.top, vy)
    right = min(region.left + region.width, vx + vw)
    bottom = min(region.top + region.height, vy + vh)
    width = max(1, right - left)
    height = max(1, bottom - top)
    return CaptureRegion(left, top, width, height)

def capture_bgr(region: CaptureRegion) -> np.ndarray:
    """
    Returns BGR uint8 image (H, W, 3)
    """
    region = clamp_region_to_virtual_screen(region)
    with mss.mss() as sct:
        monitor = {"left": region.left, "top": region.top, "width": region.width, "height": region.height}
        raw = sct.grab(monitor)  # BGRA
        img = np.asarray(raw, dtype=np.uint8)  # H,W,4
        bgr = img[:, :, :3]  # drop alpha, still BGR
        return bgr

def anchor_crop_near_cursor(
    bgr: np.ndarray,
    cursor_x_in_roi: int,
    cursor_y_in_roi: int
) -> tuple[np.ndarray, int]:
    """
    Right-biased crop anchored to the glyph under the cursor.

    Fixes "extra left character" in small text by:
    - stronger upscale (scale=4) so gaps are wider in pixel space
    - TWO masks:
        * thr_run: mild morphology for "run end" to the right
        * thr_cc : conservative morphology for connected component under cursor
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr, 0

    # Stronger upscale helps small fonts
    scale = 4
    H, W = h * scale, w * scale

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.resize(gray, (W, H), interpolation=cv2.INTER_CUBIC)
    bgr2 = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_CUBIC)

    cx = int(np.clip(cursor_x_in_roi * scale, 0, W - 1))
    cy = int(np.clip(cursor_y_in_roi * scale, 0, H - 1))

    # Base threshold (shared)
    thr0 = cv2.adaptiveThreshold(
        gray2, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    thr0 = cv2.medianBlur(thr0, 3)

    # -----------------------
    # Mask for "runs" (right extension)
    # Mild close to connect within a word, but not too aggressive
    kernel_run = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    thr_run = cv2.morphologyEx(thr0, cv2.MORPH_CLOSE, kernel_run, iterations=1)

    # Mask for CC (hovered glyph) – conservative to avoid merging neighbors
    # NOTE: no wide horizontal close here
    kernel_cc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr_cc = cv2.morphologyEx(thr0, cv2.MORPH_CLOSE, kernel_cc, iterations=1)
    # -----------------------

    # --- Find ink runs over columns using thr_run ---
    col_sum = thr_run.sum(axis=0)
    ink_cols = col_sum > (H * 255 * 0.01)

    runs = []
    i = 0
    while i < W:
        if not ink_cols[i]:
            i += 1
            continue
        j = i
        while j < W and ink_cols[j]:
            j += 1
        runs.append((i, j))
        i = j

    if not runs:
        return bgr, int(np.clip(cursor_x_in_roi, 0, w - 1))

    containing = [r for r in runs if r[0] <= cx < r[1]]
    if containing:
        rx0, rx1 = containing[0]
    else:
        rx0, rx1 = min(runs, key=lambda r: abs(((r[0] + r[1]) // 2) - cx))

    # --- Connected component under/near cursor using thr_cc ---
    ink = (thr_cc > 0).astype(np.uint8)

    # If cursor is on background (holes/gaps), snap to nearest ink pixel
    cx2, cy2 = cx, cy
    if ink[cy2, cx2] == 0:
        r = 12 * scale
        y0p = max(0, cy2 - r)
        y1p = min(H, cy2 + r + 1)
        x0p = max(0, cx2 - r)
        x1p = min(W, cx2 + r + 1)
        patch = ink[y0p:y1p, x0p:x1p]
        ys, xs = np.where(patch > 0)
        if ys.size:
            dy = ys + y0p - cy2
            dx = xs + x0p - cx2
            k = int(np.argmin(dx * dx + dy * dy))
            cy2 = int(ys[k] + y0p)
            cx2 = int(xs[k] + x0p)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    lbl = int(labels[cy2, cx2]) if (0 <= cy2 < H and 0 <= cx2 < W) else 0

    # Fallback if CC failed
    if lbl <= 0 or lbl >= num:
        # Minimal safe crop around cursor
        x0 = max(0, cx - 8 * scale)
        x1 = min(W, rx1 + 70 * scale)
        y0, y1 = 0, H
    else:
        x_cc, y_cc, w_cc, h_cc, _area = stats[lbl]
        x_char0 = int(x_cc)
        x_char1 = int(x_cc + w_cc)
        y_char0 = int(y_cc)
        y_char1 = int(y_cc + h_cc)

        # Small margins to avoid anti-alias clipping
        char_margin_x = 3 * scale
        char_margin_y = 3 * scale

        # Anchor LEFT to the hovered glyph bbox (prevents grabbing previous char)
        x0 = max(0, x_char0 - char_margin_x)

        # Never cut the right side: dictionary matcher will decide what to use.
        x1 = W

        # Vertical crop: start at glyph bbox, then expand to include ink in [x0:x1]
        y0 = max(0, y_char0 - char_margin_y)
        y1 = min(H, y_char1 + char_margin_y)

        roi_thr = thr_run[:, x0:x1]
        row_sum = roi_thr.sum(axis=1)
        ink_r = row_sum > ((x1 - x0) * 255 * 0.01)
        ys = np.where(ink_r)[0]
        if ys.size > 0:
            y0 = max(0, min(y0, int(ys.min()) - 8 * scale))
            y1 = min(H, max(y1, int(ys.max()) + 8 * scale))

    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return bgr, int(np.clip(cursor_x_in_roi, 0, w - 1))

    crop2 = bgr2[y0:y1, x0:x1]
    out = cv2.resize(
        crop2,
        (max(1, crop2.shape[1] // scale), max(1, crop2.shape[0] // scale)),
        interpolation=cv2.INTER_AREA,
    )
    cursor_x_in_crop = int(np.clip((cx - x0) // scale, 0, out.shape[1] - 1))
    return out, cursor_x_in_crop


def isolate_text_line_near_cursor(bgr: np.ndarray, cursor_y_in_roi: int) -> np.ndarray:
    """
    Crop ROI vertically to only the text line closest to cursor_y.
    Keeps ROI width unchanged; only reduces height intelligently.
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upscale a bit for more stable binarization on small fonts
    scale = 2
    gray2 = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cy = int(cursor_y_in_roi * scale)

    # Binary ink map (adaptive handles varying backgrounds)
    bw = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # Connect characters into bigger blobs horizontally, modestly vertically
    bw = cv2.medianBlur(bw, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # Filter components near cursor row
    # stats: [x, y, w, h, area]
    near = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 40 * scale:  # ignore tiny noise
            continue
        y0, y1 = y, y + hh
        # keep comps whose vertical span intersects a band around cursor
        band = 18 * scale
        if y0 <= cy + band and y1 >= cy - band:
            near.append((x, y, ww, hh))

    if not near:
        # fallback: return original ROI
        return bgr

    # Union their vertical extent (crop only Y)
    y0 = min(y for _, y, _, _ in near)
    y1 = max(y + hh for _, y, _, hh in near)

    # Pad vertically so tall fonts still fit
    pad = 10 * scale
    y0 = max(0, y0 - pad)
    y1 = min(gray2.shape[0], y1 + pad)

    # Crop from original bgr with corresponding scale
    bgr2 = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cropped2 = bgr2[y0:y1, :]

    # Downscale back
    out = cv2.resize(
        cropped2,
        (w, max(1, (y1 - y0) // scale)),
        interpolation=cv2.INTER_AREA,
    )
    return out

def isolate_text_line_band(bgr: np.ndarray, cursor_y_in_roi: int) -> np.ndarray:
    """
    Strong single-line isolation using row-projection.
    Works well for horizontal UI text and avoids mixing lines above/below.
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upscale for small fonts
    scale = 2
    gray2 = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cy = int(cursor_y_in_roi * scale)

    # Ink mask
    bw = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    bw = cv2.medianBlur(bw, 3)

    # Light horizontal close so characters become a single band, but don't merge lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Row projection (ink per row)
    row = bw.sum(axis=1)  # 0..255*w
    # Smooth projection a bit
    row = cv2.GaussianBlur(row.astype(np.float32), (1, 9), 0).reshape(-1)

    # Search for the strongest peak near cursor
    band = 40 * scale
    y0s = max(0, cy - band)
    y1s = min(len(row), cy + band)
    local = row[y0s:y1s]
    if local.size == 0:
        return bgr

    peak_rel = int(np.argmax(local))
    peak = y0s + peak_rel
    peak_val = float(row[peak])
    if peak_val < 1e-6:
        return bgr

    # Expand around peak until projection drops below threshold
    thresh = peak_val * 0.25  # tighten/loosen: 0.20–0.35
    y0 = peak
    while y0 > 0 and row[y0] > thresh:
        y0 -= 1
    y1 = peak
    while y1 < len(row) - 1 and row[y1] > thresh:
        y1 += 1

    # Pad to handle large fonts
    pad = 10 * scale
    y0 = max(0, y0 - pad)
    y1 = min(gray2.shape[0], y1 + pad)

    # Crop original (scaled) then downscale back to original width
    bgr2 = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    crop2 = bgr2[y0:y1, :]

    out_h = max(1, (y1 - y0) // scale)
    out = cv2.resize(crop2, (w, out_h), interpolation=cv2.INTER_AREA)
    return out

def isolate_line_by_row_signal(bgr: np.ndarray, cursor_y_in_roi: int) -> np.ndarray:
    """
    1) grayscale
    2) binarize to ink mask (text = 1)
    3) row_ink[y] = fraction of ink pixels in that row
    4) threshold row_ink -> 0/1
    5) contiguous 1-run containing cursor_y is the line band
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # optional upscale for small fonts
    scale = 2
    gray2 = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cy = int(cursor_y_in_roi * scale)
    cy = max(0, min(cy, gray2.shape[0] - 1))

    # Binarize ink: white background -> 0, text -> 1
    bw = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # tiny denoise, but avoid vertical merging
    bw = cv2.medianBlur(bw, 3)

    # row ink fraction in [0,1]
    row_ink = (bw > 0).mean(axis=1).astype(np.float32)

    # choose a threshold: above this is "has text"
    # robust: baseline + margin
    baseline = float(np.median(row_ink))
    thr = max(0.01, baseline * 3.0)  # tune 2.0–4.0

    # if cursor row isn't "texty", snap to nearest texty row nearby
    if row_ink[cy] < thr:
        band = 60
        y0s = max(0, cy - band)
        y1s = min(len(row_ink), cy + band + 1)
        idx = np.where(row_ink[y0s:y1s] >= thr)[0]
        if idx.size == 0:
            return bgr
        cy = y0s + int(idx[np.argmin(np.abs((y0s + idx) - cy))])

    # Binarize the row signal
    row_bin = row_ink >= thr

    # Find contiguous True segment containing cy
    y0 = cy
    while y0 > 0 and row_bin[y0 - 1]:
        y0 -= 1
    y1 = cy
    while y1 < len(row_bin) - 1 and row_bin[y1 + 1]:
        y1 += 1

    # pad to include ascenders/descenders / thicker fonts
    pad = 12 * scale
    y0 = max(0, y0 - pad)
    y1 = min(gray2.shape[0] - 1, y1 + pad)

    # crop scaled image and downscale back
    bgr2 = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    crop2 = bgr2[y0:y1 + 1, :]

    out_h = max(1, crop2.shape[0] // scale)
    out = cv2.resize(crop2, (w, out_h), interpolation=cv2.INTER_AREA)
    return out

def detect_line_crop_bounds_y(bgr, cursor_y: int):
    h, w = bgr.shape[:2]
    cursor_y = int(np.clip(cursor_y, 0, h - 1))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Binary: text dark (1), background (0)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # invert => ink is white
        31, 7
    )

    # ink fraction per row in [0..1]
    ink = (bw > 0).mean(axis=1).astype(np.float32)

    # smooth (moving average)
    k = 9
    kernel = np.ones(k, dtype=np.float32) / k
    ink_s = np.convolve(ink, kernel, mode="same")

    mx = float(ink_s.max()) if ink_s.size else 0.0
    if mx <= 1e-6:
        return 0, h  # nothing detected

    # line threshold relative to max ink
    line_thr = mx * 0.35
    valley_thr = mx * 0.08  # "almost empty" rows

    # Find peak band containing cursor
    y0 = cursor_y
    while y0 > 0 and ink_s[y0] >= line_thr:
        y0 -= 1
    y1 = cursor_y
    while y1 < h - 1 and ink_s[y1] >= line_thr:
        y1 += 1

    # Helper to find nearest valley region going upward/downward
    def find_valley_up(start_y):
        y = start_y
        # skip any remaining ink
        while y > 0 and ink_s[y] > valley_thr:
            y -= 1
        if y <= 0:
            return None
        v1 = y
        while y > 0 and ink_s[y] <= valley_thr:
            y -= 1
        v0 = y + 1
        return (v0, v1)

    def find_valley_down(start_y):
        y = start_y
        while y < h - 1 and ink_s[y] > valley_thr:
            y += 1
        if y >= h - 1:
            return None
        v0 = y
        while y < h - 1 and ink_s[y] <= valley_thr:
            y += 1
        v1 = y - 1
        return (v0, v1)

    up = find_valley_up(y0)
    down = find_valley_down(y1)

    # If valleys exist, use their midpoints as crop edges.
    # Otherwise fall back to ROI edges.
    top = (up[0] + up[1]) // 2 if up else 0
    bot = (down[0] + down[1]) // 2 if down else h - 1

    # Ensure cursor line is inside bounds
    top = min(top, cursor_y)
    bot = max(bot, cursor_y)

    # Add a tiny safety margin (1–2 px) to avoid cutting antialiasing
    top = max(0, top - 2)
    bot = min(h - 1, bot + 2)

    return top, bot + 1  # python slice end-exclusive

def crop_and_mask_to_line(bgr: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """
    Crops ROI to [y0:y1] and then WHITENS everything outside the strongest
    text band inside that crop using a second-pass row filter.
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if y1 <= y0:
        return bgr

    crop = bgr[y0:y1+1, :].copy()

    # Second-pass: within the crop, build an ink mask and remove "thin" stray rows
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    bw = cv2.medianBlur(bw, 3)

    row_ink = (bw > 0).mean(axis=1).astype(np.float32)
    # keep rows that look like part of the main line:
    # threshold relative to max in this crop (very effective at killing other lines)
    mx = float(row_ink.max() if row_ink.size else 0.0)
    if mx <= 0.0:
        return crop

    keep_thr = mx * 0.35  # stricter keep; tune 0.25–0.45
    keep = row_ink >= keep_thr

    # Allow small gaps so the line isn't shredded
    # Simple gap fill: dilate keep mask in 1D
    k = np.ones(7, dtype=np.uint8)
    keep_u8 = keep.astype(np.uint8)
    keep_u8 = np.convolve(keep_u8, k, mode="same") > 0
    keep = keep_u8

    # Whitening mask: rows we do NOT keep become white
    for y in range(crop.shape[0]):
        if not keep[y]:
            crop[y, :, :] = 255

    return crop
