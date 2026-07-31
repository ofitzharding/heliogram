"""Dense grid frame codec: render data frames, locate them in captures, sample cells.

Layout (in cell units, default 120x68):
  - 1-cell timing ring around the border (alternating B/W) — alignment aid
  - 7x7 QR-style finder patterns in all four corners (+1-cell separator)
  - header region: first rows between the top finders, RS-heavy, mono always
  - everything else: payload cells, row-major order

v0 decodes with a single homography from the four finder centers. That
ignores lens distortion; at phone-filming-laptop distances this costs a
fraction of a cell at the frame edges, which the per-frame Reed-Solomon
absorbs. Fix with a timing-ring refinement pass if edge cells dominate the
error budget (they will show up in decode.py's per-cell error map).
"""
from __future__ import annotations

import struct
import zlib

import cv2
import numpy as np
from reedsolo import RSCodec, ReedSolomonError

MAGIC = b"SCPC"
HEADER_LEN = 24          # bytes, pre-RS
HEADER_ECC = 12          # RS parity bytes on the header
PAYLOAD_ECC = 32         # RS parity bytes per 255-byte chunk of payload

MODE_MONO = 0
MODE_COLOR8 = 1

# 8-color constellation for color8 mode: corners of the RGB cube.
PALETTE = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255],
], dtype=np.float32)  # index = 3 bits


class Layout:
    def __init__(self, gw: int = 120, gh: int = 68):
        self.gw, self.gh = gw, gh
        self.finder = 7
        f = self.finder
        # cell classes
        self.is_finder = np.zeros((gh, gw), dtype=bool)
        for (r0, c0) in [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]:
            self.is_finder[r0:r0 + f, c0:c0 + f] = True
        self.is_ring = np.zeros((gh, gw), dtype=bool)
        self.is_ring[0, :] = self.is_ring[-1, :] = True
        self.is_ring[:, 0] = self.is_ring[:, -1] = True
        # separators: one cell around each finder
        self.is_sep = np.zeros((gh, gw), dtype=bool)
        for (r0, c0) in [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]:
            self.is_sep[max(0, r0 - 1):r0 + f + 1, max(0, c0 - 1):c0 + f + 1] = True
        self.is_sep &= ~self.is_finder

        # header region: rows just inside the top ring, between the finder zones
        need_bits = (HEADER_LEN + HEADER_ECC) * 8
        c_lo, c_hi = f + 2, gw - f - 2
        per_row = c_hi - c_lo
        n_rows = int(np.ceil(need_bits / per_row))
        self.is_header = np.zeros((gh, gw), dtype=bool)
        self.is_header[1:1 + n_rows, c_lo:c_hi] = True
        self.is_header &= ~(self.is_finder | self.is_sep | self.is_ring)
        self.header_cells = np.argwhere(self.is_header)  # (r, c) row-major
        assert len(self.header_cells) >= need_bits, "grid too small for header"

        reserved = self.is_finder | self.is_sep | self.is_ring | self.is_header
        self.payload_cells = np.argwhere(~reserved)

    def payload_capacity_bytes(self, mode: int) -> int:
        bits = len(self.payload_cells) * (3 if mode == MODE_COLOR8 else 1)
        raw = bits // 8
        # largest b whose RS-encoded length fits in raw bytes
        b = raw * (255 - PAYLOAD_ECC) // 255
        while rs_encoded_len(b + 1) <= raw:
            b += 1
        while b > 0 and rs_encoded_len(b) > raw:
            b -= 1
        return b


def rs_encoded_len(n: int) -> int:
    """reedsolo chunks into 223-data + 32-parity codewords."""
    chunks = -(-n // (255 - PAYLOAD_ECC))
    return n + chunks * PAYLOAD_ECC


def pack_header(seq: int, k: int, block_size: int, file_size: int, mode: int) -> bytes:
    body = MAGIC + struct.pack("<BBIIHIB", 1, mode, seq, k, block_size, file_size, PAYLOAD_ECC)
    body += b"\x00" * (HEADER_LEN - 2 - len(body))
    body += struct.pack("<H", zlib.crc32(body) & 0xFFFF)
    return bytes(RSCodec(HEADER_ECC).encode(body))


def unpack_header(raw: bytes):
    try:
        body = bytes(RSCodec(HEADER_ECC).decode(raw)[0])
    except ReedSolomonError:
        return None
    if body[:4] != MAGIC:
        return None
    crc = struct.unpack("<H", body[-2:])[0]
    if zlib.crc32(body[:-2]) & 0xFFFF != crc:
        return None
    ver, mode, seq, k, block_size, file_size, ecc = struct.unpack("<BBIIHIB", body[4:21])
    return dict(version=ver, mode=mode, seq=seq, k=k,
                block_size=block_size, file_size=file_size, ecc=ecc)


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()


def render_frame(layout: Layout, header: bytes, payload: bytes,
                 mode: int = MODE_MONO, cell_px: int = 12) -> np.ndarray:
    """Returns a BGR image of the full frame."""
    gh, gw = layout.gh, layout.gw
    cells = np.zeros((gh, gw, 3), dtype=np.float32)

    # timing ring: alternating
    for c in range(gw):
        v = 255.0 if c % 2 == 0 else 0.0
        cells[0, c] = cells[-1, c] = v
    for r in range(gh):
        v = 255.0 if r % 2 == 0 else 0.0
        cells[r, 0] = cells[r, -1] = v

    # separators white
    cells[layout.is_sep] = 255.0

    # finders: black 7x7, white 5x5, black 3x3
    f = layout.finder
    tpl = np.zeros((f, f), dtype=np.float32)
    tpl[1:-1, 1:-1] = 255.0
    tpl[2:-2, 2:-2] = 0.0
    for (r0, c0) in [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]:
        cells[r0:r0 + f, c0:c0 + f] = tpl[..., None]

    # header: mono bits
    hb = _bits(header)
    for bit, (r, c) in zip(hb, layout.header_cells):
        cells[r, c] = 255.0 * bit
    # unused header cells stay black

    # payload
    enc = RSCodec(PAYLOAD_ECC)
    coded = bytes(enc.encode(payload))
    if mode == MODE_MONO:
        pb = _bits(coded)
        n = min(len(pb), len(layout.payload_cells))
        for bit, (r, c) in zip(pb[:n], layout.payload_cells[:n]):
            cells[r, c] = 255.0 * bit
    else:  # color8: 3 bits per cell
        pb = _bits(coded)
        pad = (-len(pb)) % 3
        pb = np.concatenate([pb, np.zeros(pad, dtype=np.uint8)])
        syms = pb.reshape(-1, 3) @ np.array([4, 2, 1])
        n = min(len(syms), len(layout.payload_cells))
        for s, (r, c) in zip(syms[:n], layout.payload_cells[:n]):
            cells[r, c] = PALETTE[s][::-1]  # BGR

    img = cv2.resize(cells.astype(np.uint8), (gw * cell_px, gh * cell_px),
                     interpolation=cv2.INTER_NEAREST)
    return img


# ---------------------------------------------------------------- locating

def _finder_centers(gray: np.ndarray):
    """QR-style finder detection: nested-contour test."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    cands = []
    for i, cnt in enumerate(contours):
        # count nesting depth under this contour
        depth, j = 0, hier[i][2]
        while j != -1 and depth < 4:
            depth += 1
            j = hier[j][2]
        if depth >= 2:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) == 4:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cands.append((area, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    cands.sort(reverse=True)
    # dedup near-coincident detections (outer + inner rings of same finder)
    out = []
    for a, x, y in cands:
        if all((x - x2) ** 2 + (y - y2) ** 2 > a * 0.5 for _, x2, y2 in out):
            out.append((a, x, y))
    return [(x, y) for _, x, y in out[:4]]


def locate(img: np.ndarray, layout: Layout):
    """Return homography unit-grid -> image, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pts = _finder_centers(gray)
    if len(pts) != 4:
        return None
    pts = np.array(pts, dtype=np.float32)
    # order TL, TR, BL, BR by the classic sum/diff trick
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]; bl = pts[np.argmin(d)]
    f = layout.finder
    fc = f / 2.0  # finder center offset from its zone origin, in cells
    src = np.array([
        [1 + fc, 1 + fc],
        [layout.gw - 1 - f + fc, 1 + fc],
        [1 + fc, layout.gh - 1 - f + fc],
        [layout.gw - 1 - f + fc, layout.gh - 1 - f + fc],
    ], dtype=np.float32)
    dst = np.array([tl, tr, bl, br], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def sample_cells(img: np.ndarray, layout: Layout, H: np.ndarray, cells: np.ndarray):
    """Sample given (r,c) cells; returns float32 (n, 3) BGR means of 3x3 patches."""
    centers = np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5], axis=1).astype(np.float32)
    pts = cv2.perspectiveTransform(centers[None], H)[0]
    h, w = img.shape[:2]
    xs = np.clip(pts[:, 0].round().astype(int), 1, w - 2)
    ys = np.clip(pts[:, 1].round().astype(int), 1, h - 2)
    imgf = img.astype(np.float32)
    # vectorized 3x3 patch means
    acc = np.zeros((len(xs), 3), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += imgf[ys + dy, xs + dx]
    return acc / 9.0


def decode_frame(img: np.ndarray, layout: Layout):
    """Full frame decode. Returns (header_dict, payload_bytes, stats)."""
    stats = {"located": False, "header_ok": False, "rs_ok": False, "cell_margin": 0.0}
    H = locate(img, layout)
    if H is None:
        return None, None, stats
    stats["located"] = True

    hdr_samples = sample_cells(img, layout, H, layout.header_cells)
    hdr_lum = hdr_samples.mean(axis=1)
    pay_samples = sample_cells(img, layout, H, layout.payload_cells)
    pay_lum = pay_samples.mean(axis=1)
    all_lum = np.concatenate([hdr_lum, pay_lum]).astype(np.uint8)
    th, _ = cv2.threshold(all_lum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_hdr_bits = (HEADER_LEN + HEADER_ECC) * 8
    hdr_bits = (hdr_lum[:n_hdr_bits] > th).astype(np.uint8)
    header = unpack_header(_bytes(hdr_bits))
    if header is None:
        return None, None, stats
    stats["header_ok"] = True

    spread = max(1e-3, np.percentile(pay_lum, 90) - np.percentile(pay_lum, 10))
    stats["cell_margin"] = float(np.mean(np.abs(pay_lum - th)) / spread)

    enc = RSCodec(PAYLOAD_ECC)
    if header["mode"] == MODE_MONO:
        bits = (pay_lum > th).astype(np.uint8)
        raw = _bytes(bits)
    else:
        # normalize channels against sampled extremes, then nearest palette color
        lo = np.percentile(pay_samples, 3, axis=0)
        hi = np.percentile(pay_samples, 97, axis=0)
        norm = (pay_samples - lo) / np.maximum(hi - lo, 1e-3) * 255.0
        d = ((norm[:, None, :] - PALETTE[None, :, ::-1]) ** 2).sum(axis=2)
        syms = d.argmin(axis=1).astype(np.uint8)
        bits = np.stack([(syms >> 2) & 1, (syms >> 1) & 1, syms & 1], axis=1).reshape(-1)
        raw = _bytes(bits)

    coded_len = rs_encoded_len(header["block_size"] + 4)  # +4: block CRC32
    try:
        payload = bytes(enc.decode(raw[:coded_len])[0])
    except ReedSolomonError:
        return header, None, stats
    stats["rs_ok"] = True
    return header, payload, stats
