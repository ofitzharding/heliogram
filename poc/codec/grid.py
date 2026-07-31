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
HEADER_LEN = 28          # bytes, pre-RS (v2: +zone_w, +zone_modes, +2 spare)
HEADER_ECC = 40          # RS parity bytes on the header.
                         # Was 12 (corrects 6 byte errors) and measured as the
                         # dominant loss on real captures: 1346 frames located,
                         # only 346 headers readable. The header strip spans the
                         # full width, so it eats the same edge defect as the
                         # payload but with a fraction of the protection. It is
                         # ~0.4% of the grid; over-armouring it is nearly free,
                         # and every header recovered is a whole frame recovered.
PAYLOAD_ECC = 32         # RS parity bytes per 255-byte chunk of payload
                         # (set_ecc() overrides; the value travels in the header)


def set_ecc(n: int) -> None:
    """Set payload parity per codeword. Measured on real captures: a mild
    optical defect (one soft/glared screen edge) puts 25-30 byte errors in a
    255-byte codeword while overall BER is only ~1.2%. ECC 32 corrects 16 and
    fails; the errors are real and concentrated, not noise."""
    global PAYLOAD_ECC
    PAYLOAD_ECC = n

MODE_MONO = 0
MODE_COLOR8 = 1
MODE_COLOR4 = 3  # black/red/green/blue, 2 bits/cell.
                 # Measured on real footage (8 frames, 78,912 cells): 99.6%
                 # symbol accuracy at 4.0 sigma margin, versus 97.6% at 1.0
                 # sigma for the 8-colour set that failed. Doubling the bits
                 # costs nothing on the transmit side; the alphabet size, not
                 # colour itself, was what broke earlier attempts.
COLOR4 = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
                  dtype=np.float32)   # RGB rows; rendered reversed to BGR
COLOR4_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)

MODE_ADAPTIVE = 4  # spatially-adaptive bit-loading (waterfilling, pillar 3).
                   # The screen is parallel subchannels with measured unequal
                   # quality (0.0% error center vs 13% at one edge on real
                   # captures; mono carries 9.8 sigma of unspent margin). A
                   # uniform code transmits at worst-region rate everywhere.
                   # This mode loads bits per zone: cells within `zone_w` of
                   # the payload boundary use zone-0's alphabet, cells beyond
                   # 3*zone_w use zone-2's, the band between uses zone-1's.
                   # zone_modes packs 2 bits per zone: 0=mono(1b), 1=color4(2b).

def zone_mode(zone_modes: int, z: int) -> int:
    return (zone_modes >> (2 * z)) & 0x3   # 0=mono, 1=color4


MODE_GRAY4 = 2   # 4 luminance levels, 2 bits/cell. Chosen from measurement:
                 # real captures show black/white separated by ~12 sigma while
                 # chroma axes collapse, so spend bits on the axis that works.

GRAY4_LEVELS = np.array([0.0, 85.0, 170.0, 255.0])
# Gray-coded so an adjacent-level error costs one bit, not two
GRAY4_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
GRAY4_SYM = {v: k for k, v in GRAY4_BITS.items()}

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

    def zone_of(self, zone_w: int) -> np.ndarray:
        """Zone index (0 edge / 1 mid / 2 center) per payload cell."""
        r, c = self.payload_cells[:, 0], self.payload_cells[:, 1]
        d = np.minimum(np.minimum(r - 1, self.gh - 2 - r),
                       np.minimum(c - 1, self.gw - 2 - c))
        return np.where(d < zone_w, 0, np.where(d >= 3 * zone_w, 2, 1))

    def bits_per_cell(self, mode: int, zone_w: int = 0, zone_modes: int = 0):
        if mode != MODE_ADAPTIVE:
            bpc = {MODE_MONO: 1, MODE_COLOR8: 3, MODE_GRAY4: 2, MODE_COLOR4: 2}[mode]
            return np.full(len(self.payload_cells), bpc, dtype=np.int64)
        z = self.zone_of(zone_w)
        zm = np.array([zone_mode(zone_modes, i) for i in range(3)])
        return np.where(zm[z] == 1, 2, 1)

    def payload_capacity_bytes(self, mode: int, zone_w: int = 0,
                               zone_modes: int = 0) -> int:
        bits = int(self.bits_per_cell(mode, zone_w, zone_modes).sum())
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


def pack_header(seq: int, k: int, block_size: int, file_size: int, mode: int,
                zone_w: int = 0, zone_modes: int = 0) -> bytes:
    body = MAGIC + struct.pack("<BBIIHIBBB", 2, mode, seq, k, block_size,
                               file_size, PAYLOAD_ECC, zone_w, zone_modes)
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
    ver, mode, seq, k, block_size, file_size, ecc, zw, zm = \
        struct.unpack("<BBIIHIBBB", body[4:23])
    return dict(version=ver, mode=mode, seq=seq, k=k, block_size=block_size,
                file_size=file_size, ecc=ecc, zone_w=zw, zone_modes=zm)


def header_templates(proto: dict, max_seq: int) -> np.ndarray:
    """Expected header bit patterns for every candidate seq, (max_seq+1, nbits).

    Only `seq` varies between frames; k, block_size, file_size and mode are
    constants of the transfer. So a header is not really 64 unknown bytes, it
    is one unknown integer, and the receiver can test all its values.
    """
    return np.array([_bits(pack_header(s, proto["k"], proto["block_size"],
                                       proto["file_size"], proto["mode"],
                                       proto.get("zone_w", 0),
                                       proto.get("zone_modes", 0)))
                     for s in range(max_seq + 1)], dtype=np.int8)


def ml_header_seq(hdr_lum: np.ndarray, templates: np.ndarray):
    """Maximum-likelihood seq from raw header luminances. Returns (seq, margin).

    Correlates soft (unthresholded) measurements against every candidate
    template. With ~500 bits of evidence the true seq wins by a wide margin
    even when hard-decision RS decoding of the same strip fails outright,
    which is the whole point: thresholding throws away the evidence that
    distinguishes the candidates.
    """
    nb = templates.shape[1]
    v = hdr_lum[:nb].astype(np.float64)
    z = v - np.median(v)
    scores = (templates.astype(np.float64) * 2 - 1) @ z
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    spread = scores.std() + 1e-9
    return best, float((scores[order[0]] - scores[order[1]]) / spread)


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()


def render_frame(layout: Layout, header: bytes, payload: bytes,
                 mode: int = MODE_MONO, cell_px: int = 12,
                 zone_w: int = 0, zone_modes: int = 0) -> np.ndarray:
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
    hc = layout.header_cells[: len(hb)]
    cells[hc[:, 0], hc[:, 1]] = (255.0 * hb)[:, None]
    # unused header cells stay black

    # payload
    coded = bytes(RSCodec(PAYLOAD_ECC).encode(payload))
    if mode == MODE_MONO:
        pb = _bits(coded)
        n = min(len(pb), len(layout.payload_cells))
        pc = layout.payload_cells[:n]
        cells[pc[:, 0], pc[:, 1]] = (255.0 * pb[:n])[:, None]
    elif mode == MODE_ADAPTIVE:
        pb = _bits(coded)
        bpc = layout.bits_per_cell(mode, zone_w, zone_modes)
        total = int(bpc.sum())
        pb = np.concatenate([pb, np.zeros(max(0, total - len(pb)), np.uint8)])[:total]
        starts = np.cumsum(bpc) - bpc
        pc = layout.payload_cells
        monoM = bpc == 1
        colorM = ~monoM
        cells[pc[monoM, 0], pc[monoM, 1]] = (255.0 * pb[starts[monoM]])[:, None]
        syms = pb[starts[colorM]] * 2 + pb[starts[colorM] + 1]
        cells[pc[colorM, 0], pc[colorM, 1]] = COLOR4[syms][:, ::-1]   # BGR
    elif mode == MODE_COLOR4:  # 2 bits per cell, 4-colour constellation
        pb = _bits(coded)
        pb = np.concatenate([pb, np.zeros((-len(pb)) % 2, dtype=np.uint8)])
        syms = pb.reshape(-1, 2) @ np.array([2, 1])
        n = min(len(syms), len(layout.payload_cells))
        pc = layout.payload_cells[:n]
        cells[pc[:, 0], pc[:, 1]] = COLOR4[syms[:n]][:, ::-1]   # BGR
    elif mode == MODE_GRAY4:  # 2 bits per cell, Gray-coded luminance levels
        pb = _bits(coded)
        pb = np.concatenate([pb, np.zeros((-len(pb)) % 2, dtype=np.uint8)])
        pairs = pb.reshape(-1, 2)
        syms = np.array([GRAY4_SYM[(int(a), int(b))] for a, b in pairs])
        n = min(len(syms), len(layout.payload_cells))
        pc = layout.payload_cells[:n]
        cells[pc[:, 0], pc[:, 1]] = GRAY4_LEVELS[syms[:n]][:, None]
    else:  # color8: 3 bits per cell
        pb = _bits(coded)
        pb = np.concatenate([pb, np.zeros((-len(pb)) % 3, dtype=np.uint8)])
        syms = pb.reshape(-1, 3) @ np.array([4, 2, 1])
        n = min(len(syms), len(layout.payload_cells))
        pc = layout.payload_cells[:n]
        cells[pc[:, 0], pc[:, 1]] = PALETTE[syms[:n]][:, ::-1]  # BGR

    img = cv2.resize(cells.astype(np.uint8), (gw * cell_px, gh * cell_px),
                     interpolation=cv2.INTER_NEAREST)
    return img


# ---------------------------------------------------------------- locating

def known_cells(layout: Layout):
    """(cells, expected01) for structure the receiver knows a priori:
    the timing ring and the finder patterns."""
    gh, gw, f = layout.gh, layout.gw, layout.finder
    cells, exp = [], []
    for c in range(gw):
        cells += [(0, c), (gh - 1, c)]
        exp += [c % 2 == 0] * 2
    for r in range(1, gh - 1):
        cells += [(r, 0), (r, gw - 1)]
        exp += [r % 2 == 0] * 2
    tpl = np.zeros((f, f), dtype=bool)
    tpl[1:-1, 1:-1] = True
    tpl[2:-2, 2:-2] = False
    for (r0, c0) in [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]:
        for rr in range(f):
            for cc in range(f):
                cells.append((r0 + rr, c0 + cc))
                exp.append(bool(tpl[rr, cc]))
    return np.array(cells), np.array(exp, dtype=bool)


def refine_H(img: np.ndarray, layout: Layout, H: np.ndarray, radius: int = 3):
    """Snap an approximate (tracked) homography onto this capture by local
    search over image-space translation, scored against the known timing
    ring + finder cells. A tracked pose is a prediction; this is the
    per-capture measurement update, and it converts alignment bias (which
    evidence averaging cannot fix) into noise (which it can)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cells, exp = known_cells(layout)
    centers = np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5], axis=1).astype(np.float32)
    pts = cv2.perspectiveTransform(centers[None], H)[0]
    h, w = gray.shape
    best, best_off = None, (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xs = np.clip((pts[:, 0] + dx).astype(int), 0, w - 1)
            ys = np.clip((pts[:, 1] + dy).astype(int), 0, h - 1)
            lum = gray[ys, xs].astype(np.float32)
            score = lum[exp].mean() - lum[~exp].mean()
            if best is None or score > best:
                best, best_off = score, (dx, dy)
    T = np.eye(3)
    T[0, 2], T[1, 2] = best_off
    return T @ H

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


def sample_frame(img: np.ndarray, layout: Layout, H: np.ndarray | None = None):
    """Locate + sample + parse header, WITHOUT deciding the payload.
    Returns (header_dict|None, pay_samples (n,3)|None, stats).

    This split exists for the evidence-integrating receiver: samples from
    multiple captures of the same displayed frame can be accumulated before
    any hard decision is made. Pass H to skip detection and sample with a
    tracked homography instead (the tracking receiver).
    """
    stats = {"located": False, "header_ok": False, "rs_ok": False, "cell_margin": 0.0}
    if H is None:
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

    spread = max(1e-3, np.percentile(pay_lum, 90) - np.percentile(pay_lum, 10))
    stats["cell_margin"] = float(np.mean(np.abs(pay_lum - th)) / spread)

    n_hdr_bits = (HEADER_LEN + HEADER_ECC) * 8
    hdr_bits = (hdr_lum[:n_hdr_bits] > th).astype(np.uint8)
    header = unpack_header(_bytes(hdr_bits))
    stats["header_ok"] = header is not None
    # Return the payload samples even when the header is unreadable: the
    # geometry succeeded, so the frame is still worth rescuing by other means
    # (see ml_header_seq). Discarding it here was throwing away 73% of
    # successfully located frames on real captures.
    return header, pay_samples, stats


def _local_normalize(samples: np.ndarray, layout: Layout, k: int = 15):
    """Divide each cell by the local mean of its neighbourhood (gray world).

    Cancels vignetting, off-axis contrast roll-off and lighting gradients,
    which is what let a global palette collapse at the screen edges. Measured
    to lift 8-colour accuracy from 50% to 62% and to make 4-colour reach 99.6%.
    """
    cells = layout.payload_cells[: len(samples)]
    g = np.zeros((layout.gh, layout.gw, 3), np.float32)
    mk = np.zeros((layout.gh, layout.gw), np.float32)
    g[cells[:, 0], cells[:, 1]] = samples
    mk[cells[:, 0], cells[:, 1]] = 1
    num = cv2.boxFilter(g, -1, (k, k), normalize=False)
    den = cv2.boxFilter(mk, -1, (k, k), normalize=False)
    loc = num / np.maximum(den, 1)[..., None]
    lm = loc[cells[:, 0], cells[:, 1]]
    return samples / np.maximum(lm, 1.0) * 127.0


def _learn_color4(X: np.ndarray) -> np.ndarray:
    """Estimate the four centroids from the data, then label them by structure.

    k-means finds the clusters; the labelling uses facts that hold for any
    display: black is the darkest cluster, and each remaining cluster is named
    by whichever channel dominates it. No reference palette is used, so the
    receiver adapts to the actual colour transfer of this screen and camera.
    """
    Z = X.astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    _r, lab, cent = cv2.kmeans(Z, 4, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = np.zeros((4, 3), np.float32)
    order = np.argsort(cent.mean(axis=1))
    dark = int(order[0])
    out[0] = cent[dark]                      # symbol 0 = black
    rest = [i for i in range(4) if i != dark]
    # samples are BGR; symbol 1=red -> channel 2, 2=green -> 1, 3=blue -> 0
    for sym, ch in ((1, 2), (2, 1), (3, 0)):
        best = max(rest, key=lambda i: cent[i][ch] - np.delete(cent[i], ch).mean())
        out[sym] = cent[best]
        rest.remove(best)
    return out


def decide_payload(header: dict, pay_samples: np.ndarray,
                   layout: 'Layout' = None):
    """Hard-decide cells from (possibly evidence-averaged) samples, then RS."""
    pay_lum = pay_samples.mean(axis=1)
    margins = None
    if header["mode"] == MODE_ADAPTIVE:
        bpc = layout.bits_per_cell(MODE_ADAPTIVE, header["zone_w"],
                                   header["zone_modes"])
        n = min(len(bpc), len(pay_lum))
        bpc = bpc[:n]
        starts = np.cumsum(bpc) - bpc
        total = int(bpc.sum())
        bits = np.zeros(total, dtype=np.uint8)
        margins = np.zeros(total, dtype=np.float32)
        monoM = bpc == 1
        colorM = ~monoM
        # mono zone: Otsu over the mono cells only
        ml = pay_lum[:n][monoM]
        th, _ = cv2.threshold(np.clip(ml, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bits[starts[monoM]] = (ml > th).astype(np.uint8)
        spread = max(1e-3, np.percentile(ml, 90) - np.percentile(ml, 10))
        margins[starts[monoM]] = np.abs(ml - th) / spread
        # colour zone: self-calibrating 4-colour on those cells only
        X = _local_normalize(pay_samples[:n], layout)[colorM]
        cent = _learn_color4(X)
        d = ((X[:, None, :] - cent[None]) ** 2).sum(axis=2)
        syms = d.argmin(axis=1)
        bits[starts[colorM]] = (syms >> 1).astype(np.uint8)
        bits[starts[colorM] + 1] = (syms & 1).astype(np.uint8)
        srt = np.sort(d, axis=1)
        cm = ((np.sqrt(srt[:, 1]) - np.sqrt(srt[:, 0])) /
              max(1e-3, float(np.median(np.sqrt(srt[:, 1])))))
        margins[starts[colorM]] = cm
        margins[starts[colorM] + 1] = cm
        raw = _bytes(bits)
    elif header["mode"] == MODE_COLOR4:
        # Self-calibrating constellation. Two stages, both measured as
        # necessary: (1) local gray-world normalisation removes the vignetting
        # and off-axis contrast roll-off that made a global palette fail;
        # (2) learn the four centroids from this frame's own data and label
        # them by structure (darkest = black; the rest by dominant channel),
        # so no fixed palette is assumed anywhere.
        X = _local_normalize(pay_samples, layout)
        cent = _learn_color4(X)
        d = ((X[:, None, :] - cent[None]) ** 2).sum(axis=2)
        syms = d.argmin(axis=1)
        bits = COLOR4_BITS[syms].reshape(-1)
        raw = _bytes(bits)
        srt = np.sort(d, axis=1)
        margins = np.repeat(
            (np.sqrt(srt[:, 1]) - np.sqrt(srt[:, 0])) /
            max(1e-3, np.median(np.sqrt(srt[:, 1]))), 2)
    elif header["mode"] == MODE_GRAY4:
        # learn the 4 level centers from this frame's own data (symbols are
        # ~uniform, so quartile means are unbiased estimates), then classify
        # by nearest learned center — the receiver adapts to whatever gamma,
        # exposure, and contrast the channel actually delivered
        q = np.quantile(pay_lum, [0.125, 0.375, 0.625, 0.875])
        edges = np.quantile(pay_lum, [0.25, 0.5, 0.75])
        centers = np.array([
            pay_lum[pay_lum <= edges[0]].mean(),
            pay_lum[(pay_lum > edges[0]) & (pay_lum <= edges[1])].mean(),
            pay_lum[(pay_lum > edges[1]) & (pay_lum <= edges[2])].mean(),
            pay_lum[pay_lum > edges[2]].mean(),
        ])
        syms = np.abs(pay_lum[:, None] - centers[None]).argmin(axis=1)
        bits = np.array([GRAY4_BITS[int(s)] for s in syms], dtype=np.uint8).reshape(-1)
        raw = _bytes(bits)
        bounds = (centers[:-1] + centers[1:]) / 2
        dist = np.min(np.abs(pay_lum[:, None] - bounds[None]), axis=1)
        spread = max(1e-3, centers[-1] - centers[0])
        cm = dist / spread
        margins = np.repeat(cm, 2)   # each cell contributes 2 bits
    elif header["mode"] == MODE_MONO:
        th, _ = cv2.threshold(np.clip(pay_lum, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bits = (pay_lum > th).astype(np.uint8)
        raw = _bytes(bits)
        spread = max(1e-3, np.percentile(pay_lum, 90) - np.percentile(pay_lum, 10))
        margins = np.abs(pay_lum - th) / spread   # per-cell confidence
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
    rs = RSCodec(PAYLOAD_ECC)
    try:
        return bytes(rs.decode(raw[:coded_len])[0])
    except ReedSolomonError:
        pass
    if margins is None:
        return None
    # Soft-decision rescue: RS corrects e errors + s erasures with 2e+s <= 32,
    # so telling it WHERE the doubt is doubles the budget. A byte is doubtful
    # if it contains a cell whose luminance sat near the threshold.
    byte_margin = margins[: (coded_len) * 8].reshape(-1, 8).min(axis=1)
    out = bytearray()
    pos = 0
    while pos < coded_len:
        k = min(255, coded_len - pos)
        chunk = raw[pos:pos + k]
        bm = byte_margin[pos:pos + k]
        erase = list(np.argsort(bm)[: PAYLOAD_ECC - 6])   # keep >=3-error headroom
        erase = [int(i) for i in erase if bm[i] < 0.15]
        try:
            out += rs.decode(chunk, erase_pos=erase)[0]
        except ReedSolomonError:
            return None
        pos += k
    return bytes(out)


def decode_frame(img: np.ndarray, layout: Layout):
    """Single-capture decode (the classical receiver). Returns (header, payload, stats)."""
    header, pay_samples, stats = sample_frame(img, layout)
    if header is None or pay_samples is None:
        return None, None, stats
    payload = decide_payload(header, pay_samples, layout)
    if payload is not None:
        stats["rs_ok"] = True
    return header, payload, stats
