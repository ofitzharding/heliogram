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


def finder_scale_for(gw: int) -> int:
    """Cells per finder module, derived from grid width so BOTH ends agree
    without signalling it.

    Measured failure that motivated this: at 560 cells across, a 7-cell finder
    is only ~40 camera px and its three-ring structure stops resolving — the
    dense capture located 0/10 frames despite the cells themselves being
    perfectly sharp. Detection, not sampling, is what caps density. Scaling
    the finder keeps markers ~100+ px at any density; the cost is under 2% of
    payload cells. gw=252 maps to 1, preserving existing captures.
    """
    return max(1, round(gw / 180))


class Layout:
    def __init__(self, gw: int = 120, gh: int = 68, finder_scale: int = None):
        self.gw, self.gh = gw, gh
        self.fs = finder_scale_for(gw) if finder_scale is None else finder_scale
        self.finder = 7 * self.fs
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

        # Header region: CENTRED rows, not the top edge.
        #
        # It used to sit in rows 1-3, which measurement showed is the worst
        # real estate on the grid: radial distortion is largest at the frame
        # edge, and a top edge is the first thing lost when framing drifts.
        # Headers were readable on only ~65% of located frames, and since a
        # frame without a header is discarded entirely, that alone capped
        # yield. The centre measured 0.0% cell error on the same captures.
        need_bits = (HEADER_LEN + HEADER_ECC) * 8
        c_lo, c_hi = f + 2, gw - f - 2
        per_row = c_hi - c_lo
        n_rows = int(np.ceil(need_bits / per_row))
        self.is_header = np.zeros((gh, gw), dtype=bool)
        r0 = max(1, (gh - n_rows) // 2) if HEADER_CENTERED else 1
        self.is_header[r0:r0 + n_rows, c_lo:c_hi] = True
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


def sub_count(layout, mode: int, zone_w: int = 0, zone_modes: int = 0) -> int:
    """Number of 255-byte codewords a frame can carry in subblock mode.

    Derived from the grid's RAW cell capacity, not from block_size. The old
    rule, n_sub = (block_size + 4) // 255, floors against a block_size that
    is itself already RS-inflated, so it systematically under-fills: measured
    77-81% of every grid painted, the remaining ~20% rendered as dead black
    cells. That is a fifth of the transmit rate thrown away on every profile,
    and it also corrupts receiver palette estimation (the dead mass is an
    extra population the k-means has to model).

    Both encoder and decoder compute this from (layout, mode) alone, so it
    needs no header field and stays compatible either way.
    """
    raw_bytes = int(layout.bits_per_cell(mode, zone_w, zone_modes).sum()) // 8
    return max(1, raw_bytes // 255)


def rs_encoded_len(n: int) -> int:
    """reedsolo chunks into 223-data + 32-parity codewords."""
    chunks = -(-n // (255 - PAYLOAD_ECC))
    return n + chunks * PAYLOAD_ECC


_HDR_MASK_CACHE = {}


def _hdr_mask(n: int, phase: int = 0) -> bytes:
    """Deterministic whitening sequence for the header.

    The header is structurally low-entropy — magic bytes, small integers
    whose high bytes are zero, and literal zero padding — so it renders as a
    visibly dark band across the middle of every frame: measured 39% white
    with a 45-cell run of solid black. Three consequences, none cosmetic:
    a global threshold computed over header+payload is dragged off the
    header's own eye; long uniform runs carry no transitions, so they cannot
    excite a channel estimate; and the band's mean luma differs from the
    payload's, which biases camera metering.

    Whitening is the standard fix (DVB, Ethernet, USB all scramble). XOR is
    applied AFTER Reed-Solomon so the code structure is untouched.
    """
    key = (n, phase)
    if key not in _HDR_MASK_CACHE:
        # phase < 0 reproduces the original single-mask whitening. Transmits
        # rendered during the brief window when whitening was fixed-mask are
        # otherwise undecodable, because no non-negative phase reuses that
        # seed. That cost a filmed take; keep it forever.
        x = 0xACE1 if phase < 0 else (0xACE1 ^ (0x1D3F * (phase + 1) & 0xFFFF)) or 0xACE1
        out = bytearray()
        for _ in range(n):
            for _ in range(8):
                lsb = x & 1
                x >>= 1
                if lsb:
                    x ^= 0xB400
            out.append(x & 0xFF)
        _HDR_MASK_CACHE[key] = bytes(out)
    return _HDR_MASK_CACHE[key]


HDR_PHASES = 8      # header whitening cycles over this many patterns


def _hdr_xor(b: bytes, phase: int = 0) -> bytes:
    m = _hdr_mask(len(b), phase)
    return bytes(x ^ y for x, y in zip(b, m))


def pack_header(seq: int, k: int, block_size: int, file_size: int, mode: int,
                zone_w: int = 0, zone_modes: int = 0) -> bytes:
    body = MAGIC + struct.pack("<BBIIHIBBB", 2, mode, seq, k, block_size,
                               file_size, PAYLOAD_ECC, zone_w, zone_modes)
    body += b"\x00" * (HEADER_LEN - 2 - len(body))
    body += struct.pack("<H", zlib.crc32(body) & 0xFFFF)
    # Whiten with a SEQ-DEPENDENT phase. A fixed mask left the band static:
    # only `seq` differs between frames, and the RS codeword is systematic,
    # so the 28 data bytes (all constant but seq) land together and barely
    # change. Measured on the rendered transmit: rows 76 and 78 varied at
    # 29% of a typical payload row while row 77 varied normally — i.e. TWO
    # stationary bands with a moving one between them, exactly as observed.
    # Cycling the mask makes every header cell change frame to frame and
    # pushes template Hamming distance toward the ideal 50%, which is what
    # ML sequence detection discriminates on.
    return _hdr_xor(bytes(RSCodec(HEADER_ECC).encode(body)), seq % HDR_PHASES)


LOCAL_TH = 15    # cells. Window for the local decision threshold; 0 = the v1
                 # global-Otsu behaviour.
                 #
                 # Measured on IMG_7870 (record take, 252x140, 734 frames whose
                 # header decoded): ONE global Otsu over all 35,280 cells
                 # certified 19.8% of codewords. A box mean over a 15x15 cell
                 # neighbourhood certified 42.4% — 2.1x, on identical samples,
                 # for one box filter.
                 #
                 # Why it works: the payload is RS+fountain output, so it is
                 # pseudorandom, so over a few hundred cells its mean converges
                 # on the midpoint between the black and white levels AT THAT
                 # POINT ON THE SCREEN. That makes a local mean a direct
                 # estimate of the local decision threshold, and it tracks
                 # vignetting, backlight non-uniformity, glare and off-axis
                 # roll-off. A global threshold has to be wrong somewhere on any
                 # screen that is not uniformly lit, and no screen filmed by a
                 # hand-held camera is.
                 #
                 # The window is a compromise: too small and the mean starts
                 # tracking the DATA rather than the illumination (local9 scored
                 # 39.0%), too large and it stops tracking the gradient
                 # (local61 scored 22.8%).


def set_local_threshold(k: int) -> None:
    global LOCAL_TH
    LOCAL_TH = int(k)


def local_levels(values: np.ndarray, layout: 'Layout', cells: np.ndarray,
                 k: int):
    """Local (threshold, spread) per cell, estimated in CELL space.

    `cells` may cover only part of the grid (payload cells skip finders, ring,
    separators and the header), so the box filter is normalised by an
    occupancy count rather than by window area — otherwise every cell near a
    hole is biased toward zero.
    """
    g = np.zeros((layout.gh, layout.gw), np.float32)
    m = np.zeros((layout.gh, layout.gw), np.float32)
    g2 = np.zeros((layout.gh, layout.gw), np.float32)
    v = values.astype(np.float32)
    g[cells[:, 0], cells[:, 1]] = v
    g2[cells[:, 0], cells[:, 1]] = v * v
    m[cells[:, 0], cells[:, 1]] = 1.0
    kk = (k, k)
    br = cv2.BORDER_REFLECT
    num = cv2.boxFilter(g, -1, kk, normalize=False, borderType=br)
    num2 = cv2.boxFilter(g2, -1, kk, normalize=False, borderType=br)
    den = np.maximum(cv2.boxFilter(m, -1, kk, normalize=False, borderType=br), 1.0)
    mean = num / den
    var = np.maximum(num2 / den - mean * mean, 1e-6)
    return (mean[cells[:, 0], cells[:, 1]],
            np.sqrt(var)[cells[:, 0], cells[:, 1]])


def _mono_decide(lum: np.ndarray, layout: 'Layout', cells: np.ndarray):
    """Hard bits + per-cell confidence for a two-level alphabet."""
    if LOCAL_TH and layout is not None and cells is not None:
        th, sd = local_levels(lum, layout, cells[: len(lum)], LOCAL_TH)
        return (lum > th).astype(np.uint8), np.abs(lum - th) / np.maximum(sd, 1e-3)
    t, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    spread = max(1e-3, np.percentile(lum, 90) - np.percentile(lum, 10))
    return (lum > t).astype(np.uint8), np.abs(lum - t) / spread


HEADER_CENTERED = True   # v3 layout. Older captures have it at the top edge.


def set_header_centered(on: bool) -> None:
    """Header row placement. Must be set before constructing a Layout.

    Captures made before the centring change carry the header at the top, so
    a receiver has to be able to try both.
    """
    global HEADER_CENTERED
    HEADER_CENTERED = on


def set_header_len(n: int) -> None:
    """Header size, pre-RS. v1 = 24 bytes, v2 = 28 (adds zone fields).

    Changing this changes the grid layout (header_cells), so it must be set
    before constructing a Layout. Videos encoded with an older build carry
    v1 headers; decode.py auto-detects rather than assuming.
    """
    global HEADER_LEN
    HEADER_LEN = n


def unpack_header(raw: bytes):
    # Try whitened first, then raw: captures filmed before whitening was
    # introduced must keep decoding, and one extra RS attempt is free.
    # ML proposes, RS verifies: try each whitening phase, plus the raw path
    # for captures filmed before whitening existed. A phase is only accepted
    # if the seq it decodes to actually belongs to that phase, which makes a
    # false accept require an RS miscorrection AND a phase coincidence.
    body = None
    for ph in list(range(HDR_PHASES)) + [-1, None]:
        cand = raw if ph is None else _hdr_xor(raw, ph)
        try:
            b = bytes(RSCodec(HEADER_ECC).decode(cand)[0])
        except ReedSolomonError:
            continue
        if b[:4] != MAGIC:
            continue
        if ph is not None and ph >= 0 and len(b) >= 10:
            if struct.unpack("<I", b[6:10])[0] % HDR_PHASES != ph:
                continue
        body = b
        break
    if body is None:
        return None
    crc = struct.unpack("<H", body[-2:])[0]
    if zlib.crc32(body[:-2]) & 0xFFFF != crc:
        return None
    ver = body[4]
    if ver >= 2:
        ver, mode, seq, k, block_size, file_size, ecc, zw, zm = \
            struct.unpack("<BBIIHIBBB", body[4:23])
    else:
        ver, mode, seq, k, block_size, file_size, ecc = \
            struct.unpack("<BBIIHIB", body[4:21])
        zw = zm = 0
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
    m = layout.fs
    base = np.zeros((7, 7), dtype=np.float32)
    base[1:-1, 1:-1] = 255.0
    base[2:-2, 2:-2] = 0.0
    tpl = np.kron(base, np.ones((m, m), dtype=np.float32))
    for (r0, c0) in [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]:
        cells[r0:r0 + f, c0:c0 + f] = tpl[..., None]

    # header: mono bits
    hb = _bits(header)
    hc = layout.header_cells[: len(hb)]
    cells[hc[:, 0], hc[:, 1]] = (255.0 * hb)[:, None]
    # Fill the LEFTOVER header cells with the whitening sequence instead of
    # leaving them black. The header band is sized in whole rows, so the
    # remainder is large — 200 of 744 cells at 280x155, i.e. 27% of the band
    # nailed to black in every frame. That is what reads as a dark stationary
    # line across the middle of the picture: measured 72 luma against 127 for
    # payload rows. The receiver only ever reads the first len(hb) cells, so
    # this is photometric only and changes no protocol.
    rest = layout.header_cells[len(hb):]
    if len(rest):
        filler = _bits(_hdr_mask(-(-len(rest) // 8) + 1,
                                 sum(header) % HDR_PHASES))[: len(rest)]
        cells[rest[:, 0], rest[:, 1]] = (255.0 * filler)[:, None]

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
    pts = _apply_radial(pts, img.shape)
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
    return [(x, y) for _, x, y in out[:12]]


def _best_quad(pts, aspect):
    """Choose the 4 points that actually look like the code's corners.

    Taking the top 4 by area is wrong twice over, and both failures were
    observed on real frames: a Dock icon outranks a true marker (rounded
    nested squares are the same pattern), and on a clean render a chance
    nested quad in the random payload displaced the top-left finder, after
    which the sum/diff ordering assigned one point to both TL and BL.

    Score every 4-subset by how close it is to a parallelogram of the right
    aspect, which is what four corners of a plane actually project to.
    """
    import itertools
    best, best_score = None, -1.0
    for combo in itertools.combinations(range(len(pts)), 4):
        q = pts[list(combo)]
        s_ = q.sum(axis=1)
        d_ = q[:, 0] - q[:, 1]
        tl, br = q[np.argmin(s_)], q[np.argmax(s_)]
        tr, bl = q[np.argmax(d_)], q[np.argmin(d_)]
        dst = np.array([tl, tr, bl, br], dtype=np.float32)
        if min(np.hypot(*(dst[i] - dst[j]))
               for i in range(4) for j in range(i + 1, 4)) < 8.0:
            continue
        w = (np.hypot(*(dst[1] - dst[0])) + np.hypot(*(dst[3] - dst[2]))) / 2
        h = (np.hypot(*(dst[2] - dst[0])) + np.hypot(*(dst[3] - dst[1]))) / 2
        if w < 20 or h < 20:
            continue
        # parallelogram-ness: opposite sides should match
        par = (abs(np.hypot(*(dst[1] - dst[0])) - np.hypot(*(dst[3] - dst[2]))) / w +
               abs(np.hypot(*(dst[2] - dst[0])) - np.hypot(*(dst[3] - dst[1]))) / h)
        asp = abs((w / h) - aspect) / aspect
        score = w * h / (1.0 + 6.0 * par + 6.0 * asp)
        if score > best_score:
            best, best_score = dst, score
    return best


def locate(img: np.ndarray, layout: Layout):
    """Return homography unit-grid -> image, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pts = _finder_centers(gray)
    if len(pts) < 4:
        return None
    pts = np.array(pts, dtype=np.float32)
    f = layout.finder
    span_w = layout.gw - 1 - f      # centre-to-centre in cells
    span_h = layout.gh - 1 - f
    dst_best = _best_quad(pts, span_w / max(span_h, 1))
    if dst_best is None:
        return None
    tl, tr, bl, br = dst_best
    fc = f / 2.0  # finder center offset from its zone origin, in cells
    src = np.array([
        [1 + fc, 1 + fc],
        [layout.gw - 1 - f + fc, 1 + fc],
        [1 + fc, layout.gh - 1 - f + fc],
        [layout.gw - 1 - f + fc, layout.gh - 1 - f + fc],
    ], dtype=np.float32)
    dst = np.array([tl, tr, bl, br], dtype=np.float32)

    # Reject degenerate quads. The sum/diff ordering happily returns the SAME
    # point as both TL and BL when the candidates are nearly collinear, and
    # getPerspectiveTransform then yields a garbage homography that every
    # downstream stage reports as "located" — which is how a failed take was
    # scored as 100% localization on six different profiles at once.
    # Real cause of that collinearity: the macOS Dock was in frame and its
    # rounded-square icons are textbook nested-contour finder candidates,
    # outranking the true markers by area.
    for i in range(4):
        for j in range(i + 1, 4):
            if np.hypot(*(dst[i] - dst[j])) < 8.0:
                return None
    quad = dst[[0, 1, 3, 2]].astype(np.float32)     # TL, TR, BR, BL
    area = abs(cv2.contourArea(quad))
    side = np.hypot(*(dst[1] - dst[0])) * np.hypot(*(dst[2] - dst[0]))
    if area < 0.25 * max(side, 1.0):                # near-collinear
        return None
    return cv2.getPerspectiveTransform(src, dst)


RADIAL_K1 = 0.0   # lens radial distortion, normalized by image width
_FAST_SAMPLE = True   # grayscale box-filter + single gather (see sample_cells)
_BOX_CACHE = {}


def set_fast_sample(on: bool) -> None:
    """Fast path is grayscale-only. Colour modes must disable it."""
    global _FAST_SAMPLE
    _FAST_SAMPLE = on


def set_radial(k1: float) -> None:
    """Set the radial distortion coefficient used when sampling cells.

    A homography maps plane to plane, but a real wide-angle phone lens bends
    straight lines, so sample points drift progressively off cell centres
    toward the frame edges. Measured on real 4K handheld footage at 252 cells
    across: the middle of the code decoded at 0.0% error while the left/right
    edge columns failed at 11-18%. Correcting with a single k1 term took that
    frame from 3.73% to 0.44% bit error, an 8.5x reduction, and flattened the
    error map. This was the density wall, not motion and not exposure.
    """
    global RADIAL_K1
    RADIAL_K1 = k1


def _apply_radial(pts: np.ndarray, shape) -> np.ndarray:
    if RADIAL_K1 == 0.0:
        return pts
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx = (pts[:, 0] - cx) / w
    dy = (pts[:, 1] - cy) / w
    f = 1.0 + RADIAL_K1 * (dx * dx + dy * dy)
    out = np.empty_like(pts)
    out[:, 0] = cx + (pts[:, 0] - cx) * f
    out[:, 1] = cy + (pts[:, 1] - cy) * f
    return out


def estimate_radial(img: np.ndarray, layout: Layout, H: np.ndarray,
                    lo: float = -0.04, hi: float = 0.10, steps: int = 29):
    """Self-calibrate k1 from the capture itself, with no ground truth.

    Sweeps k1 and keeps the value that maximizes bimodality of the sampled
    cell luminances (Otsu between-class variance). Correct geometry samples
    cell centres, giving two tight populations; wrong geometry samples across
    cell boundaries and smears them together. No reference image needed, so
    it runs at receive time on any device pair.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    grayf = gray.astype(np.float32)
    h, w = gray.shape
    ctr = np.stack([layout.payload_cells[:, 1] + 0.5,
                    layout.payload_cells[:, 0] + 0.5], axis=1).astype(np.float32)
    base = cv2.perspectiveTransform(ctr[None], H.astype(np.float32))[0]
    cx, cy = w / 2.0, h / 2.0
    best_k, best_score = 0.0, -1.0
    prev = RADIAL_K1
    for k1 in np.linspace(lo, hi, steps):
        dx = (base[:, 0] - cx) / w
        dy = (base[:, 1] - cy) / w
        f = 1.0 + k1 * (dx * dx + dy * dy)
        px = np.clip((cx + (base[:, 0] - cx) * f).round().astype(int), 1, w - 2)
        py = np.clip((cy + (base[:, 1] - cy) * f).round().astype(int), 1, h - 2)
        lum = grayf[py, px]
        u8 = np.clip(lum, 0, 255).astype(np.uint8)
        th, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a, b = lum[lum <= th], lum[lum > th]
        if len(a) < 10 or len(b) < 10:
            continue
        wa, wb = len(a) / len(lum), len(b) / len(lum)
        score = wa * wb * (a.mean() - b.mean()) ** 2   # between-class variance
        if score > best_score:
            best_k, best_score = float(k1), float(score)
    set_radial(prev)
    return best_k


def estimate_psf_from_finders(img: np.ndarray, layout: Layout, H: np.ndarray,
                              size: int = 9):
    """Measure the camera's blur kernel using the finder patterns as a probe.

    Every frame carries four 7x7 finder patterns whose exact shape the receiver
    already knows. That makes them a free, in-band point-spread-function probe:
    render the ideal finder, warp it through the same homography, and solve for
    the kernel that turns ideal into observed. Blind deconvolution made
    non-blind by the fiducials.

    Motivation from measurement: ~26% of handheld captures are discarded as
    motion-blurred (BER > 3%) while the sharp ones sit at 0.4%. Blur is a
    linear, invertible corruption, so those frames are not noise — they are
    recoverable signal that the receiver currently throws away.

    Returns a (size, size) kernel normalised to sum 1, or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gh, gw, f = layout.gh, layout.gw, layout.finder
    ideal_cells = np.zeros((gh, gw), np.float32)
    tpl = np.zeros((f, f), np.float32)
    tpl[1:-1, 1:-1] = 255.0
    tpl[2:-2, 2:-2] = 0.0
    corners = [(1, 1), (1, gw - 1 - f), (gh - 1 - f, 1), (gh - 1 - f, gw - 1 - f)]
    for (r0, c0) in corners:
        ideal_cells[r0:r0 + f, c0:c0 + f] = tpl
    # upsample the ideal grid to image scale, warp with the SAME homography
    scale = 8
    big = cv2.resize(ideal_cells, (gw * scale, gh * scale),
                     interpolation=cv2.INTER_NEAREST)
    S = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]])
    Hs = H @ S
    warped = cv2.warpPerspective(big, Hs, (gray.shape[1], gray.shape[0]))

    patches_i, patches_o = [], []
    pad = size
    for (r0, c0) in corners:
        pt = cv2.perspectiveTransform(
            np.array([[[c0 + f / 2.0, r0 + f / 2.0]]], np.float32),
            H.astype(np.float32))[0][0]
        x, y = int(round(pt[0])), int(round(pt[1]))
        # patch big enough to contain the finder plus kernel support
        half = int(abs(H[0, 0]) * f * 0.9) + pad
        if x - half < 0 or y - half < 0 or \
           x + half >= gray.shape[1] or y + half >= gray.shape[0]:
            continue
        patches_i.append(warped[y - half:y + half, x - half:x + half])
        patches_o.append(gray[y - half:y + half, x - half:x + half].astype(np.float32))
    if not patches_i:
        return None

    # Solve min ||conv(ideal, k) - observed||^2 in the frequency domain,
    # accumulated over all available finders (Wiener-style, regularised).
    num = None
    den = None
    for I, O in zip(patches_i, patches_o):
        I = I - I.mean()
        O = O - O.mean()
        FI = np.fft.rfft2(I)
        FO = np.fft.rfft2(O)
        num = FI.conj() * FO if num is None else num + FI.conj() * FO
        den = np.abs(FI) ** 2 if den is None else den + np.abs(FI) ** 2
    K = np.fft.irfft2(num / (den + 1e-2 * den.max()), s=patches_i[0].shape)
    K = np.fft.fftshift(K)
    c = K.shape[0] // 2, K.shape[1] // 2
    h = size // 2
    k = K[c[0] - h:c[0] + h + 1, c[1] - h:c[1] + h + 1].astype(np.float32)
    if k.size == 0 or not np.isfinite(k).all():
        return None
    s = k.sum()
    if abs(s) < 1e-6:
        return None
    return k / s


def deconvolve(img: np.ndarray, kernel: np.ndarray, iters: int = 12):
    """Richardson-Lucy deconvolution with the measured kernel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    obs = gray.astype(np.float32) + 1e-3
    est = obs.copy()
    kf = kernel.astype(np.float32)
    kfm = kf[::-1, ::-1].copy()
    for _ in range(iters):
        conv = cv2.filter2D(est, -1, kf, borderType=cv2.BORDER_REPLICATE)
        rel = obs / np.maximum(conv, 1e-3)
        est *= cv2.filter2D(rel, -1, kfm, borderType=cv2.BORDER_REPLICATE)
        np.clip(est, 0, 255, out=est)
    return est


def refine_homography(img: np.ndarray, layout: Layout, H: np.ndarray,
                      span: float = 2.5, rounds: int = 3):
    """Subpixel-refine H against every cell whose value we already know.

    A homography fitted from four finder centres has fixed pixel accuracy, but
    the DAMAGE that accuracy does scales with density: at 252 cells a one-cell
    sampling error needs ~13 px of geometric error, at 466 cells only ~7.45 px.
    Measured consequence: 0.4% BER at 252 cells, 14% at 466 with identical
    optics and exposure.

    So refine. The timing ring and the four finder patterns are hundreds of
    cells whose values are known a priori, which is enough signal to hill-climb
    the four corner correspondences to subpixel precision. This is the same
    role QR alignment patterns play in production decoders.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    grayf = gray.astype(np.float32)
    h_, w_ = gray.shape
    cells, exp = known_cells(layout)
    centers = np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5],
                       axis=1).astype(np.float32)
    f = layout.finder
    fc = f / 2.0
    src = np.array([
        [1 + fc, 1 + fc],
        [layout.gw - 1 - f + fc, 1 + fc],
        [1 + fc, layout.gh - 1 - f + fc],
        [layout.gw - 1 - f + fc, layout.gh - 1 - f + fc],
    ], dtype=np.float32)
    dst = cv2.perspectiveTransform(src.reshape(1, 4, 2),
                                   H.astype(np.float32)).reshape(4, 2)

    def score(d):
        Hc = cv2.getPerspectiveTransform(src, d.astype(np.float32))
        pts = cv2.perspectiveTransform(centers[None], Hc)[0]
        pts = _apply_radial(pts, img.shape)
        xs = np.clip(pts[:, 0].round().astype(np.int32), 0, w_ - 1)
        ys = np.clip(pts[:, 1].round().astype(np.int32), 0, h_ - 1)
        v = grayf[ys, xs]
        # separation between cells known-white and known-black
        return float(v[exp].mean() - v[~exp].mean())

    best = score(dst)
    step = span
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    cand = dst.copy()
                    cand[i] += (dx, dy)
                    s = score(cand)
                    if s > best:
                        best, dst, improved = s, cand, True
        step *= 0.4
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def sample_cells(img: np.ndarray, layout: Layout, H: np.ndarray, cells: np.ndarray):
    """Sample given (r,c) cells; returns float32 (n, 3) BGR means of 3x3 patches."""
    centers = np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5], axis=1).astype(np.float32)
    pts = cv2.perspectiveTransform(centers[None], H)[0]
    pts = _apply_radial(pts, img.shape)
    if _FAST_SAMPLE:
        # ONE box-filter over the image then ONE gather, instead of nine
        # gathers over ~170k cells. Measured at 29.2 ms/frame on 4K, which was
        # the wall for real-time decoding: box filtering is SIMD and a single
        # gather is far more cache-friendly.
        h_, w_ = img.shape[:2]
        xs = np.clip(pts[:, 0].round().astype(np.int32), 1, w_ - 2)
        ys = np.clip(pts[:, 1].round().astype(np.int32), 1, h_ - 2)
        blurred = _BOX_CACHE.get("img")
        if blurred is None or _BOX_CACHE.get("id") is not id(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            blurred = cv2.boxFilter(g, cv2.CV_32F, (3, 3))
            _BOX_CACHE.clear()
            _BOX_CACHE["img"] = blurred
            _BOX_CACHE["id"] = id(img)
        v = blurred[ys, xs]
        return np.repeat(v[:, None], 3, axis=1)
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
    # The header is always a TWO-level (mono) sub-code, even when the payload
    # is gray4/color. Thresholding it against the payload's own histogram is
    # therefore wrong whenever the payload has more than two levels: the
    # multi-level population drags the split away from the header's own eye.
    # Measured on a real gray4 capture: global Otsu read 0/6 headers, a
    # header-only Otsu read 1/6 (the one it recovered agreed exactly with ML
    # sequence detection).
    hdr_th, _ = cv2.threshold(hdr_lum.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    spread = max(1e-3, np.percentile(pay_lum, 90) - np.percentile(pay_lum, 10))
    stats["cell_margin"] = float(np.mean(np.abs(pay_lum - th)) / spread)

    n_hdr_bits = (HEADER_LEN + HEADER_ECC) * 8
    header = None
    # Local thresholds FIRST, then the two global ones. The header strip spans
    # the full width of the grid, which is the worst possible shape for a
    # global threshold — it eats the entire left-to-right illumination
    # gradient. Measured over 400 located frames of the record take: global
    # over everything 36.5%, global over the strip 41.8%, local-31 50.0%,
    # union of all rules 51.5%, with ZERO disagreement on the recovered `seq`
    # between any two rules that both parsed. Header yield multiplies codeword
    # yield, so this factor was costing more than the payload demodulator.
    cand = []
    if LOCAL_TH:
        allc = np.concatenate([layout.header_cells, layout.payload_cells])
        alllum = np.concatenate([hdr_lum, pay_lum])
        nh = len(hdr_lum)
        for k in (31, 15):
            lt, _sd = local_levels(alllum, layout, allc, k)
            cand.append(lt[:nh])
    cand += [hdr_th, th]
    for t in cand:
        header = unpack_header(_bytes((hdr_lum[:n_hdr_bits] >
                                       (t[:n_hdr_bits] if np.ndim(t) else t)
                                       ).astype(np.uint8)))
        if header is not None:
            break
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


def raw_bits_and_conf(header: dict, pay_samples: np.ndarray,
                      layout: 'Layout' = None):
    """Demodulate to raw bytes plus per-byte confidence, WITHOUT running RS.

    Sub-block recovery needs the pre-ECC bitstream so it can split it into
    codewords and decode each independently; decide_payload folds RS in and
    returns all-or-nothing. Confidence is the distance to the nearest decision
    boundary, which is what lets the RS layer place erasures intelligently.
    """
    pay_lum = pay_samples.mean(axis=1)
    mode = header["mode"]
    if mode == MODE_GRAY4:
        v = pay_lum.astype(np.float32).reshape(-1, 1)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
        _r, lab, cent = cv2.kmeans(v, 4, None, crit, 6, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(cent.ravel())
        rank = np.empty(4, np.int64)
        rank[order] = np.arange(4)
        syms = rank[lab.ravel()]
        bits = np.array([GRAY4_BITS[int(x)] for x in syms], dtype=np.uint8).reshape(-1)
        c = np.sort(cent.ravel())
        bounds = (c[:-1] + c[1:]) / 2.0
        dist = np.min(np.abs(pay_lum[:, None] - bounds[None]), axis=1)
        conf = np.repeat(dist / max(1e-3, c[-1] - c[0]), 2)
    else:
        bits, conf = _mono_decide(pay_lum, layout,
                                  None if layout is None else layout.payload_cells)
    raw = _bytes(bits)
    nb = min(len(raw), len(conf) // 8)
    byte_conf = conf[: nb * 8].reshape(nb, 8).min(axis=1)
    return raw, byte_conf


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
        # 1-D k-means rather than quantile slicing: quantiles collapse to empty
        # slices whenever many cells share a value (which happens on any clean
        # frame), and that produced NaN centres. k-means is robust to that and
        # still adapts to whatever gamma/exposure the channel delivered.
        v = pay_lum.astype(np.float32).reshape(-1, 1)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
        _r, lab, cent = cv2.kmeans(v, 4, None, crit, 6, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(cent.ravel())          # dark -> bright
        rank = np.empty(4, np.int64)
        rank[order] = np.arange(4)
        centers = np.sort(cent.ravel())
        syms = rank[lab.ravel()]
        bits = np.array([GRAY4_BITS[int(s)] for s in syms], dtype=np.uint8).reshape(-1)
        raw = _bytes(bits)
        bounds = (centers[:-1] + centers[1:]) / 2
        dist = np.min(np.abs(pay_lum[:, None] - bounds[None]), axis=1)
        spread = max(1e-3, centers[-1] - centers[0])
        cm = dist / spread
        margins = np.repeat(cm, 2)   # each cell contributes 2 bits
    elif header["mode"] == MODE_MONO:
        bits, margins = _mono_decide(
            pay_lum, layout, None if layout is None else layout.payload_cells)
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
