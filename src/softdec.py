#!/usr/bin/env python3
"""
softdec.py — one frame's worth of the certified-label receiver.

This is the mechanism that survived every negative result in this project
(Findings §14 for why nothing self-supervised works here, §3 for why the
fountain+CRC structure makes the labels exactly correct by construction),
lifted out of src/exp_probe_soft.py so the PRODUCTION decoder can run it
instead of only the experiments.

Per frame, in order:
  1. local-threshold hard decisions            (grid._mono_decide)
  2. certify each RS codeword, with soft erasures placed by confidence
  3. if a donor kernel exists, re-detect with tile-PRML and certify again,
     keeping whichever pass certified more
  4. if this frame certified enough of ITSELF, it becomes the next donor:
     kernels are fitted on the certified cells plus the structure the
     receiver knows a priori, and on nothing else

Step 4 is the honesty constraint. A receiver does not have the source file,
so it cannot reconstruct a frame's true bits from `seq`; it only knows the
cells whose codeword passed both RS and its own CRC32. Fitting on anything
else is leakage, and an earlier version of this experiment did exactly that.

Measured on IMG_7870 (record take, 100 contiguous header-bearing frames):

    global Otsu (what production shipped)      19.8% of codewords
    local threshold                            39.2%
    local + soft erasures                      45.2%
    local + erasures + rolling donor           63.1%

Erasures are not an optimisation here, they are the bootstrap: without them
no frame ever reaches the refit threshold, the donor never arms, and the
PRML pass contributes exactly nothing (measured: 0 donors, 39.2%).
"""
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
try:
    # creedsolo is the cythonized build of the same library: bit-identical
    # (verified on randomized errors+erasures), ~5x per decode call, and RS
    # decode is 79% of this decoder's wall clock.
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid

# crs2: line-by-line C port of reedsolo's decoder plus the certify ladder.
# 10,000/10,000 differential-identical, 17x on ladder-heavy codewords.
# Loaded when present; the Python path below stays as the exact fallback.
import ctypes as _ct
_CRS2 = None
try:
    _CRS2 = _ct.CDLL(str(Path(__file__).parent / "crs2.dylib"))
    _CRS2.certify_codeword.restype = _ct.c_int
    _CRS2.certify_codeword.argtypes = [
        _ct.c_char_p, _ct.POINTER(_ct.c_int32), _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_char_p, _ct.c_char_p]
except OSError:
    _CRS2 = None
import exp_tile_prml as T
import exp_turbo_frame as TB

# ---- vectorized GF(2^8) syndrome screen (prim 0x11d, generator 2, fcr 0)
# One numpy pass computes all syndromes of all codewords in a frame. A
# codeword whose syndromes are all zero IS its own decode (reedsolo returns
# the message unchanged when max(synd)==0), so the C decoder is skipped for
# it entirely - which on a good frame is most of them. Bit-identical by
# construction; only the arithmetic route changes.
_GF_EXP = np.zeros(512, np.uint8)
_GF_LOG = np.zeros(256, np.int16)
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
_GF_EXP[255:510] = _GF_EXP[:255]
# _SYN_POW[i, j] = (i * (254 - j)) % 255 : exponent of alpha^i at msg pos j
_SYN_POW = (np.arange(48)[:, None] * (254 - np.arange(255))[None, :]) % 255


def syndromes_zero(chunks_u8, nsym):
    """chunks_u8: (n, 255) uint8. Returns (n,) bool: True = clean codeword."""
    logm = _GF_LOG[chunks_u8].astype(np.int32)          # (n, 255)
    e = (logm[:, None, :] + _SYN_POW[None, :nsym, :]) % 255
    terms = _GF_EXP[e]                                   # (n, nsym, 255)
    terms = np.where(chunks_u8[:, None, :] == 0, 0, terms)
    synd = np.bitwise_xor.reduce(terms, axis=2)          # (n, nsym)
    return ~synd.any(axis=1)


class FrameDecoder:
    """Stateful across frames: holds the rolling kernel donor."""

    def __init__(self, layout, ecc, n_sub, tiles=(8, 14), sweeps=3,
                 refit=None, erase=True, prml=True, bits_per_cell=1):
        self.L = layout
        self.ecc = ecc
        self.rs = RSCodec(ecc)
        self.n_sub = n_sub
        self.sub_size = (255 - ecc) - 4
        # Cells per byte follows the ALPHABET, not the byte: mono spends 8
        # cells on a byte, gray4 spends 4. Hardcoding 8 sized every cell-index
        # array for mono, so a gray4 frame indexed past the end of its own
        # payload region and raised on the first codeword past halfway.
        self.bpc = bits_per_cell
        self.cells_per_byte = 8 // bits_per_cell
        self.cells = layout.payload_cells[: n_sub * 255 * self.cells_per_byte]
        self.known = (layout.is_finder | layout.is_sep | layout.is_ring |
                      layout.is_header)
        self.tiles = tiles
        self.sweeps = sweeps
        self.refit = n_sub - 2 if refit is None else refit
        self.erase = erase
        self.prml = prml
        self.tap = self.bias = None
        self.donors = 0
        self.pending = None
        self.pending_whole = None
        self.first_donor = None   # (tap, bias, frame) of the first arm
        self.donor_frame = 0      # caller stamps the frame number
        # Vertical offsets from the 12px-era measurement ("dx drift nil"),
        # PLUS horizontal and diagonal ones: at the vertical centre the
        # radial residual points HORIZONTALLY, invisible to a vertical-only
        # search, and at 9px margins that kills the middle bands (measured:
        # 38-55% mid-frame vs 100% top/bottom at 9.9 cam-px/cell, while
        # 11px shows no dip). CRC-gated hypotheses can only add.
        self.geom_offsets = [(0.0, d) for d in
                             (0.20, -0.20, 0.35, -0.35, 0.10, -0.10)] +                             [(d, 0.0) for d in
                             (0.20, -0.20, 0.35, -0.35)] +                             [(dx, dy) for dx in (0.25, -0.25)
                             for dy in (0.25, -0.25)]
        self._blank = np.zeros(layout.payload_capacity_bytes(grid.MODE_MONO),
                               np.uint8).tobytes()
        self._st_cache = {}
        # CERTIFIED-INTERFERER CANCELLATION state: the previous frame's
        # certified cell labels (seq, cmask, cbits). See cancel_prev().
        self.prev_labels = None

    # ---- structure the receiver knows before decoding anything
    def struct_truth(self, header):
        # Pure function of these four ints; render_frame+pack_header cost
        # ~105 ms and were 36% of decode wall time, called twice per frame
        # (PRML pass + commit) on headers that recur every transmit loop.
        key = (int(header["seq"]), int(header["k"]),
               int(header["block_size"]), int(header["file_size"]))
        hit = self._st_cache.get(key)
        if hit is not None:
            return hit
        raw = grid.pack_header(key[0], key[1], key[2], key[3],
                               grid.MODE_MONO, 0, 0)
        img = grid.render_frame(self.L, raw, self._blank, grid.MODE_MONO,
                                cell_px=1)
        out = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
        if len(self._st_cache) > 4096:
            self._st_cache.clear()
        self._st_cache[key] = out
        return out

    # ---- certify every codeword in one set of hard decisions
    def certify(self, bits, byteconf=None, only=None):
        by = np.packbits(bits[: self.n_sub * 255 * 8].astype(np.uint8))
        blocks = []
        cmask = np.zeros(len(self.cells), bool)
        cbits = np.zeros(len(self.cells), np.float32)
        js = list(range(self.n_sub) if only is None else only)
        if _CRS2 is not None:
            blk_buf = _ct.create_string_buffer(self.sub_size)
            cod_buf = _ct.create_string_buffer(255)
            for j in js:
                lo, hi = j * 255, (j + 1) * 255
                chunk = bytes(by[lo:hi])
                use_ladder = 1 if (self.erase and byteconf is not None) else 0
                if use_ladder:
                    order = np.ascontiguousarray(
                        np.argsort(byteconf[lo:hi]), dtype=np.int32)
                    optr = order.ctypes.data_as(_ct.POINTER(_ct.c_int32))
                else:
                    optr = None
                ok = _CRS2.certify_codeword(chunk, optr, self.ecc,
                                            self.sub_size, use_ladder,
                                            blk_buf, cod_buf)
                if not ok:
                    continue
                blocks.append((j, blk_buf.raw))
                if self.bpc == 1:
                    cmask[lo * 8:hi * 8] = True
                    cbits[lo * 8:hi * 8] = np.unpackbits(
                        np.frombuffer(cod_buf.raw, np.uint8))
            return blocks, cmask, cbits
        clean = syndromes_zero(by[: self.n_sub * 255].reshape(
            self.n_sub, 255)[js], self.ecc) if js else []
        for ci, j in enumerate(js):
            lo, hi = j * 255, (j + 1) * 255
            chunk = bytes(by[lo:hi])
            dec = None
            if clean[ci]:
                dec = chunk[: 255 - self.ecc]   # zero syndromes: decode is
                                                # the message itself
            else:
              try:
                dec = bytes(self.rs.decode(chunk)[0])
              except ReedSolomonError:
                if self.erase and byteconf is not None:
                    order = np.argsort(byteconf[lo:hi])
                    for n_er in range(4, int(self.ecc * 0.7) + 1, 6):
                        try:
                            dec = bytes(self.rs.decode(
                                chunk,
                                erase_pos=[int(i) for i in order[:n_er]])[0])
                            break
                        except ReedSolomonError:
                            continue
            if dec is None or len(dec) < 4 + self.sub_size:
                continue
            blk = dec[4:4 + self.sub_size]
            if zlib.crc32(blk) & 0xFFFFFFFF != struct.unpack("<I", dec[:4])[0]:
                continue
            blocks.append((j, blk))
            if self.bpc == 1:
                # Certified LABELS are per-cell binary, so they only mean
                # anything for a two-level alphabet. gray4 cells carry a
                # 4-level symbol; teaching the binary tile-PRML from them would
                # need a 4-level Viterbi, which does not exist yet. Codeword
                # counting (what the probe measures) is unaffected.
                coded = bytes(self.rs.encode(dec))
                cmask[lo * 8:hi * 8] = True
                cbits[lo * 8:hi * 8] = np.unpackbits(
                    np.frombuffer(coded, np.uint8))
        return blocks, cmask, cbits

    # ---- whole-block transmits (one fountain block per frame, one CRC)
    def _whole_from_bits(self, bits, byteconf, bs):
        """RS+CRC a whole-frame block. Returns the payload bytes or None."""
        coded_len = grid.rs_encoded_len(bs + 4)
        raw = grid._bytes(np.asarray(bits).astype(np.uint8))[:coded_len]
        if len(raw) < coded_len:
            return None
        out = bytearray()
        pos = 0
        while pos < coded_len:
            n = min(255, coded_len - pos)
            chunk = raw[pos:pos + n]
            dec = None
            try:
                dec = bytes(self.rs.decode(chunk)[0])
            except ReedSolomonError:
                if self.erase and byteconf is not None:
                    order = np.argsort(byteconf[pos:pos + n])
                    for n_er in range(4, int(self.ecc * 0.7) + 1, 6):
                        try:
                            dec = bytes(self.rs.decode(
                                chunk,
                                erase_pos=[int(i) for i in order[:n_er]])[0])
                            break
                        except ReedSolomonError:
                            continue
            if dec is None:
                return None
            out += dec
            pos += n
        payload = bytes(out)
        if len(payload) < 4 + bs:
            return None
        blk = payload[4:4 + bs]
        if zlib.crc32(blk) & 0xFFFFFFFF != struct.unpack("<I", payload[:4])[0]:
            return None
        return payload

    def decode_whole(self, y, header, allow_refit=True, resample=None):
        """One fountain block per frame. Returns block bytes or None.

        A whole-block transmit has a single CRC over the frame, so a frame
        either certifies ENTIRELY or not at all. That is worse for harvesting -
        there is no partial credit - but strictly better for teaching: a frame
        that passes hands the equalizer its complete transmitted bit pattern,
        every payload cell of it, instead of the fraction that per-codeword
        certification recovers. The donor this produces is exact.
        """
        pc = self.L.payload_cells
        lum = y[pc[:, 0], pc[:, 1]]
        bits1d, conf1d = grid._mono_decide(lum, self.L, pc)
        bs = int(header["block_size"])
        nb = grid.rs_encoded_len(bs + 4)
        bc = conf1d[: nb * 8].reshape(nb, 8).min(axis=1) if self.erase else None

        payload = self._whole_from_bits(bits1d, bc, bs)
        x_used = np.zeros(y.shape, np.float32)
        x_used[pc[:, 0], pc[:, 1]] = bits1d

        if payload is None and self.prml and self.tap is not None:
            xt = self.struct_truth(header)
            x1 = T.prml_tiles(y, self.tap, self.bias, self.known, xt, x_used,
                              sweeps=self.sweeps)
            p1 = self._whole_from_bits(x1[pc[:, 0], pc[:, 1]], bc, bs)
            if p1 is not None:
                payload, x_used = p1, x1

        # Same code-validated geometry search as the per-codeword path, at
        # whole-frame granularity because that is the granularity the CRC
        # covers here. A frame that certifies at ANY offset is correct.
        if payload is None and resample is not None:
            for (dx, dy) in self.geom_offsets:
                y2 = resample(dx, dy)
                l2 = y2[pc[:, 0], pc[:, 1]]
                b2, c2 = grid._mono_decide(l2, self.L, pc)
                bc2 = (c2[: nb * 8].reshape(nb, 8).min(axis=1)
                       if self.erase else None)
                p2 = self._whole_from_bits(b2, bc2, bs)
                if p2 is not None:
                    payload = p2
                    x_used = np.zeros(y.shape, np.float32)
                    x_used[pc[:, 0], pc[:, 1]] = b2
                    y = y2
                    break

        self.pending_whole = (y, header, payload) if payload is not None else None
        if payload is not None and allow_refit:
            self.commit_whole()
        return None if payload is None else payload[4:4 + bs]

    def commit_whole(self):
        if not self.prml or getattr(self, "pending_whole", None) is None:
            return False
        y, header, payload = self.pending_whole
        coded = bytes(self.rs.encode(payload))
        tb = np.unpackbits(np.frombuffer(coded, np.uint8)).astype(np.float32)
        pc = self.L.payload_cells[: len(tb)]
        xt = self.struct_truth(header)
        lab = np.zeros(y.shape, np.float32)
        lab[pc[:, 0], pc[:, 1]] = tb[: len(pc)]
        lab[self.known] = xt[self.known]
        sel = self.known.copy()
        sel[pc[:, 0], pc[:, 1]] = True
        self.tap, self.bias = TB.fit_tiles_sel(y, lab, sel, *self.tiles)
        self.donors += 1
        # Remember the FIRST kernel that ever armed, so a caller can go back
        # and re-decode the stretch that ran before it existed. Kernels only
        # transfer for about a second, so the earliest one is the only useful
        # teacher for the frames just before it.
        if getattr(self, "first_donor", None) is None:
            self.first_donor = (self.tap, self.bias, self.donor_frame)
        return True

    # ---- CERTIFIED-INTERFERER CANCELLATION (oracle 2026-08-04: mixed
    # strobe frames 14.1 -> 17.1 of 19, worst frames +10/+11; clean frames
    # exact no-op, zero losses on 48 real probes)
    #
    # NOT the refuted SIC: that subtracted the strong component to chase the
    # weak one under the noise floor. Here the interferer is the PREVIOUS
    # code frame, whose cells the receiver certified through RS+CRC32 one
    # camera frame earlier; re-encoding certified codewords reproduces the
    # transmitted cells exactly (the donor/CAG structural argument). We fit
    # its amplitude on the covered cells and subtract, then re-certify what
    # the plain pass missed. The decoded component is the STRONG one.
    def cancel_prev(self, lum, header, byteconf_shape_nb):
        if self.prev_labels is None:
            return None
        pseq, pmask, pbits = self.prev_labels
        if int(header["seq"]) != pseq + 1 or pmask.sum() < 2000:
            return None
        n_cells = len(self.cells)
        idx = np.flatnonzero(pmask)
        # amplitude of the interferer, fitted only on its certified cells
        cy = lum[:n_cells][idx]
        cx = pbits[idx]
        A = np.stack([cx, np.ones_like(cx)], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, cy, rcond=None)
        except np.linalg.LinAlgError:
            return None
        a = float(coef[0])
        # a is the mixing amplitude of the PREVIOUS frame inside this
        # exposure. Near zero means no straddle: nothing to cancel.
        if a < 8.0:
            return None
        out = lum.copy()
        out[:n_cells][idx] = lum[:n_cells][idx] - a * pbits[idx]
        return out

    def note_labels(self, header, cmask, cbits):
        """Remember this frame's certified cells for the next frame."""
        if cmask is not None and cmask.any():
            self.prev_labels = (int(header["seq"]), cmask, cbits)
        else:
            self.prev_labels = None

    def quick_count(self, y):
        """How many codewords certify from hard decisions alone.

        Used to choose between candidate geometries before paying for the
        PRML pass. Cheap: one box filter, one threshold, n_sub RS decodes.
        """
        pc = self.L.payload_cells
        lum = y[pc[:, 0], pc[:, 1]]
        bits1d, conf1d = grid._mono_decide(lum, self.L, pc)
        nb = self.n_sub * 255
        bc = conf1d[: nb * 8].reshape(nb, 8).min(axis=1) if self.erase else None
        return len(self.certify(bits1d, bc)[0])

    # ---- the whole per-frame path
    def decode(self, y, header, allow_refit=True, resample=None):
        """y: (gh, gw) raw cell luminance. Returns [(sub_index, block_bytes)].

        allow_refit=False evaluates the frame WITHOUT letting it become the
        kernel donor. A caller sweeping candidate geometries must use that:
        every candidate but one is wrong by construction, and letting a wrong
        one teach the equalizer poisons every later frame. Sweep with
        allow_refit=False, then call commit() once on the winner.
        """
        pc = self.L.payload_cells
        lum = y[pc[:, 0], pc[:, 1]]
        bits1d, conf1d = grid._mono_decide(lum, self.L, pc)
        x0 = np.zeros(y.shape, np.float32)
        x0[pc[:, 0], pc[:, 1]] = bits1d
        nb = self.n_sub * 255
        bc = conf1d[: nb * 8].reshape(nb, 8).min(axis=1) if self.erase else None

        blocks, cmask, cbits = self.certify(bits1d, bc)
        best = (len(blocks), blocks, cmask, cbits, x0)

        # Nothing left to win on this frame, so do not pay 192 ms for the
        # equalizer. Half the frames of a good take certify in full from hard
        # decisions alone, and PRML was running on every one of them.
        if self.prml and self.tap is not None and len(blocks) >= self.n_sub:
            self.pending = (y, header, len(blocks), cmask, cbits, x0)
            if allow_refit:
                self.commit()
                self.note_labels(header, cmask, cbits)
            return blocks

        if self.prml and self.tap is not None:
            xt = self.struct_truth(header)
            x1 = T.prml_tiles(y, self.tap, self.bias, self.known, xt, x0,
                              sweeps=self.sweeps)
            b1 = x1[pc[:, 0], pc[:, 1]]
            blocks1, cm1, cb1 = self.certify(b1, bc)
            if len(blocks1) > best[0]:
                best = (len(blocks1), blocks1, cm1, cb1, x1)

        n, blocks, cmask, cbits, x = best

        # ---- CODE-VALIDATED PER-CODEWORD GEOMETRY SEARCH
        #
        # One homography plus one radial coefficient is a single camera pose
        # and a single lens model for the whole frame. Neither holds: measured
        # on IMG_7870, giving each row band its own sub-cell VERTICAL offset
        # (dx drift is nil, 0.0001 cells/band) lifts codewords-inside-budget
        # from 45.3% to 57.0%, and letting the radial CENTRE move lifts it from
        # 32.5% to 45.0% with the optimum landing outside the code entirely.
        # There is a row-dependent residual of up to +-0.3 cells that no
        # frame-global geometry can express.
        #
        # A codeword is a contiguous band of cells, so its geometry error is
        # essentially one number - and RS+CRC32 is a free, exact accept test
        # for it. So the receiver can SEARCH geometry per codeword and keep any
        # hit: a codeword that certifies is correct by construction, whatever
        # offset produced it, so no candidate can do harm and no ground truth
        # is needed. The same structural argument that makes certified labels
        # safe for the equalizer makes them safe here, applied to geometry.
        if resample is not None and n < self.n_sub:
            got = {j for j, _b in blocks}
            for (dx, dy) in self.geom_offsets:
                miss = [j for j in range(self.n_sub) if j not in got]
                if not miss:
                    break
                y2 = resample(dx, dy)
                lum2 = y2[pc[:, 0], pc[:, 1]]
                b2, c2 = grid._mono_decide(lum2, self.L, pc)
                bc2 = (c2[: nb * 8].reshape(nb, 8).min(axis=1)
                       if self.erase else None)
                blk2, cm2, cb2 = self.certify(b2, bc2, only=miss)
                for j, blk in blk2:
                    got.add(j)
                    blocks.append((j, blk))
                if blk2:
                    sl = cm2
                    cmask = cmask | sl
                    cbits = np.where(sl, cb2, cbits)
            n = len(blocks)

        # certified-interferer cancellation: last rescue stage, fires only
        # when codewords are still missing AND the previous camera frame
        # certified cells of seq-1 AND the fitted interferer amplitude is
        # material. Clean captures skip it entirely (a ~ 0).
        #
        # Rescued cells go to the fountain and to next-frame cancellation
        # labels, but NOT into the donor fit: their labels are proven, but
        # the raw y they would be fitted against still contains the
        # interference, and correct-label-vs-polluted-observation teaches a
        # wrong kernel (measured: broke strict-superset vs the off path).
        lab_mask, lab_bits = cmask, cbits
        n_donor = n                     # pre-cancel count gates the refit
        if n < self.n_sub:
            lc = self.cancel_prev(lum, header, nb)
            if lc is not None:
                got = {j for j, _b in blocks}
                miss = [j for j in range(self.n_sub) if j not in got]
                b3, c3 = grid._mono_decide(lc, self.L, pc)
                bc3 = (c3[: nb * 8].reshape(nb, 8).min(axis=1)
                       if self.erase else None)
                blk3, cm3, cb3 = self.certify(b3, bc3, only=miss)
                for j, blk in blk3:
                    blocks.append((j, blk))
                if blk3:
                    lab_mask = cmask | cm3
                    lab_bits = np.where(cm3, cb3, cbits)

        self.pending = (y, header, n_donor, cmask, cbits, x)
        if allow_refit:
            self.commit()
            self.note_labels(header, lab_mask, lab_bits)
        return blocks

    def commit(self):
        """Let the frame most recently decoded become the kernel donor, if it
        certified enough of itself to be worth learning from."""
        if not self.prml or self.pending is None:
            return False
        y, header, n, cmask, cbits, x = self.pending
        if n < self.refit:
            return False
        xt = self.struct_truth(header)
        lab = x.copy()
        lab[self.known] = xt[self.known]
        cc = self.cells[cmask]
        lab[cc[:, 0], cc[:, 1]] = cbits[cmask]
        sel = self.known.copy()
        sel[cc[:, 0], cc[:, 1]] = True
        self.tap, self.bias = TB.fit_tiles_sel(y, lab, sel, *self.tiles)
        self.donors += 1
        return True
