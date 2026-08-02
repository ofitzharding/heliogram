#!/usr/bin/env python3
"""
softdec.py — one frame's worth of the certified-label receiver.

This is the mechanism that survived every negative result in this project
(Findings §14 for why nothing self-supervised works here, §3 for why the
fountain+CRC structure makes the labels exactly correct by construction),
lifted out of poc/exp_probe_soft.py so the PRODUCTION decoder can run it
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
from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
import exp_tile_prml as T
import exp_turbo_frame as TB


class FrameDecoder:
    """Stateful across frames: holds the rolling kernel donor."""

    def __init__(self, layout, ecc, n_sub, tiles=(8, 14), sweeps=3,
                 refit=None, erase=True, prml=True):
        self.L = layout
        self.ecc = ecc
        self.rs = RSCodec(ecc)
        self.n_sub = n_sub
        self.sub_size = (255 - ecc) - 4
        self.cells = layout.payload_cells[: n_sub * 255 * 8]
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
        self._blank = np.zeros(layout.payload_capacity_bytes(grid.MODE_MONO),
                               np.uint8).tobytes()

    # ---- structure the receiver knows before decoding anything
    def struct_truth(self, header):
        raw = grid.pack_header(int(header["seq"]), int(header["k"]),
                               int(header["block_size"]),
                               int(header["file_size"]), grid.MODE_MONO, 0, 0)
        img = grid.render_frame(self.L, raw, self._blank, grid.MODE_MONO,
                                cell_px=1)
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)

    # ---- certify every codeword in one set of hard decisions
    def certify(self, bits, byteconf=None):
        by = np.packbits(bits[: self.n_sub * 255 * 8].astype(np.uint8))
        blocks = []
        cmask = np.zeros(len(self.cells), bool)
        cbits = np.zeros(len(self.cells), np.float32)
        for j in range(self.n_sub):
            lo, hi = j * 255, (j + 1) * 255
            chunk = bytes(by[lo:hi])
            dec = None
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
            coded = bytes(self.rs.encode(dec))
            cmask[lo * 8:hi * 8] = True
            cbits[lo * 8:hi * 8] = np.unpackbits(np.frombuffer(coded, np.uint8))
        return blocks, cmask, cbits

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
    def decode(self, y, header, allow_refit=True):
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

        if self.prml and self.tap is not None:
            xt = self.struct_truth(header)
            x1 = T.prml_tiles(y, self.tap, self.bias, self.known, xt, x0,
                              sweeps=self.sweeps)
            b1 = x1[pc[:, 0], pc[:, 1]]
            blocks1, cm1, cb1 = self.certify(b1, bc)
            if len(blocks1) > best[0]:
                best = (len(blocks1), blocks1, cm1, cb1, x1)

        n, blocks, cmask, cbits, x = best
        self.pending = (y, header, n, cmask, cbits, x)
        if allow_refit:
            self.commit()
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
