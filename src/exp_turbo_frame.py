#!/usr/bin/env python3
"""
exp_turbo_frame.py — INTRA-frame turbo equalization with RS-certified labels.

The chain of measurements that led here (all on real captures, 2026-08-01):
  - tile-PRML with kernels fitted on TRUTH rescues frames up to ~10-14%
    threshold BER (own-truth: 6.3% -> 0.84%, 9.9% -> 1.01%). The physics
    works; only the kernel is missing at decode time.
  - cross-frame kernels do NOT transfer to blurred frames (0/29 rescued):
    each blurred frame has its own smear kernel.
  - plain decision-directed fitting is ~neutral, and confidence-trimmed DD
    actively DIVERGES: selecting labels by margin biases the neighbor-pattern
    distribution the LS fit sees (high-margin cells sit in same-colored
    neighborhoods).

So the label source must be unbiased and per-frame. The code layer provides
one: individual RS codewords that decode in the frame's good regions are
certified true (48 parity bytes; miscorrection odds are negligible at this
scale). A certified codeword labels a contiguous band of ~2000 cells with
its exact transmitted bits — no selection on margin, no cross-frame
transfer. Fit tile kernels on those bands, run tile-PRML with certified
cells pinned as anchors, and try the remaining codewords again. Iterate.

This is the fountain-bootstrap mechanism (Findings §3) pushed inside a
single frame at codeword granularity.
"""
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
import exp_tile_prml as T

R = 2


def codeword_spans(n_msg: int, ecc: int):
    """[(msg_lo, msg_hi, coded_lo, coded_hi)] byte ranges per RS codeword."""
    data_per = 255 - ecc
    spans = []
    m = c = 0
    while m < n_msg:
        d = min(data_per, n_msg - m)
        spans.append((m, m + d, c, c + d + ecc))
        m += d
        c += d + ecc
    return spans


class SubBlock:
    """Per-codeword certify/label machinery for one frame's payload."""

    def __init__(self, layout, ecc, n_msg):
        self.L = layout
        self.ecc = ecc
        self.rs = RSCodec(ecc)
        self.spans = codeword_spans(n_msg, ecc)
        self.n_coded = self.spans[-1][3]
        self.cells = layout.payload_cells[: self.n_coded * 8]

    def try_certify(self, bits):
        """bits: hard decisions over payload cells (row-major payload order).
        Returns (cert_mask over cells, cert_bits, n_ok, ok_flags)."""
        by = np.packbits(bits[: self.n_coded * 8].astype(np.uint8))
        cert_mask = np.zeros(len(self.cells), bool)
        cert_bits = np.zeros(len(self.cells), np.float32)
        ok_flags = []
        for (mlo, mhi, clo, chi) in self.spans:
            chunk = bytes(by[clo:chi])
            try:
                dec = bytes(self.rs.decode(chunk)[0])
                coded = bytes(self.rs.encode(dec))
                cb = np.unpackbits(np.frombuffer(coded, np.uint8))
                cert_mask[clo * 8:chi * 8] = True
                cert_bits[clo * 8:chi * 8] = cb
                ok_flags.append(True)
            except ReedSolomonError:
                ok_flags.append(False)
        return cert_mask, cert_bits, sum(ok_flags), ok_flags


def turbo_frame(y, x0, layout, sub, struct_truth, known, tiles=(8, 14),
                rounds=4, sweeps=2, verbose=True):
    """Intra-frame turbo: certify -> fit on certified bands -> PRML -> repeat.

    The detector never sees ground truth. Labels come from (a) grid structure
    the receiver knows a priori, (b) RS-certified codewords.
    """
    gh, gw = y.shape
    cells = sub.cells
    x = x0.copy()
    x[known] = struct_truth[known]
    history = []

    for rd in range(rounds):
        bits = x[cells[:, 0], cells[:, 1]]
        cert_mask, cert_bits, n_ok, _ = sub.try_certify(bits)
        history.append(n_ok)
        if verbose:
            print(f"    round {rd}: {n_ok}/{len(sub.spans)} codewords certified")
        if n_ok == len(sub.spans):
            break

        # label image: structure + certified codeword cells
        lab = x.copy()
        lab[known] = struct_truth[known]
        sel = known.copy()
        cc = cells[cert_mask]
        lab[cc[:, 0], cc[:, 1]] = cert_bits[cert_mask]
        sel[cc[:, 0], cc[:, 1]] = True

        # uncertified cells keep current decisions as fit fallback: tiles with
        # too few certified cells fit on everything they have (attenuation
        # from label noise beats a singular system).
        tap, bias = fit_tiles_sel(y, lab, sel, *tiles)

        # certified cells are pinned in the Viterbi exactly like structure
        kn2 = sel
        truth2 = lab
        x = T.prml_tiles(y, tap, bias, kn2, truth2, x, sweeps=sweeps)
    bits = x[cells[:, 0], cells[:, 1]]
    _, _, n_final, ok_flags = sub.try_certify(bits)
    history.append(n_final)
    return x, history, ok_flags


def fit_tiles_sel(y, x, sel, tiles_r, tiles_c, reg=1e-3):
    """Per-tile LS fit using only `sel` cells as equations; the regressor
    columns still use ALL current decisions (neighbors of a labeled cell may
    be unlabeled — their current estimate stands in)."""
    gh, gw = x.shape
    cols = []
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            cols.append(np.roll(np.roll(x, -dr, 0), -dc, 1).ravel())
    cols.append(np.ones(x.size, np.float32))
    A = np.stack(cols, 1)
    tap = np.zeros((gh, gw, 25), np.float32)
    bias = np.zeros((gh, gw), np.float32)
    interior = np.zeros((gh, gw), bool)
    interior[R:gh - R, R:gw - R] = True
    for i in range(tiles_r):
        for j in range(tiles_c):
            r0, r1 = i * gh // tiles_r, (i + 1) * gh // tiles_r
            c0, c1 = j * gw // tiles_c, (j + 1) * gw // tiles_c
            m = np.zeros((gh, gw), bool)
            m[r0:r1, c0:c1] = True
            s = (m & interior & sel).ravel()
            if s.sum() < 80:                      # not enough certified here
                s = (m & interior).ravel()        # fall back to all decisions
            At, bt = A[s], y.ravel()[s]
            ATA = At.T @ At + reg * np.eye(26) * At.shape[0]
            k = np.linalg.solve(ATA, At.T @ bt)
            tap[r0:r1, c0:c1] = k[:-1]
            bias[r0:r1, c0:c1] = k[-1]
    return tap, bias
