#!/usr/bin/env python3
"""
exp_soft_real.py — the decisive test of §19 on REAL footage, no new filming.

CHAIN UNDER TEST
----------------
certified-label kernels (from a neighbour frame, as the fountain layer would
supply) -> tile-PRML detection at 8.2 px/cell -> per-cell SOFT output ->
LDPC decoding at rates the hard-RS path cannot touch.

The channel is not simulated: per-cell soft outputs come from a real capture
(take466 f35), and the LDPC replay maps an arbitrary codeword onto those real
cells by flipping LLR signs where the codeword disagrees with what was truly
displayed. Valid under channel symmetry (mono 0/1 is symmetric to first
order); stated as an assumption, not hidden.

Outputs:
  1. hard BER of the detector (sanity: ~2-3% expected, matching §13)
  2. empirical mutual information per cell — the direct measurement of the
     bits/cell claim in §17/§19
  3. LDPC block success rate vs code rate on the real soft outputs
  4. the resulting information rate in KB/s
"""
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid, fountain
from codec.ldpc import LDPC
import exp_tile_prml as T

S = ("/private/tmp/claude-502/-Users-oscarfitzharding-Documents-claude-obsidian/"
     "70e6cc8f-bc18-4a90-9409-a0b49bb559f9/scratchpad/")
R = 2


def load466(name, L, enc, bs):
    img = cv2.imread(S + "frames466/" + name)
    H = grid.locate(img, L)
    if H is None:
        return None
    hd, _s, _t = grid.sample_frame(img, L, H)
    if hd is None:
        return None
    blk = enc.block(hd["seq"]); blk = blk + b"\x00" * (bs - len(blk))
    p = struct.pack("<I", zlib.crc32(blk) & 0xFFFFFFFF) + blk
    tr = grid.render_frame(L, grid.pack_header(hd["seq"], hd["k"],
                           hd["block_size"], hd["file_size"], 0, 0, 0),
                           p, grid.MODE_MONO, cell_px=1)
    xt = (cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
    allc = np.argwhere(np.ones((L.gh, L.gw), bool))
    y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(L.gh, L.gw)
    return dict(y=y.astype(np.float32), xt=xt)


def soft_llr(y, x, tap, bias, known):
    """Per-cell LLR from the fitted channel model.

    E(b) is the sum of squared residuals over the neighbourhood a cell
    touches, with that cell forced to b and all others at their decisions.
    Flipping cell i changes prediction j by d*t_ij, so
       E(flip) - E(cur) = sum_j t_ij^2 - 2 d sum_j r_j t_ij
    with d=+1 for 0->1. LLR = (E(1)-E(0)) / (2 sigma^2); positive means 0.
    Verified against brute-force recomputation below.
    """
    gh, gw = y.shape
    pred = bias + T.conv_varying(x, tap)
    r = y - pred
    # corr_i = sum over offsets of tap_i[off] * r[i+off]  (slow-varying taps)
    rp = np.pad(r, R, mode="edge")
    corr = np.zeros_like(r)
    T2 = np.zeros_like(r)
    idx = 0
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            w = tap[:, :, idx]
            corr += w * rp[R + dr:R + dr + gh, R + dc:R + dc + gw]
            T2 += w * w
            idx += 1
    d = 1.0 - 2.0 * x                    # +1 if x=0 (flip adds), -1 if x=1
    dE_flip = T2 - 2.0 * d * corr        # E(flip) - E(current)
    E1_minus_E0 = np.where(x > 0.5, -dE_flip, dE_flip)
    sigma2 = max(np.var(r[~known]), 1e-3)
    return E1_minus_E0 / (2.0 * sigma2), r


def main():
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(False); grid.set_radial(0.020)
    L = grid.Layout(466, 259)
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay = ~known
    data = Path(__file__).parent.parent.joinpath("demo/payload_big.png").read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)

    donor = load466("n35.5.png", L, enc, bs)     # certified-label source
    target = load466("f35.png", L, enc, bs)      # the frame under test
    if donor is None or target is None:
        print("missing frames"); return

    # kernels from the DONOR's certified truth (production path)
    tap, bias = T.fit_tile_kernels(donor["y"], donor["xt"], 16, 28)
    th, _ = cv2.threshold(np.clip(target["y"].ravel(), 0, 255).astype(np.uint8),
                          0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x0 = (target["y"] > th).astype(np.float32)
    # known cells (structure a priori; header after its RS decode, which this
    # frame passed) are pinned to their TRUE values - that is receiver
    # knowledge, not truth leakage. Payload cells stay free.
    x = T.prml_tiles(target["y"], tap, bias, known, target["xt"], x0,
                     sweeps=3)
    ber = (x[pay] != target["xt"][pay]).mean()
    print(f"detector hard BER at 8.2 px/cell, cross-frame kernels: "
          f"{100*ber:.2f}%   (§13 measured 2.9%)")

    llr, r = soft_llr(target["y"], x, tap, bias, known)

    # numerical check of the LLR delta formula on 40 random payload cells
    rs = np.random.RandomState(0)
    cells = np.argwhere(pay)
    errs = []
    pred = bias + T.conv_varying(x, tap)
    E_cur = float(((target["y"] - pred)[pay] ** 2).sum())
    for (ri, ci) in cells[rs.choice(len(cells), 40, replace=False)]:
        xf = x.copy(); xf[ri, ci] = 1 - xf[ri, ci]
        pf = bias + T.conv_varying(xf, tap)
        dE_true = float(((target["y"] - pf)[pay] ** 2).sum()) - E_cur
        d = 1.0 - 2.0 * x[ri, ci]
        rp = np.pad(target["y"] - pred, R, mode="edge")
        corr = T2 = 0.0
        idx = 0
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                w = tap[ri, ci, idx]
                corr += w * rp[R + dr + ri, R + dc + ci]
                T2 += w * w
                idx += 1
        dE_form = T2 - 2 * d * corr
        errs.append(abs(dE_true - dE_form) / max(abs(dE_true), 1e-6))
    print(f"LLR delta formula vs brute force: median rel err "
          f"{100*np.median(errs):.2f}%  (edge cells differ, interior must not)")

    # sign convention: llr>0 means bit=0. flip so positive supports the truth,
    # then measure calibration and information
    v = llr[pay]
    t = target["xt"][pay]
    correct = ((v > 0) == (t < 0.5))
    print(f"soft sign agrees with truth: {100*correct.mean():.2f}% "
          f"(1 - hard BER cross-check)")

    a = np.abs(v)
    qs = np.quantile(a, np.linspace(0, 1, 9))
    print("\ncalibration by |LLR| octile   (p_err, bits of MI per cell)")
    mi_total, w_total = 0.0, 0.0
    def Hb(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return -p*np.log2(p) - (1-p)*np.log2(1-p)
    for i in range(8):
        m = (a >= qs[i]) & (a <= qs[i + 1] + 1e-9)
        if m.sum() == 0:
            continue
        pe = 1 - correct[m].mean()
        mi = 1 - Hb(pe)
        mi_total += mi * m.sum(); w_total += m.sum()
        print(f"   octile {i}: p_err {100*pe:6.2f}%   MI {mi:.3f}")
    mi_cell = mi_total / w_total
    ncells = int(pay.sum())
    print(f"\nEMPIRICAL MUTUAL INFORMATION: {mi_cell:.3f} bits/cell over "
          f"{ncells} cells = {mi_cell*ncells/8/1000:.1f} KB per frame "
          f"= {mi_cell*ncells*60/8/1000:.0f} KB/s at 60fps (before yield)")

    # ------- LDPC replay on the REAL soft outputs
    print("\nLDPC replay on real per-cell soft outputs "
          "(codeword mapped onto real cells; symmetric-channel assumption):")
    sgn = np.where(t < 0.5, 1.0, -1.0)        # +llr supports displayed bit
    chan = v * sgn                            # >0 = channel supports the bit
    rs = np.random.RandomState(3)
    order = rs.permutation(len(chan))         # interleave against spatial bursts
    for wr, tag in ((15, "rate 0.80"), (12, "rate 0.75"), (10, "rate 0.70"),
                    (8, "rate 0.625")):
        code = LDPC(n=6000, wr=wr)
        nb = len(chan) // code.n
        ok = 0
        biterr = 0
        for b in range(nb):
            cw = code.encode(rs.randint(0, 2, code.k).astype(np.uint8))
            seg = chan[order[b * code.n:(b + 1) * code.n]]
            llr_b = np.where(cw == 0, seg, -seg)
            dec, s_ok = code.decode(llr_b, iters=60)
            good = s_ok and (dec == cw).all()
            ok += int(good)
            if not good:
                biterr += int((dec != cw).sum())
        eff = code.rate * ok / nb
        kbs = eff * ncells * 60 / 8 / 1000
        print(f"   (3,{wr:2d}) {tag}: {ok:2d}/{nb} blocks clean "
              f"-> {kbs:6.1f} KB/s at 60fps before yield"
              + ("" if ok == nb else f"   (residual {biterr} bit errs)"))


if __name__ == "__main__":
    main()


def calibrated():
    """Production-honest LLR calibration: kernels from certified frame A,
    |LLR|->p_err lookup from certified frame B, evaluation on frame C.
    Motivated by the octile-7 inversion in the raw run: the most confident
    cells carried 3.06% error, and confidently-wrong LLRs are the worst
    input min-sum can receive."""
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(False); grid.set_radial(0.020)
    L = grid.Layout(466, 259)
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay = ~known
    data = Path(__file__).parent.parent.joinpath("demo/payload_big.png").read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)

    A = load466("n35.5.png", L, enc, bs)   # kernel donor
    B = load466("n34.9.png", L, enc, bs)   # calibration donor
    C = load466("f35.png", L, enc, bs)     # frame under test
    tap, bias = T.fit_tile_kernels(A["y"], A["xt"], 16, 28)

    def llr_of(F):
        th, _ = cv2.threshold(np.clip(F["y"].ravel(), 0, 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (F["y"] > th).astype(np.float32)
        x = T.prml_tiles(F["y"], tap, bias, known, F["xt"], x0, sweeps=3)
        llr, _ = soft_llr(F["y"], x, tap, bias, known)
        return llr[pay], F["xt"][pay]

    vB, tB = llr_of(B)
    vC, tC = llr_of(C)

    # calibration on B: 24 quantile bins of |llr| -> log((1-pe)/pe), clipped
    nb = 24
    qs = np.quantile(np.abs(vB), np.linspace(0, 1, nb + 1))
    okB = ((vB > 0) == (tB < 0.5))
    lut = np.zeros(nb)
    for i in range(nb):
        m = (np.abs(vB) >= qs[i]) & (np.abs(vB) <= qs[i + 1] + 1e-9)
        pe = min(max(1 - okB[m].mean(), 1e-4), 0.5) if m.sum() else 0.3
        lut[i] = np.log((1 - pe) / pe)
    binC = np.clip(np.searchsorted(qs[1:-1], np.abs(vC)), 0, nb - 1)
    vC_cal = np.sign(vC) * lut[binC]

    sgn = np.where(tC < 0.5, 1.0, -1.0)
    chan = vC_cal * sgn
    rs = np.random.RandomState(3)
    order = rs.permutation(len(chan))
    ncells = int(pay.sum())
    print("CALIBRATED replay (kernels frame A, calibration frame B, test C):")
    for wr, tag in ((12, "rate 0.75"), (10, "rate 0.70"), (8, "rate 0.625")):
        code = LDPC(n=6000, wr=wr)
        nblk = len(chan) // code.n
        ok = 0
        for b in range(nblk):
            cw = code.encode(rs.randint(0, 2, code.k).astype(np.uint8))
            seg = chan[order[b * code.n:(b + 1) * code.n]]
            dec, s_ok = code.decode(np.where(cw == 0, seg, -seg), iters=60)
            ok += int(s_ok and (dec == cw).all())
        kbs = code.rate * ok / nblk * ncells * 60 / 8 / 1000
        print(f"   (3,{wr:2d}) {tag}: {ok:2d}/{nblk} clean -> {kbs:6.1f} KB/s "
              f"before yield")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cal":
    calibrated()
