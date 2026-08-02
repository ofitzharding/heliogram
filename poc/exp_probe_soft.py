#!/usr/bin/env python3
"""
exp_probe_soft.py — apply the certified-label pipeline to the density that
ALREADY WORKS, instead of only to the ones that do not.

The 110.0 KB/s headline from the probe take was produced by
analyze_probe.py: plain threshold + hard RS. None of the machinery this
project exists to test was involved. At 252x140 the density carries 194.9
KB/s at full yield and the conventional path harvested 56.4% of it, so
beating decimen's 128 KB/s needs 66% yield - a 10-point gap, not a
different order of magnitude.

This measures what tile-PRML with certified-label kernels does to that gap,
on the same frames, with the same geometry. Kernel donor and evaluated
frames are disjoint, so no frame is scored against a model fitted on it.
"""
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid, fountain
import exp_tile_prml as T
import exp_turbo_frame as TB

CAP = "/Users/oscarfitzharding/Downloads/IMG_7867.MOV"
SUB = 255 - 48 - 4


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    grid.set_ecc(48); grid.set_header_len(28); grid.set_header_centered(True)
    L = grid.Layout(252, 140)
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay = ~known
    allc = np.argwhere(np.ones((L.gh, L.gw), bool))
    data = Path(__file__).parent.parent.joinpath("demo/payload_big.png").read_bytes()
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    enc = fountain.Encoder(data, SUB)
    sub = TB.SubBlock(L, 48, n_sub * (255 - 48))

    cap = cv2.VideoCapture(CAP)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # Sample ACROSS the whole capture, not the first N hits. The camera
    # settles over the first seconds: BER fell 8.23% -> 1.51% and codewords
    # rose 0 -> 12 across frames 127-446, so collecting the first N matches
    # measures the settling transient and nothing else.
    idxs = np.linspace(tot * 0.15, tot * 0.97, n_want * 3).astype(int)
    for fi in idxs:
        if len(frames) >= n_want:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, img = cap.read()
        if not ok:
            continue
        # pick the k1 that decodes BEST, not the first that decodes at all.
        # Taking the first header-yielding value dropped measured yield from
        # 56.4% to 22.6% on identical footage.
        cand = []
        for k1 in (0.010, 0.015, 0.020, 0.025):
            grid.set_radial(k1)
            H = grid.locate(img, L)
            if H is None:
                continue
            hd, _s, _t = grid.sample_frame(img, L, H)
            if hd is None:
                continue
            yq = grid.sample_cells(img, L, H, allc).mean(axis=1
                  ).reshape(L.gh, L.gw).astype(np.float32)
            thq, _ = cv2.threshold(np.clip(yq.ravel(), 0, 255).astype(np.uint8),
                                   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            xq = (yq > thq).astype(np.float32)
            nq = sub.try_certify(xq[sub.cells[:, 0], sub.cells[:, 1]])[2]
            cand.append((nq, k1, H, hd))
        if cand:
            cand.sort(key=lambda c: -c[0])
            _nq, k1, H, hd = cand[0]
            # truth via the fountain, exactly as the certified path would
            parts = []
            for j in range(n_sub):
                b = enc.block(hd["seq"] * n_sub + j)
                b = b + b"\x00" * (SUB - len(b))
                parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
            tr = grid.render_frame(L, grid.pack_header(hd["seq"], hd["k"],
                                   hd["block_size"], hd["file_size"],
                                   grid.MODE_MONO, hd.get("zone_w", 0), 0),
                                   b"".join(parts), grid.MODE_MONO, cell_px=1)
            xt = (cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
            y = grid.sample_cells(img, L, H, allc).mean(axis=1
                 ).reshape(L.gh, L.gw).astype(np.float32)
            frames.append(dict(fi=int(fi), seq=hd["seq"],
                               hold=hd.get("zone_w", 0), y=y, xt=xt, k1=k1))
    cap.release()
    print(f"{len(frames)} frames at 252x140 with certified headers\n")
    if len(frames) < 3:
        print("not enough"); return

    # ROLLING donor, which is what the receiver actually does. A fixed donor
    # over a 71-second capture is wrong by construction: section 13 measured
    # that kernels transfer for roughly a second before geometry drifts, and
    # the fixed-donor run showed exactly that - frames near the donor hit
    # 16/16 codewords while frames 5 seconds away barely moved.
    #
    # Production rule: decode with the current kernel; whenever a frame
    # certifies enough of itself, it becomes the new donor. Bootstrap from the
    # ordinary hard-RS path, which is how the first kernel is obtained at all.
    def hard_of(F):
        th, _ = cv2.threshold(np.clip(F["y"].ravel(), 0, 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (F["y"] > th).astype(np.float32)
        return x0, float((x0[pay] != F["xt"][pay]).mean())

    frames.sort(key=lambda F: F["fi"])
    tap = bias = None
    donor_fi = None
    # A donor must be a frame the receiver could ACTUALLY reconstruct, i.e.
    # one whose codewords certify. Truth here comes from the fountain given
    # seq, which the real receiver cannot do (it lacks the source file), so
    # this threshold is what keeps the experiment honest: only frames that
    # would genuinely be recoverable are allowed to teach.
    REFIT = 14
    print(f"{'frame':>7s} {'donor':>7s} {'hardBER':>8s} {'PRML BER':>9s} "
          f"{'hard cw':>8s} {'PRML cw':>8s}")
    th_tot = pr_tot = n_ev = 0
    for F in frames:
        x0, hb = hard_of(F)
        cw0 = sub.try_certify(x0[sub.cells[:, 0], sub.cells[:, 1]])[2]
        if tap is not None:
            x = T.prml_tiles(F["y"], tap, bias, known, F["xt"], x0, sweeps=3)
            pb = float((x[pay] != F["xt"][pay]).mean())
            cw1 = sub.try_certify(x[sub.cells[:, 0], sub.cells[:, 1]])[2]
        else:
            x, pb, cw1 = x0, hb, cw0        # no kernel yet: conventional only
        th_tot += cw0; pr_tot += cw1; n_ev += 1
        print(f"{F['fi']:7d} {str(donor_fi or '-'):>7s} {100*hb:7.2f}% "
              f"{100*pb:8.2f}% {cw0:5d}/{n_sub:<3d} {cw1:5d}/{n_sub:<3d}")
        # this frame certifies well enough to become the next donor
        if max(cw0, cw1) >= REFIT:
            # BLOCKER 3: fit ONLY on cells the receiver genuinely recovered.
            # A donor certifying 15/16 knows 15/16 of its payload; using full
            # fountain-reconstructed truth would be leakage, because the real
            # receiver lacks the source file. Uncertified codewords are
            # excluded from the fit rather than assumed.
            best = x if cw1 >= cw0 else x0
            bits = best[sub.cells[:, 0], sub.cells[:, 1]]
            cmask, cbits, _n, _f = sub.try_certify(bits)
            lab = best.copy()
            cc = sub.cells[cmask]
            lab[cc[:, 0], cc[:, 1]] = cbits[cmask]
            fitsel = known.copy()
            fitsel[cc[:, 0], cc[:, 1]] = True
            tap, bias = TB.fit_tiles_sel(F["y"], lab, fitsel, 8, 14)
            donor_fi = F["fi"]

    y0 = th_tot / (n_ev * n_sub)
    y1 = pr_tot / (n_ev * n_sub)
    rate = n_sub * SUB * 60 / 1000
    print(f"\n--- {n_ev} frames, 252x140 carries {rate:.1f} KB/s at full yield ---")
    print(f"hard RS (the 110 KB/s headline path) : {100*y0:.1f}% yield -> "
          f"{rate*y0:.1f} KB/s")
    print(f"certified-label, ROLLING donor       : {100*y1:.1f}% yield -> "
          f"{rate*y1:.1f} KB/s")
    print(f"\ndecimen handheld: 128 KB/s   "
          f"({'PASSED' if rate*y1 > 128 else 'short by %.1f' % (128 - rate*y1)})")


if __name__ == "__main__":
    main()
