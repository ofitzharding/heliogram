#!/usr/bin/env python3
"""
profile_decode.py — where the decoder's time actually goes, per stage.

The decoder runs at 4-9 frames/s on 4K with 8 workers, against a 60 fps
real-time target. Before optimising anything, measure: the stages differ by
orders of magnitude and the intuition about which dominates is usually wrong.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
import exp_tile_prml as T
from softdec import FrameDecoder


class Timer:
    def __init__(self):
        self.t = {}
        self.n = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt
        self.n[k] = self.n.get(k, 0) + 1

    def report(self, frames):
        print(f"\n{'stage':>26s} {'ms/frame':>10s} {'% of total':>11s} "
              f"{'calls':>7s}")
        tot = sum(self.t.values())
        for k in sorted(self.t, key=lambda x: -self.t[x]):
            print(f"{k:>26s} {1000*self.t[k]/frames:10.1f} "
                  f"{100*self.t[k]/tot:10.1f}% {self.n[k]:7d}")
        print(f"{'TOTAL':>26s} {1000*tot/frames:10.1f} {100.0:10.1f}%")
        print(f"\n{frames/tot:.1f} frames/s single-threaded, "
              f"{6*frames/tot:.1f} frames/s on 6 performance cores")
        print(f"real time needs 60 fps -> "
              f"{60/(6*frames/tot):.1f}x more speed required")


def main():
    cap_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "Downloads/IMG_7872.MOV")
    gw, gh = 252, 163
    nframes = 12
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(0.010)
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    allc = np.argwhere(np.ones((gh, gw), bool))
    pc = L.payload_cells
    fd = FrameDecoder(L, 48, n_sub, erase=True, prml=True)
    tm = Timer()

    cap = cv2.VideoCapture(cap_path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(tot * 0.4))
    done = 0
    while done < nframes:
        t0 = time.perf_counter()
        ok, img = cap.read()
        tm.add("video read (H.264 4K)", time.perf_counter() - t0)
        if not ok:
            break

        t0 = time.perf_counter()
        small = cv2.resize(img, None, fx=0.5, fy=0.5)
        Hs = grid.locate(small, L)
        H = (np.diag([2.0, 2.0, 1.0]) @ Hs) if Hs is not None else None
        tm.add("locate (finders)", time.perf_counter() - t0)
        if H is None:
            continue

        t0 = time.perf_counter()
        hd, _s, _t = grid.sample_frame(img, L, H)
        tm.add("sample_frame + header RS", time.perf_counter() - t0)
        if hd is None:
            continue

        t0 = time.perf_counter()
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        tm.add("sample_cells (all)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        lum = y[pc[:, 0], pc[:, 1]]
        bits, conf = grid._mono_decide(lum, L, pc)
        tm.add("demod (local threshold)", time.perf_counter() - t0)

        nb = n_sub * 255
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)

        t0 = time.perf_counter()
        fd.erase = False
        blocks_h, _m, _b = fd.certify(bits)
        tm.add("RS certify (hard only)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        fd.erase = True
        blocks_e, _m, _b = fd.certify(bits, bc)
        tm.add("RS certify (+ erasures)", time.perf_counter() - t0)

        if fd.tap is not None:
            t0 = time.perf_counter()
            xt = fd.struct_truth(hd)
            x0 = np.zeros(y.shape, np.float32)
            x0[pc[:, 0], pc[:, 1]] = bits
            T.prml_tiles(y, fd.tap, fd.bias, fd.known, xt, x0, sweeps=3)
            tm.add("tile-PRML (3 sweeps)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        fd.decode(y, hd)
        tm.add("full decode (incl refit)", time.perf_counter() - t0)
        done += 1
        print(f"  frame {done}/{nframes}: hard {len(blocks_h)}/{n_sub}, "
              f"erase {len(blocks_e)}/{n_sub}", end="\r")
    cap.release()
    print()
    tm.report(done)


if __name__ == "__main__":
    main()
