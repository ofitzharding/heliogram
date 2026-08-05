#!/usr/bin/env python3
"""
exp_preeq2.py — transmitter-side pre-equalisation against the MEASURED PSF.

THE ASYMMETRY
-------------
Every attempt so far to beat the resolution limit has been receiver-side:
deconvolution, DFE, tile-PRML. All of them fight the same losing battle,
because inverting a low-pass channel at the receiver multiplies the noise by
1/|H(f)| exactly where |H| is small. That is why the density wall feels hard.

Pre-equalising at the TRANSMITTER has no such penalty. The display emits
x' = H^-1 x, the camera applies H, and what lands on the sensor is x with the
noise added AFTERWARDS, unamplified. Same inverse filter, opposite noise
consequence.

WHAT IT COSTS
-------------
Display headroom. H^-1 overshoots, and the panel clips at 0 and 255. So the
base levels must be pulled inward (say 64/192) to leave room for overshoot,
which costs contrast, i.e. SNR. Pre-emphasis strength therefore has an
optimum: too little and cells stay blurred, too much and contrast collapses.
That optimum is what this measures.

Crucially the overshoot levels carry NO extra bits. They are pre-compensation,
not a constellation, so this does not run into the four-level barrier that
killed gray4 and the temporal-multiplexing idea: the receiver still makes one
binary decision per cell.

The PSF used here is measured off a real capture, not assumed.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid


def measure_psf(capture_png, gw, gh, centered, radial, payload, ecc=48):
    """Fit the camera PSF sigma in CAMERA pixels off a real capture."""
    import struct, zlib
    from codec import fountain
    grid.set_ecc(ecc); grid.set_header_len(28)
    grid.set_header_centered(centered); grid.set_radial(radial)
    L = grid.Layout(gw, gh)
    img = cv2.imread(capture_png)
    H = grid.locate(img, L)
    hd, _s, _t = grid.sample_frame(img, L, H)
    if hd is None:
        return None
    data = Path(payload).read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)
    blk = enc.block(hd["seq"]); blk = blk + b"\x00" * (bs - len(blk))
    p = struct.pack("<I", zlib.crc32(blk) & 0xFFFFFFFF) + blk
    t = grid.render_frame(L, grid.pack_header(hd["seq"], hd["k"], hd["block_size"],
                          hd["file_size"], 0, 0, 0), p, grid.MODE_MONO, cell_px=1)
    x = (cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
    allc = np.argwhere(np.ones((L.gh, L.gw), bool))
    y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(L.gh, L.gw)
    c = np.array([[0, 0], [L.gw, 0]], np.float32).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    pxc = float(np.linalg.norm(q[1] - q[0]) / L.gw)
    best = None
    for sg in np.arange(0.10, 1.60, 0.02):
        k = int(sg * 6) | 1
        pred = cv2.GaussianBlur(x, (k, k), sg)
        A = np.stack([pred.ravel(), np.ones(pred.size)], 1)
        co, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
        r = float(((A @ co - y.ravel()) ** 2).mean())
        if best is None or r < best[0]:
            best = (r, sg, co)
    res, sg, co = best
    resid = np.sqrt(res)
    return dict(sigma_cells=sg, px_per_cell=pxc, sigma_px=sg * pxc,
                gain=co[0], bias=co[1], noise=resid)


def trial(px_per_cell, sigma_px, noise, pre, headroom, n=220, seed=0):
    """One synthetic frame at a given cell size, with and without pre-eq.

    Everything is done at CAMERA-pixel resolution so the PSF stays fixed in
    physical units while the cells shrink, which is the actual physics.
    """
    rs = np.random.RandomState(seed)
    bits = rs.randint(0, 2, (n, n)).astype(np.float32)
    up = int(round(px_per_cell))
    img = np.kron(bits, np.ones((up, up), np.float32))

    lo, hi = (128 - 127 * headroom), (128 + 127 * headroom)
    base = lo + img * (hi - lo)
    if pre > 0:
        k = int(sigma_px * 6) | 1
        blur = cv2.GaussianBlur(base, (k, k), sigma_px)
        base = base + pre * (base - blur)          # unsharp = partial H^-1
    tx = np.clip(base, 0, 255)

    k = int(sigma_px * 6) | 1
    rx = cv2.GaussianBlur(tx, (k, k), sigma_px)
    rx = rx + rs.normal(0, noise, rx.shape)

    c = up // 2
    samp = rx[c::up, c::up][:n, :n]
    th = np.median(samp)
    return float(((samp > th).astype(np.float32) != bits).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=None)
    args = ap.parse_args()

    S = ("/private/tmp/claude-502/-Users-oscarfitzharding-Documents-claude-obsidian/"
         "70e6cc8f-bc18-4a90-9409-a0b49bb559f9/scratchpad/")
    psf = measure_psf(args.capture or S + "frames252/s051.png", 252, 140, False,
                      0.020, str(Path(__file__).parent.parent / "demo" / "payload.png"))
    if psf is None:
        print("could not measure PSF"); return
    print("PSF measured off a REAL capture (252x140, the take that decoded):")
    print(f"   {psf['px_per_cell']:.1f} camera px/cell, sigma {psf['sigma_cells']:.2f} cells "
          f"= {psf['sigma_px']:.2f} CAMERA px")
    print(f"   contrast gain {psf['gain']:.0f}, residual noise {psf['noise']:.1f} counts\n")

    sigma_px = psf["sigma_px"]
    noise = max(psf["noise"], 2.0)
    print("BER vs cell size, no pre-eq  ->  best pre-eq   (RS budget ~1.2%)")
    print(f"{'px/cell':>8s} {'plain':>9s} {'pre-eq':>9s} {'strength':>9s} {'headroom':>9s}")
    for pxc in (14, 12, 10, 9, 8, 7, 6, 5):
        plain = trial(pxc, sigma_px, noise, 0.0, 1.0)
        best = (1e9, 0, 0)
        for pre in (0.4, 0.8, 1.2, 1.6, 2.2, 3.0):
            for hr in (1.0, 0.8, 0.65, 0.5):
                b = trial(pxc, sigma_px, noise, pre, hr)
                if b < best[0]:
                    best = (b, pre, hr)
        b, pre, hr = best
        flag = "  <- decodable" if b < 0.012 <= plain else ""
        print(f"{pxc:8d} {100*plain:8.2f}% {100*b:8.2f}% {pre:9.1f} {hr:9.2f}{flag}")


if __name__ == "__main__":
    main()
