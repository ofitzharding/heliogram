#!/usr/bin/env python3
"""
exp_rolling_poc.py — the rolling-shutter harvest, proven end to end in silico.

SYNTHESIZER: camera frames at 60 fps from a 120 fps transmit, composed the
way the sensor actually sees them (measured on IMG_7908, Findings §27):
rows above the seam carry code frame 2f+1, rows below carry 2f (display
scan and shutter readout both top-to-bottom), the seam advances by a fixed
per-frame drift (clock beat), and a band of rows around the seam is a 50/50
blend of both frames (exposure straddle: erasures, not errors).

DECODER: locate + header as ever; the header band pins seq for its side of
the seam; the other side is seq+-1 by scan direction. Codewords are
contiguous row bands, so each certifies under its side's seq. Certification
stays the arbiter: a block only enters the fountain after RS+CRC.

Scope note: the seam position here comes from the tracker's PREDICTION
(r0 + f*dr), which the synthesizer also uses - this proves the harvest
mechanics, band assignment, and fountain closure. On a real capture r0/dr
are fitted from the photometric seam (analyze_rolling measures it today);
that estimator is the one open piece and is architected in the Attack Plan.

Success = sha256 bit-exact AND effective symbol rate clearly above the
60 fps single-seq baseline on the same synthetic frames.
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from softdec import FrameDecoder

CELL_PX = 5
MARGIN = 40


def load_webtx(d):
    meta = json.load(open(d / "meta.json"))
    bits = np.fromfile(d / "frames.bin", np.uint8)
    bpf = meta["bytes_per_frame"]
    gw, gh = meta["gw"], meta["gh"]

    def frame(seq):
        seq %= meta["frames"]
        return np.unpackbits(
            bits[seq * bpf:(seq + 1) * bpf])[:gw * gh].reshape(gh, gw)
    return meta, frame


def synth_camera_frame(cells_top, cells_bot, seam_row, mix_rows):
    """Compose one camera frame in cell space, then render to pixels."""
    gh, gw = cells_top.shape
    comp = cells_bot.astype(np.float32).copy()
    comp[:seam_row] = cells_top[:seam_row]
    lo = max(0, seam_row - mix_rows // 2)
    hi = min(gh, seam_row + (mix_rows + 1) // 2)
    if hi > lo:
        comp[lo:hi] = 0.5 * (cells_top[lo:hi] + cells_bot[lo:hi])
    img = cv2.resize((comp * 255).astype(np.uint8), (gw * CELL_PX,
                                                     gh * CELL_PX),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((gh * CELL_PX + 2 * MARGIN,
                       gw * CELL_PX + 2 * MARGIN), np.uint8)
    canvas[MARGIN:-MARGIN, MARGIN:-MARGIN] = img
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def main():
    d = Path(__file__).parent.parent / "demo" / "webtx"
    meta, frame = load_webtx(d)
    gw, gh = meta["gw"], meta["gh"]
    grid.set_ecc(meta["ecc"]); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(0.0)
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = meta["sub"]
    payload = Path(__file__).parent.parent / "demo" / meta["name"]
    want = hashlib.sha256(payload.read_bytes()).hexdigest()

    # codeword j -> grid row range (codewords are contiguous in payload order)
    cells = L.payload_cells[: n_sub * 255 * 8]
    cw_rows = []
    for j in range(n_sub):
        rs = cells[j * 255 * 8:(j + 1) * 255 * 8, 0]
        cw_rows.append((int(rs.min()), int(rs.max())))
    hdr_rows = (int(L.header_cells[:, 0].min()),
                int(L.header_cells[:, 0].max()))

    r0, dr, mix = 30.0, 11.7, 12       # seam start, drift/frame, mixed band
    N_CAM = 520                        # 1040 code frames; each code frame
                                       # contributes only its captured side
    fd = FrameDecoder(L, meta["ecc"], n_sub, erase=True, prml=False)
    allc = np.argwhere(np.ones((gh, gw), bool))
    pc = L.payload_cells
    nb = n_sub * 255

    got_roll, got_base = {}, {}
    for f in range(N_CAM):
        seam = int(r0 + f * dr) % gh
        s_bot, s_top = 2 * f, 2 * f + 1
        img = synth_camera_frame(frame(s_top), frame(s_bot), seam, mix)
        H = grid.locate(img, L)
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None:
            continue
        s_hdr = int(hd["seq"])
        # which side does the header band sit on this frame?
        hdr_on_top = hdr_rows[1] < seam
        exp_top, exp_bot = (s_hdr, s_hdr - 1) if hdr_on_top else \
                           (s_hdr + 1, s_hdr)
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        lum = y[pc[:, 0], pc[:, 1]]
        bits1d, conf = grid._mono_decide(lum, L, pc)
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)
        blocks, _m, _b = fd.certify(bits1d, bc)
        for j, blk in blocks:
            # 60fps baseline: single-seq assumption assigns EVERY certified
            # block to the header's seq - blocks from the other side of the
            # seam land on wrong indices and poison that fountain
            got_base.setdefault(s_hdr * n_sub + j, blk)
            lo_r, hi_r = cw_rows[j]
            if hi_r < seam - mix // 2:
                seq_j = exp_top
            elif lo_r >= seam + mix // 2:
                seq_j = exp_bot
            else:
                continue                     # seam band: erasure
            got_roll.setdefault(seq_j * n_sub + j, blk)

    def closes(pool):
        dec = fountain.Decoder(-(-meta["file_size"] // SUB), SUB,
                               meta["file_size"])
        for idx in sorted(pool):
            if idx in dec.seen:
                continue
            dec.add(idx, pool[idx])
            if len(dec.seen) >= dec.k and not dec.done:
                dec.gaussian_fallback()
            if dec.done:
                break
        if not dec.done:
            dec.gaussian_fallback()
        if not dec.done:
            return None
        data = dec.result()
        return hashlib.sha256(data).hexdigest()

    print(f"rolling harvest: {len(got_roll)} symbols from {N_CAM} camera "
          f"frames = {len(got_roll)/N_CAM:.1f}/frame "
          f"(60fps single-seq baseline: {len(got_base)/N_CAM:.1f}/frame)")
    rate = len(got_roll) / N_CAM * SUB * 60 / 1024
    base = len(got_base) / N_CAM * SUB * 60 / 1024
    print(f"effective rate at 4K60: rolling {rate:.1f} KB/s  "
          f"baseline {base:.1f} KB/s  ({rate/max(base,1e-9):.2f}x)")
    print(f"distinct symbols: rolling {len(got_roll)}  "
          f"baseline {len(got_base)}  (k={-(-meta['file_size']//SUB)})")
    hb = closes(got_base)
    print(f"baseline single-seq: closes={hb is not None} "
          f"bit-exact={hb == want}  (wrong-side blocks poison its indices)")
    h = closes(got_roll)
    print(f"rolling harvest:     closes={h is not None} bit-exact={h == want}")
    if h == want:
        print("ROLLING HARVEST PROVEN: 120fps transmit through a 60fps "
              "camera, bit-exact, both seam sides harvested")


if __name__ == "__main__":
    main()
