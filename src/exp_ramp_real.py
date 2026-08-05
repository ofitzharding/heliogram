#!/usr/bin/env python3
"""
exp_ramp_real.py — THE test: does a channel model learned at a decodable
density carry the receiver into a density that certifies nothing?

Why this take makes it answerable and no previous one did: every density in
IMG_7867 shares ONE geometry, focus, exposure and distance, because they are
blocks of a single 71-second capture. An earlier attempt transferred a PSF
between two SEPARATE takes and measured worse-than-nothing (11.8% -> 26.0%),
for the good reason that the spatially varying part of the channel is
take-specific. That confound is now gone by construction.

Chain:
  252x140 frames certify by ordinary RS  -> truth known
  -> fit PSF in CAMERA pixels (density-independent units)
  -> rescale into 350x194 cell units
  -> tile-PRML on 350x194 frames, which certify NOTHING cold
  -> count codewords recovered

Truth at 350x194 comes from ML sequence detection (§6) against templates
built from constants derived analytically, since no hard header decodes
there - that is the receiver's real situation, not a shortcut.
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
R = 2
SUB = 255 - 48 - 4


def truth_frame(L, seq, enc, n_sub, file_size, zone_w=0):
    parts = []
    for j in range(n_sub):
        b = enc.block(seq * n_sub + j)
        b = b + b"\x00" * (SUB - len(b))
        parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
    hdr = grid.pack_header(seq, enc.k,
                           L.payload_capacity_bytes(grid.MODE_MONO) - 4,
                           file_size, grid.MODE_MONO, zone_w, 0)
    im = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO, cell_px=1)
    return (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)


def main():
    grid.set_ecc(48); grid.set_header_len(28); grid.set_header_centered(True)
    data = Path(__file__).parent.parent.joinpath("demo/payload_big.png").read_bytes()

    Ls = grid.Layout(252, 140)
    Ld = grid.Layout(350, 194)
    ns = grid.sub_count(Ls, grid.MODE_MONO)
    nd = grid.sub_count(Ld, grid.MODE_MONO)
    encs = fountain.Encoder(data, SUB)
    encd = fountain.Encoder(data, SUB)
    subd = TB.SubBlock(Ld, 48, nd * (255 - 48))
    knownS = Ls.is_finder | Ls.is_sep | Ls.is_ring | Ls.is_header
    knownD = Ld.is_finder | Ld.is_sep | Ld.is_ring | Ld.is_header
    payD = ~knownD
    allS = np.argwhere(np.ones((Ls.gh, Ls.gw), bool))
    allD = np.argwhere(np.ones((Ld.gh, Ld.gw), bool))

    # make_probe stamps the HOLD into zone_w (1, 2 or 4). Templates built with
    # zone_w=0 mismatch every real header by construction, which is why the
    # first two runs of this test found zero usable frames.
    print("building ML header templates for 350x194 (one set per hold) ...")
    TPLS = {}
    for hold_v in (1, 2, 4):
        TPLS[hold_v] = grid.header_templates(
            dict(k=encd.k,
                 block_size=Ld.payload_capacity_bytes(grid.MODE_MONO) - 4,
                 file_size=len(data), mode=0, zone_w=hold_v, zone_modes=0), 400)

    cap = cv2.VideoCapture(CAP)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---------- harvest a CERTIFIED 252x140 frame (the ordinary path)
    donor = None
    for fi in np.linspace(tot * 0.05, tot * 0.95, 90).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, img = cap.read()
        if not ok:
            continue
        for k1 in (0.015, 0.020, 0.025):
            grid.set_radial(k1)
            H = grid.locate(img, Ls)
            if H is None:
                continue
            hd, _s, _t = grid.sample_frame(img, Ls, H)
            if hd is None:
                continue
            xt = truth_frame(Ls, hd["seq"], encs, ns, len(data))
            y = grid.sample_cells(img, Ls, H, allS).mean(axis=1
                 ).reshape(Ls.gh, Ls.gw).astype(np.float32)
            th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8),
                                  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ber = float((((y > th).astype(np.float32))[~knownS] != xt[~knownS]).mean())
            c = np.array([[0, 0], [Ls.gw, 0]], np.float32).reshape(-1, 1, 2)
            p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
            pxc = float(np.linalg.norm(p[1] - p[0]) / Ls.gw)
            if ber < 0.02:
                donor = dict(y=y, xt=xt, pxc=pxc, k1=k1, seq=hd["seq"], ber=ber)
                break
        if donor:
            break
    if donor is None:
        print("no certified 252x140 donor found"); return
    print(f"certified donor: 252x140 seq {donor['seq']}, {donor['pxc']:.1f} px/cell, "
          f"BER {100*donor['ber']:.2f}%, k1 {donor['k1']}")

    # PSF in CAMERA pixels from the donor
    best = None
    for sg in np.arange(0.10, 1.20, 0.02):
        k = int(sg * 6) | 1
        pred = cv2.GaussianBlur(donor["xt"], (k, k), sg)
        A = np.stack([pred.ravel(), np.ones(pred.size)], 1)
        co, *_ = np.linalg.lstsq(A, donor["y"].ravel(), rcond=None)
        r = float(((A @ co - donor["y"].ravel()) ** 2).mean())
        if best is None or r < best[0]:
            best = (r, sg, co)
    _, sg_s, co_s = best
    sigma_cam = sg_s * donor["pxc"]
    print(f"PSF: {sg_s:.2f} cells at 252 = {sigma_cam:.2f} CAMERA px")

    # ---------- 350x194 frames: cold (threshold) vs ramp-carried (tile-PRML)
    print(f"\n{'frame':>7s} {'seq':>5s} {'marg':>6s} {'px/cell':>8s} "
          f"{'cold BER':>9s} {'cold cw':>8s} {'PRML BER':>9s} {'PRML cw':>8s}")
    tested = 0
    tot_cold = tot_prml = 0
    for fi in np.linspace(tot * 0.05, tot * 0.95, 200).astype(int):
        if tested >= 8:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, img = cap.read()
        if not ok:
            continue
        hit = None
        for k1 in (0.015, 0.020, 0.025):
            grid.set_radial(k1)
            H = grid.locate(img, Ld)
            if H is None:
                continue
            # NOT structure agreement: that metric is biased at finder_scale>1
            # (bigger finders shift the black/white balance of the known set,
            # so a median threshold misclassifies). The clean transmit file
            # scores only 72.8% at 350x194 while its header decodes perfectly.
            # Gate on ML header margin instead, which is what actually matters.
            hl_ = grid.sample_cells(img, Ld, H, Ld.header_cells).mean(axis=1)
            for hv, TP in TPLS.items():
                _sq, mg = grid.ml_header_seq(hl_, TP)
                if mg >= 3.0:
                    hit = (H, k1, mg, hv, TP)
                    break
            if hit:
                break
        if hit is None:
            continue
        H, k1, margin, hv, TP = hit
        grid.set_radial(k1)
        hl = grid.sample_cells(img, Ld, H, Ld.header_cells).mean(axis=1)
        seq, margin = grid.ml_header_seq(hl, TP)
        c = np.array([[0, 0], [Ld.gw, 0]], np.float32).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        pxc = float(np.linalg.norm(p[1] - p[0]) / Ld.gw)
        xt = truth_frame(Ld, seq, encd, nd, len(data), zone_w=hv)
        y = grid.sample_cells(img, Ld, H, allD).mean(axis=1
             ).reshape(Ld.gh, Ld.gw).astype(np.float32)
        th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (y > th).astype(np.float32)
        cold_ber = float((x0[payD] != xt[payD]).mean())
        cold_cw = subd.try_certify(x0[subd.cells[:, 0], subd.cells[:, 1]])[2]

        # THE RAMP: rescale the donor PSF into 350x194 cell units
        sg_d = sigma_cam / pxc
        ax = np.arange(-R, R + 1, dtype=np.float32)
        g1 = np.exp(-0.5 * (ax / sg_d) ** 2)
        K = np.outer(g1, g1); K /= K.sum()
        predk = cv2.filter2D(xt * knownD, -1, K, borderType=cv2.BORDER_REPLICATE)
        A = np.stack([predk[knownD].ravel(), np.ones(int(knownD.sum()))], 1)
        co, *_ = np.linalg.lstsq(A, y[knownD].ravel(), rcond=None)
        tap = np.zeros((Ld.gh, Ld.gw, 25), np.float32)
        tap[:, :] = (K.ravel() * co[0])[None, None, :]
        bias = np.full((Ld.gh, Ld.gw), co[1], np.float32)
        xr = T.prml_tiles(y, tap, bias, knownD, xt * knownD, x0, sweeps=3)
        prml_ber = float((xr[payD] != xt[payD]).mean())
        prml_cw = subd.try_certify(xr[subd.cells[:, 0], subd.cells[:, 1]])[2]

        tot_cold += cold_cw; tot_prml += prml_cw; tested += 1
        print(f"{fi:7d} {seq:5d} {margin:6.1f} {pxc:8.1f} "
              f"{100*cold_ber:8.2f}% {cold_cw:5d}/{nd:<3d} "
              f"{100*prml_ber:8.2f}% {prml_cw:5d}/{nd:<3d}")
    cap.release()
    if tested:
        print(f"\n{tested} frames at 350x194: cold {tot_cold} codewords total, "
              f"ramp-carried {tot_prml}")
        rate = nd * SUB * 60 / 1000
        print(f"350x194 carries {rate:.0f} KB/s at 100% yield; "
              f"ramp yield {100*tot_prml/(tested*nd):.1f}% "
              f"-> {rate*tot_prml/(tested*nd):.1f} KB/s")


if __name__ == "__main__":
    main()
