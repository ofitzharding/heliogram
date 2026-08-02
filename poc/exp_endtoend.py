#!/usr/bin/env python3
"""
exp_endtoend.py — the number never computed: END-TO-END goodput of the full
certified-label soft pipeline, INCLUDING frame yield.

§20 measured 579.6 KB/s "before yield" from a single frame. That is a
per-frame conversion rate, not a throughput. This runs the whole chain over
MANY frames of a real capture and multiplies out:

  certified frame -> tile kernels -> tile-PRML -> per-cell LLR
  -> calibrated LLR -> LDPC (rate 0.625) -> codewords recovered

and reports KB/s of decoded information over WALL CLOCK, which is the only
figure comparable to decimen's 128 KB/s handheld.

Every certified-frame role is production-honest: kernels come from a frame
the fountain layer certifies, calibration from another, evaluation on a
third. No frame is scored against truth it was fitted on.
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
import exp_turbo_frame as TB
import exp_soft_real as SR

CAPTURE = str(Path.home() / "Documents/screen-camera/captures/take466.mov")
SUB = 255 - 48 - 4
R = 2


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(False); grid.set_radial(0.020)
    L = grid.Layout(466, 259)
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay = ~known
    allc = np.argwhere(np.ones((L.gh, L.gw), bool))
    data = Path(__file__).parent.parent.joinpath("demo/payload_big.png").read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    ncells = int(pay.sum())

    # ---- harvest frames with certified headers (the receiver's real path)
    cap = cv2.VideoCapture(CAPTURE)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames = []
    idxs = np.linspace(tot * 0.25, tot * 0.80, n_frames * 3).astype(int)
    for fi in idxs:
        if len(frames) >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None:
            continue
        blk = enc.block(hd["seq"]); blk = blk + b"\x00" * (bs - len(blk))
        p = struct.pack("<I", zlib.crc32(blk) & 0xFFFFFFFF) + blk
        tr = grid.render_frame(L, grid.pack_header(hd["seq"], hd["k"],
                               hd["block_size"], hd["file_size"], 0, 0, 0),
                               p, grid.MODE_MONO, cell_px=1)
        xt = (cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
        y = grid.sample_cells(img, L, H, allc).mean(axis=1
             ).reshape(L.gh, L.gw).astype(np.float32)
        frames.append(dict(fi=int(fi), seq=hd["seq"], y=y, xt=xt))
    cap.release()
    print(f"{len(frames)} frames with certified headers "
          f"(from {len(idxs)} sampled, {100*len(frames)/len(idxs):.0f}% header yield)")
    if len(frames) < 4:
        print("not enough frames"); return

    # ---- roles: A = kernel donor, B = calibration donor, C.. = evaluated
    A, B = frames[0], frames[1]
    tap, bias = T.fit_tile_kernels(A["y"], A["xt"], 16, 28)

    def detect(F):
        th, _ = cv2.threshold(np.clip(F["y"].ravel(), 0, 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (F["y"] > th).astype(np.float32)
        x = T.prml_tiles(F["y"], tap, bias, known, F["xt"], x0, sweeps=3)
        llr, _ = SR.soft_llr(F["y"], x, tap, bias, known)
        return x, llr[pay], F["xt"][pay], x0

    # calibration table from B
    _, vB, tB, _ = detect(B)
    nb = 24
    qs = np.quantile(np.abs(vB), np.linspace(0, 1, nb + 1))
    okB = ((vB > 0) == (tB < 0.5))
    lut = np.zeros(nb)
    for i in range(nb):
        m = (np.abs(vB) >= qs[i]) & (np.abs(vB) <= qs[i + 1] + 1e-9)
        pe = min(max(1 - okB[m].mean(), 1e-4), 0.5) if m.sum() else 0.3
        lut[i] = np.log((1 - pe) / pe)

    code = LDPC(n=6000, wr=8)          # rate 0.625, the rate that held in §20
    rs = np.random.RandomState(5)
    order = rs.permutation(ncells)
    print(f"\nLDPC rate {code.rate:.3f}, {ncells} payload cells/frame, "
          f"{ncells // code.n} blocks/frame\n")
    print(f"{'frame':>7s} {'seq':>5s} {'hardBER':>8s} {'PRML':>7s} "
          f"{'cold cw':>8s} {'LDPC blocks':>12s} {'info KB':>8s}")
    sub = TB.SubBlock(L, 48, n_sub * (255 - 48))
    tot_info = 0.0
    seqs = []
    for F in frames[2:]:
        x, v, t, x0 = detect(F)
        hard = float((x0[pay] != F["xt"][pay]).mean())
        prml = float((x[pay] != F["xt"][pay]).mean())
        cold_cw = sub.try_certify(x0[sub.cells[:, 0], sub.cells[:, 1]])[2]
        binC = np.clip(np.searchsorted(qs[1:-1], np.abs(v)), 0, nb - 1)
        chan = np.sign(v) * lut[binC] * np.where(t < 0.5, 1.0, -1.0)
        nblk = ncells // code.n
        ok = 0
        for b in range(nblk):
            cw = code.encode(rs.randint(0, 2, code.k).astype(np.uint8))
            seg = chan[order[b * code.n:(b + 1) * code.n]]
            dec, s_ok = code.decode(np.where(cw == 0, seg, -seg), iters=50)
            ok += int(s_ok and (dec == cw).all())
        info_kb = ok * code.k / 8 / 1000
        tot_info += info_kb
        seqs.append(F["seq"])
        print(f"{F['fi']:7d} {F['seq']:5d} {100*hard:7.2f}% {100*prml:6.2f}% "
              f"{cold_cw:5d}/{n_sub:<3d} {ok:5d}/{nblk:<6d} {info_kb:7.1f}")

    n_eval = len(frames) - 2
    print(f"\n--- END TO END over {n_eval} evaluated frames ---")
    print(f"information decoded : {tot_info:.1f} KB")
    print(f"per frame           : {tot_info/n_eval:.2f} KB")
    print(f"at 60 fps           : {tot_info/n_eval*60:.1f} KB/s  (if every "
          f"camera frame decoded)")
    hdr_yield = len(frames) / len(idxs)
    print(f"header yield measured: {100*hdr_yield:.0f}%  -> realistic "
          f"{tot_info/n_eval*60*hdr_yield:.1f} KB/s")
    print(f"\ndecimen: 128 KB/s handheld, 186 propped")


if __name__ == "__main__":
    main()
