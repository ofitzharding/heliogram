#!/usr/bin/env python3
"""
verify_webtx.py — one command, every transmitter proven before any take.

Bit-compares every demo/webtx*/frames.bin against the truth renderer (the
same encoder+renderer chain the analyzers were dress-rehearsed on). A take
filmed against a verified transmitter cannot be wasted by an encoding bug.

    python3 poc/verify_webtx.py            # all demo/webtx* dirs
    python3 poc/verify_webtx.py demo/webtx11
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from analyze_strobe import truth_cells


def verify(d):
    meta = json.load(open(d / "meta.json"))
    gw, gh, nf = meta["gw"], meta["gh"], meta["frames"]
    grid.set_ecc(meta["ecc"]); grid.set_header_len(28)
    grid.set_header_centered(True)
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    payload = Path(__file__).parent.parent / "demo" / meta["name"]
    data = payload.read_bytes()
    enc = fountain.Encoder(data, meta["sub"])
    if (n_sub, enc.k, len(data)) != (meta["n_sub"], meta["k"],
                                     meta["file_size"]):
        return f"META MISMATCH n_sub/k/size"
    bpf = meta["bytes_per_frame"]
    scratch = {}
    bad = 0
    with open(d / "frames.bin", "rb") as f:
        for seq in range(nf):
            chunk = np.frombuffer(f.read(bpf), np.uint8)
            if len(chunk) < bpf:
                return f"TRUNCATED at frame {seq}"
            want = truth_cells(L, enc, n_sub, meta["sub"], len(data), seq,
                               cache=scratch)
            got = np.unpackbits(chunk)[:gw * gh].reshape(gh, gw)
            if not np.array_equal(got, want.astype(np.uint8)):
                bad += 1
            scratch.clear()
    return "CLEAN" if bad == 0 else f"{bad}/{nf} frames MISMATCH"


def main():
    base = Path(__file__).parent.parent
    dirs = ([Path(a) for a in sys.argv[1:]] or
            sorted(base.glob("demo/webtx*")))
    fail = False
    for d in dirs:
        if not (d / "meta.json").exists():
            print(f"{d.name:>10s}: no meta.json (still rendering?)")
            fail = True
            continue
        v = verify(d)
        m = json.load(open(d / "meta.json"))
        print(f"{d.name:>10s}: {m['gw']}x{m['gh']} {m['name']} "
              f"k={m['k']} ceiling {m['ceiling_kbs']} -> {v}")
        fail |= v != "CLEAN"
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
