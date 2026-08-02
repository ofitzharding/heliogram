#!/usr/bin/env python3
"""
recv.py — turn a video of the screen back into the file.

    python3 recv.py ~/Downloads/IMG_7872.MOV
    python3 recv.py capture.MOV --out ~/Desktop

Runs the certified-label receiver over the capture, unwraps the container,
writes the file under its ORIGINAL name, and verifies the sha256 that
travelled with it. The transfer is only reported as successful if that hash
matches - RS plus a per-codeword CRC32 makes corruption astronomically
unlikely, but the point of the exercise is a transfer you can trust without
asking the sender.

Grid geometry is not in the header (the decoder needs it to FIND the header),
so unknown captures are tried against the grids this project has actually
transmitted, newest first.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "poc"))
import container

HERE = Path(__file__).parent
GRIDS = ["252x163", "252x140"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", default=".", help="directory to write the file to")
    ap.add_argument("--grid", default="", help="skip the grid search")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--header-top", action="store_true",
                    help="captures made before the header was centred")
    args = ap.parse_args()
    cap = Path(args.capture).expanduser()
    if not cap.is_file():
        sys.exit(f"no such capture: {cap}")
    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp()) / "raw.bin"
    for gridspec in ([args.grid] if args.grid else GRIDS):
        print(f"=== decoding {cap.name} at {gridspec} ===")
        cmd = [sys.executable, str(HERE / "poc" / "fast_decode.py"),
               str(cap), str(tmp), "--grid", gridspec, "--ecc", str(args.ecc),
               "--subblock", "--soft", "--scan"]
        if args.header_top:
            cmd.append("--header-top")
        r = subprocess.run(cmd)
        if r.returncode == 0 and tmp.is_file():
            break
        print(f"  nothing decodable at {gridspec}\n")
    else:
        sys.exit("FAILED: no grid produced a file. Run poc/quickcheck.py on "
                 "the capture to see which stage is losing it.")

    blob = tmp.read_bytes()
    try:
        name, data, ok = container.unwrap(blob)
    except ValueError as e:
        # A pre-container transmit (the demo payload) still decodes; it just
        # has no name or hash of its own to check against.
        alt = outdir / f"{cap.stem}.recovered.bin"
        alt.write_bytes(blob)
        print(f"\nrecovered {len(blob):,} bytes, but not an SCF1 container "
              f"({e}).\nwrote {alt}")
        return

    dest = outdir / name
    if dest.exists():
        dest = outdir / f"{dest.stem}.recovered{dest.suffix}"
    dest.write_bytes(data)
    print(f"\nfile      {name}")
    print(f"size      {len(data):,} bytes")
    print(f"sha256    {'VERIFIED - byte-identical to the original' if ok else 'MISMATCH'}")
    print(f"written   {dest}")
    if not ok:
        sys.exit("sha256 did not match; do not trust these bytes")


if __name__ == "__main__":
    main()
