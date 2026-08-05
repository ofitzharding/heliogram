#!/usr/bin/env python3
"""
send.py — put a file on the screen so a phone can film it.

    python3 send.py notes.pdf
    python3 send.py notes.pdf --no-play      # just build the video

Encodes any file into the 252x163 grid code, renders a lock-in lead plus the
transmit loop, and plays both fullscreen. Film it, AirDrop the video back, and
run recv.py.

The file's NAME and sha256 travel inside the payload (container.py), so the
receiver writes the right filename and can tell you whether the bytes are
right without a back-channel.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import container
import make_record

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--lead-seconds", type=float, default=22.0)
    ap.add_argument("--grid", default="252x163")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    blob = container.wrap(src.name, src.read_bytes())
    out = str(HERE / "demo" / "_send.mp4")
    info = make_record.build(blob, out, grid_spec=args.grid,
                             lead_seconds=args.lead_seconds, label=src.name)

    secs = info["frames"] / 60.0
    print(f"file         {src.name}  ({src.stat().st_size:,} B, "
          f"{len(blob):,} B wrapped)")
    print(f"one loop     {secs:.1f}s; playing {args.loops} loops "
          f"= {args.loops*secs + args.lead_seconds:.0f}s total")
    print(f"a clean receiver needs about "
          f"{1.05*info['k']/info['n_sub']/60:.1f}s of good capture\n")
    if args.no_play:
        print(f"built {out} and {info['lead']}")
        return

    subprocess.run(["osascript",
                    "-e", 'tell application "System Events" to tell dock '
                          'preferences to set autohide to true',
                    "-e", 'tell application "System Events" to tell dock '
                          'preferences to set autohide menu bar to true'],
                   capture_output=True)
    print("""------------------------------------------------------------------
  Evening light, ONE lamp. Stock Camera, 4K 60, landscape, 1x.
  Fill the frame with the screen.
  Hit record, then TAP-AND-HOLD to lock AE/AF when prompted.
  Do NOT touch the exposure slider - it cuts exposure without
  raising ISO, so a short shutter necessarily underexposes.
  Then hold still. The camera takes ~7s to settle after the lock.
------------------------------------------------------------------""")
    time.sleep(3)
    t0 = time.time()
    for f, loops in ((info["lead"], 1), (out, args.loops)):
        cmd = ["ffplay", "-v", "error", "-fs", "-alwaysontop", "-noborder",
               "-autoexit"]
        if loops > 1:
            cmd += ["-loop", str(loops)]
        subprocess.run(cmd + [f])
    subprocess.run(["osascript",
                    "-e", 'tell application "System Events" to tell dock '
                          'preferences to set autohide to false',
                    "-e", 'tell application "System Events" to tell dock '
                          'preferences to set autohide menu bar to false'],
                   capture_output=True)
    print(f"\ndisplayed {time.time()-t0:.1f}s. Stop recording, AirDrop it, then:")
    print(f"  python3 recv.py ~/Downloads/IMG_XXXX.MOV")


if __name__ == "__main__":
    main()
