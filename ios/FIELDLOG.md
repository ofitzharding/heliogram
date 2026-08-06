# heliogram iOS field log

Append one entry per live run: date, file size, full-span KB/s, yield,
ms/frame, located/tracked/frames, k1, eye, one line of conditions.
Failures with numbers are findings; log them.

## Inherited browser-pair envelope (2026-08-06, for reference)

Best completed: 1,511,987 B bit-exact at 16.8 KB/s hand-held (4K30,
28 seq/s sender). Best instantaneous pool rate 25.1 KB/s. Final state
of the browser receiver: 82% tracked share, k1 -0.020 stable, eye 74,
yield 0% steady-state: the channel sits between the header's (~29%
byte) and the payload's (~9-13% byte) correction cliffs. SPEC section
7.8 is the response. The film pipeline's offline record on the same
hardware pair is 229.7 KB/s full-span at 4K60; that is the existence
proof for the channel, not a browser expectation.
