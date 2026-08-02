# Demo payloads

The images shipped as test payloads, with their licences.

| file | source | author | licence |
|---|---|---|---|
| `kitten.png` | [Six weeks old cat (aka).jpg](https://commons.wikimedia.org/wiki/File:Six_weeks_old_cat_(aka).jpg) | André Karwath (Aka) | CC BY-SA 2.5 |
| `kitten_big.png` | [Kitten in Rizal Park, Manila.jpg](https://commons.wikimedia.org/wiki/File:Kitten_in_Rizal_Park,_Manila.jpg) | Kenny Louie, Vancouver | CC BY 2.0 |

Both are downscaled and re-encoded to PNG. `kitten.png` is sized to ~273 KB and
`kitten_big.png` to ~1.11 MB, matching the payload sizes every throughput figure
in this repository was measured against, so the demo behaves identically.

The original measurement payload was a third-party infographic that is not ours
to redistribute. It is deliberately not tracked here. Figures quoted in the
commit history against sha256 `82e04ae3...` refer to that file; `kitten.png`
reproduces the same behaviour at the same size but will of course hash
differently.
