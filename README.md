# Image Ranker

A tiny local web app for ranking images by pairwise comparison. Two images from `Samples/` are shown side-by-side on a white background; you pick the one you prefer with the arrow keys. Each vote is appended to `votes.jsonl` so rankings can be computed offline later (e.g. Elo or Bradley–Terry).

## Requirements

- Python 3.9+ (standard library only — no packages to install)
- A modern browser

## Layout

```
comparer/
├── Samples/        # put the PNG images you want to rank here
├── server.py       # local HTTP server + vote logger
├── index.html      # page shell
├── style.css       # white background, centered flex layout
├── app.js          # pair fetching, keyboard handling, vote POST
└── votes.jsonl     # created on first vote, one JSON object per line
```

## Run

From the project directory:

```bash
python3 server.py
```

You should see:

```
serving on http://127.0.0.1:8000 — 38 images loaded
```

Open http://127.0.0.1:8000 in a browser.

Stop the server with `Ctrl-C`. Votes are flushed to disk as they happen, so you can stop and resume at any time.

## Use

- Two images appear side-by-side.
- Press **`←`** if you prefer the **left** image.
- Press **`→`** if you prefer the **right** image.
- The next pair appears immediately.

That is the entire interaction. No other controls, no on-screen text — the page is deliberately minimal so nothing biases the comparison.

Behind the scenes, each pair is rendered at one of three sizes (small / medium / large), chosen randomly per pair. The size isn't shown but is recorded with the vote so you can analyze size-dependent effects later.

## Vote log format

`votes.jsonl` — one JSON object per line, append-only:

```json
{"ts":"2026-04-19T14:32:30Z","pair_id":"17eabe79…","a":"11.png","b":"19.png","size":"large","winner":"a"}
```

Fields:
- `ts` — UTC timestamp when the vote was recorded
- `pair_id` — unique id for the pair that was shown (UUID4 hex)
- `a`, `b` — filenames of the left and right images
- `size` — `small` (20 px tall), `medium` (40 px), or `large` (60 px)
- `winner` — `"a"` or `"b"`

## Computing rankings

The log format is trivial to load. Example with Python:

```python
import json
votes = [json.loads(l) for l in open("votes.jsonl")]
```

From there, feed the pairs into your favorite ranking method — Elo, Bradley–Terry, TrueSkill, or a simple win-rate sort.

## Adding or removing images

Drop `.png` files into `Samples/` (or remove them) and restart the server. The image list is captured at startup; the server ignores anything that isn't a `.png` file in that directory.

## Configuration

The constants near the top of `server.py` are the only knobs:

- `HOST`, `PORT` — where the server binds (default `127.0.0.1:8000`)
- `SIZES` — the three size buckets used for display
- The pixel heights for each size live in `style.css` under `#stage[data-size=…]`

## Notes

- The server binds to `127.0.0.1` only — not reachable from the network.
- Pairs are drawn uniformly at random with replacement, so the same pair can recur. That's intentional — repeated comparisons improve ranking stability.
- Filenames in vote requests are validated against the startup directory listing, so path traversal isn't possible.
