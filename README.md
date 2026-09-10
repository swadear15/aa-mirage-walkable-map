# Mirage Walkable Voxel Map

Builds a sparse 3D map of walkable space on `de_mirage` from real match
telemetry (`player_vector` + `player_status`), verified against Valve's radar
image.

The output is a voxel grid rather than a flat 2D floor outline, so elevation is
preserved. Mirage has genuinely overlapping geometry — palace above mid,
catwalk above connector — and a top-down projection would collapse those into
one blob, marking a location walkable when it's only walkable at one height.

## Layout

```
src/walkable_map.py      the pipeline
reference/               provider-supplied coordinate transform + radar image
sample_data/             one match of telemetry, enough to run and validate
output/                  result of running the pipeline on sample_data
```

## Quick start

```bash
pip install -r requirements.txt
cd src
python walkable_map.py
```

That runs on the bundled sample and writes `output/walkable_voxels.parquet`
(the data product) plus `output/walkable_voxels_viz.png` (a QA overlay on the
radar image). Expect roughly 11,500 voxels from ~544,000 filtered position
observations.

## Running on your own data

A match directory is any directory containing both `player_vector.parquet` and
`player_status.parquet`.

```bash
# One match
python walkable_map.py --match /path/to/match_dir

# Many matches, accumulated into a single map
python walkable_map.py --matches-root /path/to/all_matches

# Custom output location, skip the visualization
python walkable_map.py --matches-root /path/to/all_matches --out /path/to/out --no-viz
```

`--matches-root` scans one level down for match directories, and also accepts a
single match directory directly. Matches that fail to load are reported and
skipped rather than aborting the run, so one malformed file doesn't cost you a
long batch.

Memory stays flat across any number of matches: each match is processed and
merged into one shared voxel map, never held all at once.

## How it works

If a player was ever observed standing somewhere — alive, on the ground,
unassisted — that location is walkable floor. The pipeline's job is filtering
out every tick that doesn't meet that bar before counting it as evidence.

| Filter | Removes | Why |
|---|---|---|
| Alive check | Dead-player positions | A corpse's position isn't floor a live player chose to stand on. No-op on the sample (health never reaches 0 there), kept as a guard for other datasets. |
| Round-end tick | Last tick of every player-round | Confirmed artifact: position snaps at round transition, producing spurious 2000+ unit jumps. All 45 anomalous displacements in the sample land exactly here. |
| Teleport check | Impossible single-tick displacements | Safety net for anomalies beyond the round-end case. Removes 0 extra rows on the sample, which is the expected result on clean data. |
| Airborne | Jump/fall arcs, plus the landing-impact frame | A player mid-air isn't standing on floor. `fall_velocity == 0` is the primary signal; `z_vel` confirms the landing frame. See below. |
| Boost (**provisional**) | Ticks where a player stands on a teammate | A boosted position reflects assisted access, not floor a lone player can reach. Never validated against a real boost — see limitations. |

Surviving positions are converted to radar pixel space via
`reference/game_to_image.py`, then binned: 4 pixels per bin in x/y, 32 game
units per bin in z. Voxels are stored sparse, since only a small fraction of the
map's volume is ever walkable.

### On `z_vel`

`z_vel` looks like the obvious airborne signal, but it can't carry that job
alone. It equals `dz / second_diff` exactly (verified at correlation 1.0), so
requiring `z_vel == 0` for "grounded" misflags about 22% of clearly-stationary
rows — exact zero almost never holds when `z_pos` jitters by 0.002–0.05 units
per tick while a player stands still. Thresholding doesn't rescue it either:
the 99th percentile of grounded `|z_vel|` (~113) already overlaps the 1st
percentile of genuinely airborne values (~3).

It does earn a narrow role. Among otherwise-grounded ticks, `|z_vel|` above 300
marks the landing-impact frame — the instant `fall_velocity` resets to 0 while
`z_pos` still steps down sharply. Only ~0.04% of grounded rows qualify, and
nothing is lost from the map, since adjacent ticks sit at the same height
walking normally.

## Using the output

`output/walkable_voxels.parquet`:

| Column | Type | Meaning |
|---|---|---|
| `px_bin` | int | x bin in radar pixel space (`px_bin * 4` ≈ pixel x) |
| `py_bin` | int | y bin in radar pixel space |
| `z_bin` | int | height bin in game units (`z_bin * 32` ≈ game z) |
| `count` | int | observed ticks in this voxel, summed across all processed matches |

Checking whether a position is walkable:

```python
import pandas as pd, sys
sys.path.append("reference")
from game_to_image import game_to_image

voxels = pd.read_parquet("output/walkable_voxels.parquet")
lookup = set(zip(voxels.px_bin, voxels.py_bin, voxels.z_bin))

def is_walkable(x_pos, y_pos, z_pos):
    px, py = game_to_image(x_pos, y_pos)
    return (int(px // 4), int(py // 4), int(z_pos // 32)) in lookup
```

Dropping low-confidence voxels before treating the map as ground truth:

```python
confident = voxels[voxels["count"] >= 5]
```

Pick that threshold from your own count distribution — 5 is a starting point,
not a validated cutoff.

Rendering a single elevation level instead of the flattened view:

```python
from walkable_map import process_match, save_visualization
voxels = process_match("sample_data")
save_visualization(voxels, "reference/de_mirage_radar.png", "palace.png", z_bin=-1)
```

Converting a voxel back toward game coordinates, for joining against other
telemetry tables:

```python
from game_to_image import image_to_game

def voxel_to_game(px_bin, py_bin, z_bin):
    x, y = image_to_game(px_bin * 4, py_bin * 4)
    return x, y, z_bin * 32
```

This returns the voxel's corner, not the exact observed point — precision is
bounded by voxel size.

## Limitations

- **The boost filter has never fired on real data.** Its thresholds were set by
  ruling out two false-positive patterns (players on different floors, players
  descending a slope together), not by confirming a true boost. No candidate in
  the sample survived inspection as genuine. Treat it as provisional until
  tested against a confirmed example.
- **Voxel size was chosen, not swept.** 4px / 32 units separates Mirage's known
  elevation levels without exploding memory, but no alternatives were
  benchmarked.
- **This is observed play, not true map geometry.** Corners nobody visits stay
  empty even when they're technically walkable. Coverage improves with more
  matches, which is what `--matches-root` is for.
