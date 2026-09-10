"""Build a sparse 3D walkable-voxel map from CS2 match telemetry.

Takes per-tick player positions (player_vector + player_status) and produces a
sparse voxel grid of observed walkable space: if a player was ever seen standing
somewhere -- alive, on the ground, unassisted -- that location is walkable floor.

Pipeline (see each filter's docstring for the evidence behind it):
  1. Load and join player_vector + player_status on (round, player_id, tick)
  2. Filter to alive players (health > 0)
  3. Drop the last tick of every (player_id, round) -- round-transition artifact
  4. Frame-to-frame teleport check -- safety net for other anomalies
  5. Drop airborne ticks -- fall_velocity primary, z_vel as secondary
     confirmation for the landing-impact frame
  6. Drop boosted ticks (a player standing on a teammate) -- PROVISIONAL
  7. Transform (x_pos, y_pos) -> radar pixel space; z_pos stays in game units
  8. Voxelize into a sparse dict: (px_bin, py_bin, z_bin) -> observation count

Usage:
    # Run on the bundled sample
    python walkable_map.py

    # Run on your own data -- one match
    python walkable_map.py --match path/to/match_dir

    # Run on many matches, accumulating into one map
    python walkable_map.py --matches-root path/to/all_matches --out output/

Each match directory is expected to contain player_vector.parquet and
player_status.parquet. --matches-root scans one level down for such directories.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# reference/ holds provider-supplied files (coordinate transform, radar image).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
from game_to_image import game_to_image  # noqa: E402

# --- Tunable parameters -------------------------------------------------

# CS2 ground speed tops out around 250 units/sec. At this data's tick rate,
# 20 units/tick is far above real movement but far below the round-boundary
# artifacts observed (2000-3700 units/tick), so it catches genuine anomalies.
MAX_UNITS_PER_TICK = 20.0

# Above this |z_vel|, a nominally-grounded tick is treated as the landing-impact
# frame. Set well clear of the 99.9th-percentile grounded noise ceiling (~219).
Z_VEL_LANDING_THRESHOLD = 300.0

# Boost detection -- provisional, see filter_boosted.
BOOST_XY_DIST = 32.0       # roughly the CS2 player collision hull width
BOOST_Z_DIFF_MIN = 50.0    # below one player's standing height: unlikely a boost
BOOST_Z_DIFF_MAX = 95.0    # above this overlaps the inter-floor gap (~195+)

VOXEL_SIZE_PX = 4          # x/y bin size in radar pixels (image is 1024x1024)
VOXEL_SIZE_Z = 32          # z bin size in game units

VECTOR_FILE = "player_vector.parquet"
STATUS_FILE = "player_status.parquet"


# --- Load ---------------------------------------------------------------

def load_match(match_dir):
    """Load and join one match's player_vector and player_status.

    match_dir is a directory containing both parquet files. The join is
    validated one-to-one: both tables are per-tick-per-player, so a fan-out
    would mean the assumption is wrong and every downstream count inflated.
    """
    pv = pd.read_parquet(os.path.join(match_dir, VECTOR_FILE))
    ps = pd.read_parquet(os.path.join(match_dir, STATUS_FILE))
    return pv.merge(
        ps[["round", "player_id", "tick", "health"]],
        on=["round", "player_id", "tick"],
        how="left",
        validate="one_to_one",
    )


# --- Filters ------------------------------------------------------------

def filter_alive(df):
    """Keep only ticks where the player is alive.

    A dead player's recorded position isn't floor a live player chose to stand
    on. This is a no-op on the bundled sample (health never reaches 0 there),
    kept as a guard for datasets where death ticks are present.
    """
    return df[df["health"].fillna(0) > 0].copy()


def drop_round_end_tick(df):
    """Drop the last tick of every (player_id, round).

    On the sample, every single-tick displacement above ~20 units/tick occurs
    exactly on a round's final tick (45/45 cases) -- a position snap at round
    transition, not real movement. Dropping these removes the artifact at its
    source rather than relying on the speed cap to catch it downstream.
    """
    df = df.sort_values(["player_id", "round", "tick"])
    max_tick = df.groupby(["player_id", "round"])["tick"].transform("max")
    return df[df["tick"] != max_tick].copy()


def filter_teleports(df, max_units_per_tick=MAX_UNITS_PER_TICK):
    """Drop ticks implying an impossible speed since the previous tick.

    Safety net for anomalies beyond the round-end case. Removes 0 additional
    rows on the sample once round-end ticks are dropped, which is the expected
    result on clean data -- it earns its place on datasets that aren't.
    """
    df = df.sort_values(["player_id", "round", "tick"]).reset_index(drop=True)
    group = df.groupby(["player_id", "round"])
    dist = np.sqrt(
        group["x_pos"].diff() ** 2
        + group["y_pos"].diff() ** 2
        + group["z_pos"].diff() ** 2
    )
    # First row of each group has no predecessor, so it always passes.
    dist_per_tick = (dist / group["tick"].diff()).fillna(0)
    return df[dist_per_tick <= max_units_per_tick].copy()


def filter_airborne(df, z_vel_threshold=Z_VEL_LANDING_THRESHOLD):
    """Drop airborne ticks: fall_velocity primary, z_vel as confirmation.

    fall_velocity == 0 is the primary grounded signal. It's a physics-engine
    state value, not derived from position, and tracks a clean ballistic curve
    whenever it's nonzero.

    z_vel is NOT usable standalone. It equals dz/second_diff exactly (verified
    at correlation 1.0), so requiring z_vel == 0 for "grounded" misflags ~22%
    of clearly-stationary rows -- exact zero almost never holds under float
    jitter in z_pos (~0.002-0.05 units/tick while standing still). Nor does any
    threshold separate the two cleanly: the 99th percentile of grounded |z_vel|
    (~113) already overlaps the 1st percentile of genuinely airborne (~3).

    It does earn a narrow role as confirmation. Among grounded rows, |z_vel|
    beyond ~300 marks the landing-impact frame: fall_velocity has just reset to
    0 while z_pos still steps down sharply that same tick. Only ~0.04% of
    grounded rows qualify, and the position isn't lost from the map -- adjacent
    ticks sit at the same z, walking normally.
    """
    grounded = df["fall_velocity"] == 0
    landing_impact = grounded & (df["z_vel"].abs() > z_vel_threshold)
    return df[grounded & ~landing_impact].copy()


def filter_boosted(df, xy_dist=BOOST_XY_DIST,
                   z_diff_range=(BOOST_Z_DIFF_MIN, BOOST_Z_DIFF_MAX)):
    """Drop ticks where a player appears boosted (standing on a teammate).

    PROVISIONAL -- these thresholds were set by ruling out false positives, not
    by confirming a true one. A boosted position reflects assisted access, not
    floor a lone player can reach, so it shouldn't count as walkable.

    Flags pairs at the same (round, tick) whose xy positions overlap within a
    collision hull and whose z differs by roughly one player's standing height;
    the higher player is treated as boosted. Two patterns were investigated and
    deliberately excluded from that range:
      - z_diff ~195-300: two players on different floor levels (inspected:
        sustained 26 ticks, a real inter-floor gap, not a lift)
      - z_diff ~40-95 sustained: two players descending a slope together
        (inspected: both z values fall smoothly, neither player stationary)

    No candidate in the 10-match sample survived inspection as a genuine boost,
    so this has never fired on real data. Revisit the thresholds once a
    confirmed boost is available to test against.

    Cost: self-join per (round, tick), O(players^2) per tick. Fine at 10
    players; revisit if that grows.
    """
    sub = df[["round", "tick", "player_id", "x_pos", "y_pos", "z_pos"]]
    pairs = sub.merge(sub, on=["round", "tick"], suffixes=("_a", "_b"))
    pairs = pairs[pairs["player_id_a"] < pairs["player_id_b"]]

    xy = np.hypot(pairs["x_pos_a"] - pairs["x_pos_b"],
                  pairs["y_pos_a"] - pairs["y_pos_b"])
    z_diff = (pairs["z_pos_a"] - pairs["z_pos_b"]).abs()

    candidates = pairs[(xy < xy_dist) & (z_diff.between(*z_diff_range))]
    if candidates.empty:
        return df.copy()

    # The higher player in each flagged pair is the one being boosted.
    top_player = np.where(candidates["z_pos_a"] > candidates["z_pos_b"],
                          candidates["player_id_a"], candidates["player_id_b"])
    boosted = pd.MultiIndex.from_arrays(
        [candidates["round"], candidates["tick"], top_player])
    rows = pd.MultiIndex.from_arrays([df["round"], df["tick"], df["player_id"]])
    return df[~rows.isin(boosted)].copy()


# --- Transform and voxelize ---------------------------------------------

def add_pixel_coords(df):
    """Add radar-image pixel coordinates. z_pos is left in game units."""
    px, py = game_to_image(df["x_pos"].to_numpy(), df["y_pos"].to_numpy())
    return df.assign(px=px, py=py)


def voxelize(df, voxel_size_px=VOXEL_SIZE_PX, voxel_size_z=VOXEL_SIZE_Z):
    """Bin (px, py, z_pos) into a sparse voxel grid.

    Returns {(px_bin, py_bin, z_bin): observation_count}. Sparse rather than a
    dense array because only a small fraction of the map's volume is ever
    walkable -- memory tracks observed space, not map size, which is what makes
    accumulating hundreds of matches practical.
    """
    bins = pd.DataFrame({
        "px_bin": (df["px"] // voxel_size_px).astype(int),
        "py_bin": (df["py"] // voxel_size_px).astype(int),
        "z_bin": (df["z_pos"] // voxel_size_z).astype(int),
    })
    counts = bins.groupby(["px_bin", "py_bin", "z_bin"]).size()
    return {key: int(value) for key, value in counts.items()}


# --- Orchestration ------------------------------------------------------

def process_match(match_dir, verbose=True):
    """Run the full pipeline on one match directory, return its voxel dict."""
    df = load_match(match_dir)
    stages = [("loaded", len(df))]

    for name, fn in [
        ("alive", filter_alive),
        ("no-round-end", drop_round_end_tick),
        ("no-teleport", filter_teleports),
        ("grounded", filter_airborne),
        ("no-boost", filter_boosted),
    ]:
        df = fn(df)
        stages.append((name, len(df)))

    voxels = voxelize(add_pixel_coords(df))

    if verbose:
        trail = " -> ".join(f"{name} {count}" for name, count in stages)
        print(f"{os.path.basename(match_dir)}: {trail} | voxels {len(voxels)}")
    return voxels


def merge_voxels(target, source):
    """Add source's counts into target in place, summing on shared keys.

    Accumulating this way means only one map is ever held in memory, however
    many matches are processed.
    """
    for key, count in source.items():
        target[key] = target.get(key, 0) + count
    return target


def find_match_dirs(root):
    """Find match directories under root (those containing both parquet files).

    Checks root itself first, so a single match directory can be passed to
    either --match or --matches-root without surprises.
    """
    def is_match_dir(path):
        return all(os.path.isfile(os.path.join(path, f))
                   for f in (VECTOR_FILE, STATUS_FILE))

    if not os.path.isdir(root):
        return []
    if is_match_dir(root):
        return [root]
    return sorted(
        os.path.join(root, name) for name in os.listdir(root)
        if is_match_dir(os.path.join(root, name))
    )


def process_matches(match_dirs, verbose=True):
    """Run the pipeline across many matches, accumulating one shared voxel map."""
    combined = {}
    for i, match_dir in enumerate(match_dirs, 1):
        if verbose:
            print(f"[{i}/{len(match_dirs)}] ", end="")
        try:
            merge_voxels(combined, process_match(match_dir, verbose=verbose))
        except Exception as exc:
            # One malformed match shouldn't abort a long multi-match run.
            print(f"  SKIPPED {os.path.basename(match_dir)}: {exc}")
    return combined


# --- Output -------------------------------------------------------------

def export_voxels(voxels, path):
    """Write the voxel map to parquet: px_bin, py_bin, z_bin, count.

    This is the deliverable. The visualization is a QA artifact and can always
    be regenerated from this file.
    """
    out = pd.DataFrame(
        [(px, py, z, count) for (px, py, z), count in voxels.items()],
        columns=["px_bin", "py_bin", "z_bin", "count"],
    ).sort_values(["px_bin", "py_bin", "z_bin"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def save_visualization(voxels, radar_path, out_path, voxel_size_px=VOXEL_SIZE_PX,
                       z_bin=None):
    """Overlay the voxel map on the radar image for a visual sanity check.

    Flattens all heights by default. Pass z_bin (an int, or a (min, max) range)
    to render a single elevation level instead -- useful for confirming that
    stacked areas like palace and connector are actually separated in z rather
    than smeared together.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if z_bin is None:
        selected, label = voxels, "all z"
    else:
        lo, hi = (z_bin, z_bin) if isinstance(z_bin, int) else z_bin
        selected = {k: v for k, v in voxels.items() if lo <= k[2] <= hi}
        label = f"z_bin {lo}" if lo == hi else f"z_bin {lo}..{hi}"

    flat = {}
    for (px_bin, py_bin, _z), count in selected.items():
        flat[(px_bin, py_bin)] = flat.get((px_bin, py_bin), 0) + count

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Image.open(radar_path))
    if flat:
        ax.scatter(
            [k[0] * voxel_size_px for k in flat],
            [k[1] * voxel_size_px for k in flat],
            c=np.log1p(list(flat.values())), cmap="hot", s=6, alpha=0.6,
        )
    ax.set_title(f"Walkable voxels ({label})")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- CLI ----------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(here, "..")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--match", help="one match directory")
    source.add_argument("--matches-root",
                        help="directory of match directories, all accumulated together")
    parser.add_argument("--out", default=os.path.join(repo, "output"),
                        help="output directory (default: output/)")
    parser.add_argument("--radar", default=os.path.join(repo, "reference", "de_mirage_radar.png"),
                        help="radar image for the QA visualization")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the visualization, write only the parquet")
    args = parser.parse_args()

    if args.match:
        match_dirs = [args.match]
    elif args.matches_root:
        match_dirs = find_match_dirs(args.matches_root)
        if not match_dirs:
            parser.error(f"no match directories found under {args.matches_root}")
    else:
        match_dirs = [os.path.join(repo, "sample_data")]

    voxels = process_matches(match_dirs)
    if not voxels:
        parser.error("no voxels produced -- check the input data")

    os.makedirs(args.out, exist_ok=True)
    parquet_path = os.path.join(args.out, "walkable_voxels.parquet")
    export_voxels(voxels, parquet_path)
    print(f"\n{len(match_dirs)} match(es) -> {len(voxels)} voxels -> {parquet_path}")

    if not args.no_viz:
        viz_path = os.path.join(args.out, "walkable_voxels_viz.png")
        save_visualization(voxels, args.radar, viz_path)
        print(f"visualization -> {viz_path}")


if __name__ == "__main__":
    main()
