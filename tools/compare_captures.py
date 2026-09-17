#!/usr/bin/env python3
"""Equivalence gate for capture runs.

Usage:
    python3 tools/compare_captures.py <run-dir>               # must-differ gate
    python3 tools/compare_captures.py <run-dir> <other-dir>   # per-preset diff between runs

A preset pair that should differ but renders byte-identical means a setting is not
reaching the shader; a must-match pair that differs means an optimization changed the
image. The gate exits non-zero on either.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MUST_DIFFER = [
    ("shadow-interior", "shadow-interior-grazing-sun"),
    ("shadow-interior-grazing-sun-hard", "shadow-interior-grazing-sun-pcf"),
    ("shadow-interior-grazing-sun-pcf", "shadow-interior-grazing-sun-vsm"),
    ("shadow-interior-grazing-sun-hard", "shadow-interior-grazing-sun-vsm"),
    ("shadow-interior-grazing-sun-pcf", "shadow-interior-grazing-sun-bias-zero"),
    ("msaa-4x-interior", "msaa-4x-a2c-interior"),
    ("msaa-4x-interior", "msaa-sample-shading-interior"),
    ("scene-overview", "ibl-off"),
    ("shadow-interior", "ibl-off-interior"),
    ("shadow-interior", "ibl-panorama-interior"),
    ("scene-overview", "msaa-4x"),
    ("sky-procedural", "sky-off"),
    ("tonemap-reinhard", "tonemap-agx"),
    ("scene-overview", "cascades-debug"),
]

# Optimizations that must not change the image.
MUST_MATCH = [
    ("shadow-cull-side-sun", "shadow-cull-side-sun-off"),
    ("scene-overview", "perf-culling-off"),
    ("shadow-interior", "perf-culling-off-interior"),
]

# Pairs whose difference is reported but not gated (e.g. draw order only affects
# coplanar depth ties).
REPORT_ONLY = [
    ("scene-overview", "perf-sort-off"),
    ("shadow-interior", "perf-sort-off-interior"),
]


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def diff(a: Path, b: Path) -> tuple[int, float, float]:
    d = np.abs(load(a) - load(b))
    return int(d.max()), float(d.mean()), 100.0 * float(np.any(d > 0, axis=2).mean())


def gate(run: Path) -> int:
    failures = 0
    print(f"{'pair':<76} {'max':>4} {'mean':>8} {'changed%':>9}")
    for left, right in MUST_DIFFER:
        a, b = run / f"{left}.png", run / f"{right}.png"
        label = f"{left} vs {right}"
        if not (a.exists() and b.exists()):
            print(f"{label:<76} MISSING")
            failures += 1
            continue
        mx, mean, changed = diff(a, b)
        status = "" if mx > 0 else "  IDENTICAL"
        failures += mx == 0
        print(f"{label:<76} {mx:>4} {mean:>8.4f} {changed:>9.3f}{status}")
    for left, right in MUST_MATCH:
        a, b = run / f"{left}.png", run / f"{right}.png"
        label = f"{left} == {right}"
        if not (a.exists() and b.exists()):
            print(f"{label:<76} MISSING")
            failures += 1
            continue
        mx, mean, changed = diff(a, b)
        status = "" if mx == 0 else "  DIFFERS"
        failures += mx != 0
        print(f"{label:<76} {mx:>4} {mean:>8.4f} {changed:>9.3f}{status}")
    for left, right in REPORT_ONLY:
        a, b = run / f"{left}.png", run / f"{right}.png"
        if a.exists() and b.exists():
            mx, mean, changed = diff(a, b)
            print(f"{left + ' ~ ' + right:<76} {mx:>4} {mean:>8.4f} {changed:>9.3f}  (report only)")
    print("gate:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def load_stats(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def compare_runs(run: Path, other: Path) -> int:
    print(f"{'preset':<40} {'max':>4} {'mean':>8} {'changed%':>9}")
    for a in sorted(run.glob("*.png")):
        b = other / a.name
        if not b.exists():
            continue
        mx, mean, changed = diff(a, b)
        print(f"{a.stem:<40} {mx:>4} {mean:>8.4f} {changed:>9.3f}")

    # GPU timing deltas (written by capture mode as <preset>.json), run vs other.
    rows = []
    for a in sorted(run.glob("*.json")):
        new, old = load_stats(a), load_stats(other / a.name)
        if not new or not old:
            continue
        rows.append((a.stem, new, old))
    if not rows:
        return 0
    passes = sorted({k for _, n, o in rows for k in n.get("gpu_ms", {}) if k in o.get("gpu_ms", {})})
    print()
    print("gpu ms (run / other):")
    print(f"{'preset':<40} " + " ".join(f"{p:>15}" for p in passes) + f" {'draws':>9}")
    totals = {p: [0.0, 0.0] for p in passes}
    for name, new, old in rows:
        cells = []
        for p in passes:
            nv, ov = new["gpu_ms"][p], old["gpu_ms"][p]
            totals[p][0] += nv
            totals[p][1] += ov
            cells.append(f"{nv:6.2f}/{ov:6.2f}  ")
        draws = f"{new.get('draw_calls', 0)}/{old.get('draw_calls', 0)}"
        print(f"{name:<40} " + " ".join(f"{c:>15}" for c in cells) + f" {draws:>9}")
    print(f"{'sum':<40} " + " ".join(f"{t[0]:6.2f}/{t[1]:6.2f}  " for t in totals.values()))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.exit(gate(Path(sys.argv[1])))
    if len(sys.argv) == 3:
        sys.exit(compare_runs(Path(sys.argv[1]), Path(sys.argv[2])))
    print(__doc__)
    sys.exit(2)
