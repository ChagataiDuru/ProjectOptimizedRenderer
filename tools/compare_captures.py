#!/usr/bin/env python3
"""Equivalence gate for capture runs.

Usage:
    python3 tools/compare_captures.py <run-dir>               # must-differ gate
    python3 tools/compare_captures.py <run-dir> <other-dir>   # per-preset diff between runs

A preset pair that should differ but renders byte-identical means a setting is not
reaching the shader; a must-match pair that differs means an optimization changed the
image. The gate exits non-zero on either.
"""
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
    ("scene-overview", "msaa-4x"),
    ("sky-procedural", "sky-off"),
    ("tonemap-reinhard", "tonemap-agx"),
    ("scene-overview", "cascades-debug"),
]

# Optimizations that must not change the image.
MUST_MATCH = [
    ("shadow-cull-side-sun", "shadow-cull-side-sun-off"),
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
    print("gate:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def compare_runs(run: Path, other: Path) -> int:
    print(f"{'preset':<40} {'max':>4} {'mean':>8} {'changed%':>9}")
    for a in sorted(run.glob("*.png")):
        b = other / a.name
        if not b.exists():
            continue
        mx, mean, changed = diff(a, b)
        print(f"{a.stem:<40} {mx:>4} {mean:>8.4f} {changed:>9.3f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.exit(gate(Path(sys.argv[1])))
    if len(sys.argv) == 3:
        sys.exit(compare_runs(Path(sys.argv[1]), Path(sys.argv[2])))
    print(__doc__)
    sys.exit(2)
