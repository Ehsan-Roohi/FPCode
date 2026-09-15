"""CPU-only smoke test for the cylinder mesh visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mesh_visualization import plot_solver_mesh, reachable_leaf_indices


class HostMirror:
    """Minimal stand-in for a Numba device array used by the smoke test."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def __len__(self):
        return len(self.values)

    def copy_to_host(self):
        return self.values.copy()


class MockSolver:
    def __init__(self, centers, sizes, children):
        self.tree_center = HostMirror(centers)
        self.tree_size = HostMirror(sizes)
        self.tree_children = HostMirror(children)
        self.next_free = HostMirror([len(centers)])


def build_test_tree(depth: int = 3):
    centers = [[0.5, 0.5]]
    sizes = [[0.5, 0.5]]
    children = [[-1, -1, -1, -1]]
    frontier = [0]
    for _ in range(depth):
        next_frontier = []
        for parent in frontier:
            cx, cy = centers[parent]
            hw, hh = np.asarray(sizes[parent]) * 0.5
            start = len(centers)
            children[parent] = [start, start + 1, start + 2, start + 3]
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                centers.append([cx + dx * hw, cy + dy * hh])
                sizes.append([hw, hh])
                children.append([-1, -1, -1, -1])
                next_frontier.append(len(centers) - 1)
        frontier = next_frontier

    # Add an unreachable former child to verify that traversal, rather than a
    # flat children==-1 scan, selects the active mesh.
    centers.append([0.9, 0.9])
    sizes.append([0.01, 0.01])
    children.append([-1, -1, -1, -1])
    return np.asarray(centers), np.asarray(sizes), np.asarray(children, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/mesh_geometry_quick_check.png"),
    )
    args = parser.parse_args()

    centers, sizes, children = build_test_tree(depth=3)
    active = reachable_leaf_indices(children)
    plotted = plot_solver_mesh(
        MockSolver(centers, sizes, children),
        r_cyl=0.1524,
        r_dom=0.65,
        filename=args.output,
    )
    assert len(active) == 64, f"expected 64 active leaves, got {len(active)}"
    assert np.array_equal(active, plotted)
    assert args.output.is_file() and args.output.stat().st_size > 0
    print(f"PASS: {len(plotted)} active leaves; wrote {args.output}")


if __name__ == "__main__":
    main()
