"""Mesh plotting helpers for the adaptive half-annulus cylinder cases.

The solver stores every node ever allocated.  After coarsening, former children
remain in those arrays but are no longer part of the active tree.  Therefore an
active mesh must be recovered by walking from the root; neither ``rho > 0`` nor
``children[:, 0] == -1`` alone identifies the current leaf set.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Wedge


def reachable_leaf_indices(tree_children: np.ndarray, root: int = 0) -> np.ndarray:
    """Return active leaf indices by traversing the quadtree from ``root``."""
    children = np.asarray(tree_children)
    if children.ndim != 2 or children.shape[1] != 4:
        raise ValueError("tree_children must have shape (n_nodes, 4)")
    if not 0 <= root < len(children):
        raise ValueError("root index is outside tree_children")

    leaves: list[int] = []
    stack = [root]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            raise ValueError(f"cycle detected in quadtree at node {node}")
        visited.add(node)

        row = children[node]
        if row[0] < 0:
            leaves.append(node)
            continue

        for child in row[::-1]:
            child = int(child)
            if child < 0 or child >= len(children):
                raise ValueError(f"invalid child index {child} at node {node}")
            stack.append(child)

    return np.asarray(leaves, dtype=np.int64)


def _cell_polygon(
    center: np.ndarray,
    half_size: np.ndarray,
    r_cyl: float,
    r_dom: float,
    arc_points: int,
) -> np.ndarray:
    """Map one logical (xi, eta) cell to a curved physical-space polygon."""
    xi0, eta0 = center - half_size
    xi1, eta1 = center + half_size
    xi0, xi1 = np.clip([xi0, xi1], 0.0, 1.0)
    eta0, eta1 = np.clip([eta0, eta1], 0.0, 1.0)

    dr = r_dom - r_cyl
    r0 = r_cyl + xi0 * dr
    r1 = r_cyl + xi1 * dr
    theta0 = eta0 * math.pi
    theta1 = eta1 * math.pi
    theta_out = np.linspace(theta0, theta1, arc_points)
    theta_in = theta_out[::-1]

    outer = np.column_stack((r1 * np.cos(theta_out), r1 * np.sin(theta_out)))
    inner = np.column_stack((r0 * np.cos(theta_in), r0 * np.sin(theta_in)))
    return np.vstack((outer, inner))


def plot_half_annulus_mesh(
    tree_center: np.ndarray,
    tree_size: np.ndarray,
    tree_children: np.ndarray,
    r_cyl: float,
    r_dom: float,
    filename: str | Path,
    *,
    title: str = "Adaptive mesh (upper half-domain)",
    arc_points: int = 5,
) -> np.ndarray:
    """Plot the complete active mesh and return the plotted leaf indices."""
    centers = np.asarray(tree_center)
    sizes = np.asarray(tree_size)
    children = np.asarray(tree_children)
    if centers.shape != sizes.shape or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("tree_center and tree_size must both have shape (n_nodes, 2)")
    if len(children) != len(centers):
        raise ValueError("tree arrays must contain the same number of nodes")
    if not (0.0 < r_cyl < r_dom):
        raise ValueError("expected 0 < r_cyl < r_dom")

    leaf_indices = reachable_leaf_indices(children)
    polygons = []
    valid_leaves = []
    tol = 1.0e-6
    for idx in leaf_indices:
        center = centers[idx]
        half_size = sizes[idx]
        lower = center - half_size
        upper = center + half_size
        if (
            np.all(np.isfinite(center))
            and np.all(np.isfinite(half_size))
            and np.all(half_size > 0.0)
            and np.all(lower >= -tol)
            and np.all(upper <= 1.0 + tol)
        ):
            polygons.append(
                _cell_polygon(center, half_size, r_cyl, r_dom, max(2, arc_points))
            )
            valid_leaves.append(int(idx))

    if not polygons:
        raise ValueError("no valid active leaves were found")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    mesh = PolyCollection(
        polygons,
        edgecolors="black",
        facecolors="none",
        linewidths=0.35,
        rasterized=True,
    )
    ax.add_collection(mesh)

    # Explicitly mask and draw the solid cylinder so its geometry cannot be
    # confused with empty statistical cells in the particle solution.
    cylinder = Wedge(
        (0.0, 0.0),
        r_cyl,
        0.0,
        180.0,
        facecolor="white",
        edgecolor="tab:red",
        linewidth=1.8,
        zorder=5,
    )
    ax.add_patch(cylinder)
    theta = np.linspace(0.0, math.pi, 512)
    ax.plot(r_dom * np.cos(theta), r_dom * np.sin(theta), color="black", linewidth=1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.02 * r_dom, 1.02 * r_dom)
    ax.set_ylim(0.0, 1.02 * r_dom)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return np.asarray(valid_leaves, dtype=np.int64)


def plot_solver_mesh(solver, r_cyl: float, r_dom: float, filename: str | Path) -> np.ndarray:
    """Copy a solver's tree to host memory and plot its active mesh."""
    n_max = min(int(solver.next_free.copy_to_host()[0]), len(solver.tree_center))
    centers = solver.tree_center.copy_to_host()[:n_max]
    sizes = solver.tree_size.copy_to_host()[:n_max]
    children = solver.tree_children.copy_to_host()[:n_max]
    return plot_half_annulus_mesh(centers, sizes, children, r_cyl, r_dom, filename)
