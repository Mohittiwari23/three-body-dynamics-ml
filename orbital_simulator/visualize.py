"""
orbital_simulator/visualize.py
================================
All visualisation for the two-body orbital simulator.

Layout: 3-column, 3-row figure
    Left column  (spans all rows) : Animated relative orbit
    Middle column top             : Effective potential curve
    Middle column mid             : Energy conservation error
    Middle column bot             : Angular momentum conservation error
    Right column  (spans all rows): Orbit residual |Δr(t)|  ← NEW

The residual panel is the most scientifically significant addition for
the research demo.  It provides a *quantitative* bridge between:
    - the numerical trajectory (what the integrator computed)
    - the analytical orbit (what Kepler's equations predict)
A near-zero residual is the strongest possible validation of the framework.

Design principle
----------------
This module receives only SimulationResult and OrbitalSystem objects —
it has zero knowledge of physics formulas.  All numbers it plots were
computed by physics.py and integrator.py.  This makes the visualisation
layer safely replaceable (e.g., switching to a web dashboard) without
touching any physics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional

from .physics import (
    SimulationResult,
    OrbitalSystem,
    effective_potential,
    analytical_orbit_xy,
    energy_scale,
    reduced_mass,
)


# ---------------------------------------------------------------------------
# Colour palette — consistent across all panels
# ---------------------------------------------------------------------------

PALETTE = dict(
    bg_fig    = "#080810",
    bg_ax     = "#0d0d1a",
    grid      = "#1e1e2e",
    spine     = "#2a2a3e",
    title     = "#c4c4d4",
    label     = "#7777aa",
    tick      = "#666688",
    text_box  = "#9999bb",

    traj      = "#38bdf8",   # numerical trajectory — sky blue
    analytic  = "#fb923c",   # analytical orbit     — orange
    body      = "#facc15",   # central body dot     — gold
    ueff      = "#86efac",   # effective potential  — green
    energy    = "#f87171",   # energy error         — red
    momentum  = "#c084fc",   # momentum error       — purple
    residual  = "#fde68a",   # orbit residual       — amber
    rmin      = "#34d399",   # r_min marker         — teal
    rmax      = "#a78bfa",   # r_max marker         — violet
    zero_line = "#333355",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _styled_ax(fig: Figure, slot, title: str, xlabel: str, ylabel: str) -> Axes:
    """Create a consistently styled subplot."""
    ax = fig.add_subplot(slot)
    ax.set_facecolor(PALETTE["bg_ax"])
    ax.set_title(title,   color=PALETTE["title"],  fontsize=9, pad=5,
                 fontfamily="monospace")
    ax.set_xlabel(xlabel, color=PALETTE["label"],  fontsize=8)
    ax.set_ylabel(ylabel, color=PALETTE["label"],  fontsize=8)
    ax.tick_params(colors=PALETTE["tick"], labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["spine"])
    ax.grid(True, color=PALETTE["grid"], lw=0.4, alpha=0.8)
    return ax


def _legend(ax: Axes, **kwargs):
    ax.legend(
        fontsize=7,
        facecolor="#13131f",
        edgecolor=PALETTE["spine"],
        labelcolor="#bbbbcc",
        **kwargs
    )


# ---------------------------------------------------------------------------
# Main visualisation entry point
# ---------------------------------------------------------------------------

def build_animation(
    result: SimulationResult,
    stride: int = 30,
    interval_ms: int = 20,
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    Build and (optionally) save the full animated figure.

    Parameters
    ----------
    result      : SimulationResult from run_simulation()
    stride      : animation frame subsampling (higher = faster animation,
                  fewer frames rendered).  stride=30 means every 30th
                  simulation step becomes one animation frame.
    interval_ms : milliseconds between animation frames
    save_path   : if provided, save animation to this path (.mp4 or .gif)

    Returns
    -------
    FuncAnimation object (keep a reference to prevent GC before plt.show())
    """
    res  = result
    sys_ = result.system
    e, p = res.e, res.p

    # ---- Derived display quantities ----
    r_min  = res.r_min
    r_max  = res.r_max
    # Energy scale: |E0| for bound/hyperbolic; initial KE for parabolic (E0 ≈ 0).
    # Bug fix: previous code compared energy to angular momentum (wrong units).
    mu0    = reduced_mass(sys_.m1, sys_.m2)
    v_rel0 = sys_.v1_0 - sys_.v2_0
    KE0    = 0.5 * mu0 * float(np.dot(v_rel0, v_rel0))
    E_sc   = energy_scale(res.E0, KE0)

    r_lo   = r_min * 0.4
    r_hi   = r_max * 1.6 if np.isfinite(r_max) else r_min * 15.0
    r_view = r_max * 1.3 if np.isfinite(r_max) else r_min * 20.0

    # ---- Analytical orbit ----
    x_an, y_an = analytical_orbit_xy(p, e)

    # ---- Effective potential curve (normalised) ----
    r_vals = np.linspace(r_lo, r_hi, 800)
    U_vals = effective_potential(r_vals, res.L0, sys_.m1, sys_.m2, sys_.G)
    U_norm = np.clip(U_vals / E_sc, -8, 8)
    E_norm = res.E0 / E_sc

    # ---- Orbit view bounds ----
    in_view = np.linalg.norm(res.traj, axis=1) < r_view
    orb_x   = np.concatenate([res.traj[in_view, 0], x_an[np.isfinite(x_an)]])
    orb_y   = np.concatenate([res.traj[in_view, 1], y_an[np.isfinite(y_an)]])
    pad     = 1.25
    x_lo = orb_x.min() * pad if orb_x.min() < 0 else -r_min * pad
    x_hi = orb_x.max() * pad
    y_lo = orb_y.min() * pad
    y_hi = orb_y.max() * pad

    # ---- Residuals (normalised to r_min for readability) ----
    res_norm = res.residuals / r_min if res.residuals is not None else None

    # =====================================================================
    # Figure layout  (3 columns, 3 rows)
    # Col 0 : orbit (all rows)
    # Col 1 : U_eff | ΔE | ΔL
    # Col 2 : orbit residual (all rows)
    # =====================================================================
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 8), facecolor=PALETTE["bg_fig"])
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.05, right=0.97,
        top=0.91,  bottom=0.07,
        wspace=0.38, hspace=0.60
    )

    # --- Panel 1: Orbit (left, full height) ---
    ax_orb = _styled_ax(fig, gs[:, 0], "RELATIVE ORBIT", "x  [length]", "y  [length]")
    ax_orb.set_xlim(x_lo, x_hi)
    ax_orb.set_ylim(y_lo, y_hi)
    ax_orb.set_aspect("equal", "box")

    ax_orb.plot(x_an, y_an, color=PALETTE["analytic"], lw=1.0, ls="--", alpha=0.65,
                label="r(θ) = p / (1 + e·cosθ)")
    orb_line, = ax_orb.plot([], [], color=PALETTE["traj"],  lw=1.3,
                             label="Numerical  (Verlet)")
    orb_dot,  = ax_orb.plot([], [], "o", color="white", ms=5, zorder=5)
    ax_orb.plot(0, 0, "o", color=PALETTE["body"], ms=14, zorder=6, label="Body 2  (CoM origin)")
    _legend(ax_orb, loc="upper right")

    rmax_str = f"{r_max:.3e}" if np.isfinite(r_max) else "∞"
    ax_orb.text(
        0.03, 0.03,
        f"Type  :  {res.orbit_type}\n"
        f"e     =  {e:.6f}\n"
        f"p     =  {p:.3e}\n"
        f"r_min =  {r_min:.3e}\n"
        f"r_max =  {rmax_str}",
        transform=ax_orb.transAxes, color=PALETTE["text_box"],
        fontsize=7.5, fontfamily="monospace", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg_ax"],
                  edgecolor="#2a2a4e", alpha=0.9)
    )

    # --- Panel 2: Effective potential (middle-top) ---
    ax_ueff = _styled_ax(fig, gs[0, 1],
                         "EFFECTIVE POTENTIAL\n"
                         "U_eff(r) = L²/2μr² − GMm/r",
                         "r  [length]", "Energy / scale")
    ax_ueff.plot(r_vals, U_norm, color=PALETTE["ueff"], lw=1.5,
                 label="U_eff(r) / scale")
    ax_ueff.axhline(E_norm, color=PALETTE["energy"], ls="--", lw=1.2,
                    label=f"E₀ / scale = {E_norm:.3f}")
    ax_ueff.axvline(r_min, color=PALETTE["rmin"], ls=":", lw=1.0, label="r_min (periapsis)")
    if np.isfinite(r_max):
        ax_ueff.axvline(r_max, color=PALETTE["rmax"], ls=":", lw=1.0, label="r_max (apoapsis)")
    ax_ueff.fill_between(
        r_vals, U_norm, E_norm,
        where=(U_norm <= E_norm + 0.02),
        color=PALETTE["ueff"], alpha=0.08,
        label="Classically allowed region"
    )
    ax_ueff.set_ylim(-5, 5)
    _legend(ax_ueff, loc="upper right")
    ueff_dot, = ax_ueff.plot([], [], "o", color="white", ms=5, zorder=5)

    # --- Panel 3: Energy conservation (middle-mid) ---
    ax_en = _styled_ax(fig, gs[1, 1],
                       "ENERGY CONSERVATION\n"
                       "Verlet integrator quality check",
                       "time", "(E − E₀) / scale")
    ax_en.set_xlim(0, res.time[-1])
    margin_e = max(np.abs(res.dE).max() * 0.2, 1e-14)
    ax_en.set_ylim(res.dE.min() - margin_e, res.dE.max() + margin_e)
    ax_en.axhline(0, color=PALETTE["zero_line"], lw=0.8)
    en_line, = ax_en.plot([], [], color=PALETTE["energy"], lw=0.9)
    en_dot,  = ax_en.plot([], [], "o", color="white", ms=5, zorder=5)

    # Annotate with peak error
    ax_en.text(0.97, 0.05,
               f"peak |ΔE| = {res.max_energy_error:.2e}",
               transform=ax_en.transAxes, color=PALETTE["energy"],
               fontsize=7, ha="right", fontfamily="monospace")

    # --- Panel 4: Angular momentum (middle-bot) ---
    ax_lm = _styled_ax(fig, gs[2, 1],
                       "ANGULAR MOMENTUM CONSERVATION\n"
                       "Exact symmetry of central force",
                       "time", "(L − L₀) / |L₀|")
    ax_lm.set_xlim(0, res.time[-1])
    margin_l = max(np.abs(res.dL).max() * 0.2, 1e-14)
    ax_lm.set_ylim(res.dL.min() - margin_l, res.dL.max() + margin_l)
    ax_lm.axhline(0, color=PALETTE["zero_line"], lw=0.8)
    lm_line, = ax_lm.plot([], [], color=PALETTE["momentum"], lw=0.9)
    lm_dot,  = ax_lm.plot([], [], "o", color="white", ms=5, zorder=5)

    ax_lm.text(0.97, 0.05,
               f"peak |ΔL| = {res.max_momentum_error:.2e}",
               transform=ax_lm.transAxes, color=PALETTE["momentum"],
               fontsize=7, ha="right", fontfamily="monospace")

    # --- Panel 5: Orbit residual (right, full height) ---
    ax_res = _styled_ax(
        fig, gs[:, 2],
        "ORBIT RESIDUAL\n"
        "|r_numerical − r_analytical(θ)| / r_min\n"
        "← zero = perfect Kepler agreement",
        "time",
        "|Δr| / r_min"
    )
    if res_norm is not None:
        ax_res.set_xlim(0, res.time[-1])
        ymax_res = max(res_norm[np.isfinite(res_norm)].max() * 1.3, 1e-12)
        ax_res.set_ylim(0, ymax_res)
        ax_res.axhline(0, color=PALETTE["zero_line"], lw=0.8)
        res_line, = ax_res.plot([], [], color=PALETTE["residual"], lw=0.9,
                                label="|r_num − r_analytic| / r_min")
        res_dot,  = ax_res.plot([], [], "o", color="white", ms=5, zorder=5)
        ax_res.text(
            0.03, 0.97,
            "This panel validates that the\n"
            "numerical orbit matches the\n"
            "analytical conic prediction.\n\n"
            "Near-zero residual confirms:\n"
            "  • integrator is correct\n"
            "  • (e, p) invariants hold\n"
            "  • framework is trustworthy",
            transform=ax_res.transAxes,
            color=PALETTE["text_box"],
            fontsize=7, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg_ax"],
                      edgecolor="#2a2a4e", alpha=0.85)
        )
        peak_str = f"peak = {res.max_orbit_residual / r_min:.2e} r_min" if res.max_orbit_residual else ""
        ax_res.text(0.97, 0.05, peak_str,
                    transform=ax_res.transAxes, color=PALETTE["residual"],
                    fontsize=7, ha="right", fontfamily="monospace")
        _legend(ax_res, loc="upper right")
    else:
        res_line = res_dot = None
        ax_res.text(0.5, 0.5, "Residuals\nnot computed",
                    transform=ax_res.transAxes, ha="center", va="center",
                    color=PALETTE["text_box"], fontsize=9)

    # --- Figure title ---
    rev_str = (f"  ·  {res.T_orb:.3e} s/rev" if res.T_orb else "  ·  unbound")
    fig.text(
        0.5, 0.965,
        f"NEWTONIAN TWO-BODY ORBITAL SIMULATOR  ·  {sys_.label.upper()}"
        f"  ·  e = {e:.6f}{rev_str}",
        ha="center", color=PALETTE["title"],
        fontsize=9.5, fontfamily="monospace"
    )

    # --- Roadmap annotation (bottom of figure) ---
    fig.text(
        0.5, 0.005,
        "Phase 1: 2D Two-Body  ▸  Phase 2: 3D Extension  ▸  "
        "Phase 3: Perturbations  ▸  Phase 4: N-Body  ▸  Phase 5: ML Stability Prediction",
        ha="center", color="#44446a", fontsize=7, fontfamily="monospace"
    )

    # =====================================================================
    # Animation
    # =====================================================================

    all_artists = [orb_line, orb_dot, ueff_dot, en_line, en_dot, lm_line, lm_dot]
    if res_line is not None:
        all_artists += [res_line, res_dot]

    def init():
        for art in all_artists:
            art.set_data([], [])
        return tuple(all_artists)

    def update(frame: int):
        # Orbit trail
        r_norms = np.linalg.norm(res.traj[:frame], axis=1)
        mask    = r_norms < r_view * 2
        orb_line.set_data(res.traj[:frame][mask, 0], res.traj[:frame][mask, 1])
        orb_dot.set_data([res.traj[frame, 0]], [res.traj[frame, 1]])

        # U_eff dot: current (r, U_eff) position
        r_now = np.linalg.norm(res.traj[frame])
        if r_lo < r_now < r_hi:
            u_now = np.clip(
                effective_potential(r_now, res.L0, sys_.m1, sys_.m2, sys_.G) / E_sc,
                -8, 8
            )
            ueff_dot.set_data([r_now], [u_now])
        else:
            ueff_dot.set_data([], [])

        # Energy / momentum traces
        en_line.set_data(res.time[:frame], res.dE[:frame])
        en_dot.set_data([res.time[frame]], [res.dE[frame]])
        lm_line.set_data(res.time[:frame], res.dL[:frame])
        lm_dot.set_data([res.time[frame]], [res.dL[frame]])

        # Orbit residual trace
        if res_line is not None and res_norm is not None:
            res_line.set_data(res.time[:frame], res_norm[:frame])
            res_dot.set_data([res.time[frame]], [res_norm[frame]])

        return tuple(all_artists)

    frames = range(0, len(res.traj), stride)
    ani = animation.FuncAnimation(
        fig, update,
        frames=frames,
        init_func=init,
        interval=interval_ms,
        blit=True
    )

    if save_path:
        print(f"Saving animation to {save_path} …")
        ani.save(save_path, writer="ffmpeg", dpi=150)
        print("Done.")

    return ani