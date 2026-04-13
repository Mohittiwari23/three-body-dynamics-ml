"""
visualize3.py  –  Curated dark-theme animation for the 3-body simulator.

Layout (18 × 9 figure)
───────────────────────────────────────────────────────────
LEFT  (col 0, full height) : Orbital trajectories

RIGHT (col 1, 2×2 + 1 wide)
  [0,0] Conservation score  – single live number + bar
  [0,1] Pair distances      – r12, r13, r23 with close-approach markers
  [1,0] Phase portrait      – x₁ vs ẋ₁ Poincaré-style
  [1,1] Angular momentum    – ΔL/|L₀| cumulative drift
  [2, :] Centre-of-mass drift – should stay near zero; a nice sanity check
───────────────────────────────────────────────────────────

Removed panels  (and why)
  • ΔE error     – oscillates at integrator frequency, masks real signal
  • MEGNO        – flat line for figure-8 (quasi-periodic), zero information
  • Kinetic Enrg – redundant with distance panel (correlated signal)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.lines as mlines

# ── palette ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")

BG         = "#03050d"
PANEL_BG   = "#080c18"
BORDER_CLR = "#0f1a30"
GRID_CLR   = "#0d1526"
TITLE_CLR  = "#c8deff"
LABEL_CLR  = "#4a6a94"
TICK_CLR   = "#2a3f5f"
VALUE_CLR  = "#7ab0e0"

BC  = ["#00d4ff", "#ffaa00", "#7fff00"]   # body bright
BGC = ["#003344", "#332200", "#152e00"]   # body glow dark
BWC = ["#00aacc", "#cc8800", "#55cc00"]   # trail mid

COL_L   = "#8aff8a"   # momentum drift
COL_R12 = "#ff4dac"
COL_R13 = "#ffd740"
COL_R23 = "#a07cff"
COL_COM = "#ff9f43"   # centre of mass
COL_PH  = "#00d4ff"

MONO = "monospace"

# ── helpers ───────────────────────────────────────────────────────────────────

def _style_panel(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER_CLR); sp.set_linewidth(0.8)
    ax.tick_params(colors=TICK_CLR, labelsize=5.5, length=2,
                   width=0.5, labelcolor=LABEL_CLR)
    ax.grid(True, color=GRID_CLR, lw=0.35, ls="-", alpha=1.0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=6.5, color=TITLE_CLR,
                     fontfamily=MONO, fontweight="bold", pad=5, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=5.5, color=LABEL_CLR,
                      fontfamily=MONO, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=5.5, color=LABEL_CLR,
                      fontfamily=MONO, labelpad=2)


def _starfield(ax, n=260, seed=7):
    rng = np.random.default_rng(seed)
    ax.scatter(rng.uniform(0,1,n), rng.uniform(0,1,n),
               s=rng.exponential(0.3,n).clip(0.05,1.4),
               color="white", alpha=rng.uniform(0.08,0.50,n),
               transform=ax.transAxes, zorder=0, animated=False)


def _make_trail_lc(ax, color, trail_len, lw=1.3):
    nan_seg = np.full((trail_len, 2, 2), np.nan)
    rgb = matplotlib.colors.to_rgb(color)
    alphas = np.power(np.linspace(0, 1, trail_len), 1.8)
    colors = [(rgb[0], rgb[1], rgb[2], a) for a in alphas]
    lc = LineCollection(nan_seg, colors=colors, linewidths=lw,
                        capstyle="round", joinstyle="round", animated=True)
    ax.add_collection(lc)
    return lc, rgb


def _fill_poly(ax, color, alpha=0.12):
    poly = Polygon([[0,0],[0,0]], closed=True,
                   facecolor=color, edgecolor="none",
                   alpha=alpha, animated=True, zorder=0)
    ax.add_patch(poly)
    return poly


def _poly_xy(x, y):
    if len(x) == 0:
        return np.zeros((4,2))
    return np.vstack([np.c_[x,y], np.c_[x[::-1], np.zeros(len(x))]])


def _hline_note(ax, y, color, label, side="right"):
    ax.axhline(y, color=color, lw=0.6, ls="--", alpha=0.55, zorder=1)
    xp = 1.0 if side == "right" else 0.0
    ha = "left" if side == "right" else "right"
    ax.text(xp, y, f" {label} ", transform=ax.get_yaxis_transform(),
            color=color, fontsize=4.8, fontfamily=MONO,
            va="center", alpha=0.75, ha=ha)


# ── main ──────────────────────────────────────────────────────────────────────

def animate_3body(result, stride=10, trail_len=400, save_path=None):
    """
    Parameters
    ----------
    result     : simulation result object
    stride     : frame stride (raise to 15-20 to go faster)
    trail_len  : fading-trail length (reduce to 200 for speed)
    save_path  : optional path to write .mp4
    """

    # ── pre-compute ───────────────────────────────────────────────────────────
    traj = result.traj          # (T, 3, 2)
    t    = result.time
    T    = t[-1]
    Nfr  = len(traj)

    # conservation errors
    dE = (result.E_hist  - result.E0) / (np.abs(result.E0)  + 1e-12)
    dL = (result.L_hist  - result.L0) / (np.abs(result.L0)  + 1e-12)

    # "conservation score"  1 – log10 of max(|ΔE|, |ΔL|), clamped to [0,16]
    worst = np.maximum(np.abs(dE), np.abs(dL))
    score = np.clip(-np.log10(worst + 1e-16), 0, 16)

    # pair distances
    r12 = np.linalg.norm(traj[:,0]-traj[:,1], axis=1)
    r13 = np.linalg.norm(traj[:,0]-traj[:,2], axis=1)
    r23 = np.linalg.norm(traj[:,1]-traj[:,2], axis=1)
    rmin_global = min(r12.min(), r13.min(), r23.min())

    # centre-of-mass drift (should be ~0 for equal masses)
    com = traj.mean(axis=1)                     # (T, 2)
    com_drift = np.linalg.norm(com - com[0], axis=1)

    # phase-space  (x₁, ẋ₁)
    has_vel = hasattr(result, "vel_hist") and result.vel_hist is not None
    if has_vel:
        vx1 = result.vel_hist[:,0,0]
    else:
        vx1 = np.gradient(traj[:,0,0], t)

    x1  = traj[:,0,0]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor=BG, dpi=90)

    fig.text(0.5, 0.967,
             "THREE-BODY  ·  FIGURE-8  ORBIT  ·  CHAOS DIAGNOSTIC",
             ha="center", fontsize=10, fontfamily=MONO, fontweight="bold",
             color="#5a9fd4")
    fig.add_artist(mlines.Line2D(
        [0.04, 0.96], [0.946, 0.946],
        transform=fig.transFigure, color=BORDER_CLR, lw=0.7))

    # outer: orbit | right panel block
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.04, right=0.97, top=0.935, bottom=0.07,
        wspace=0.035, width_ratios=[1.05, 1.0])

    # right: 3 rows × 2 cols
    gs_r = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs_outer[1],
        hspace=0.72, wspace=0.42,
        height_ratios=[0.72, 1, 0.72])

    # ── ORBIT (left, full height) ──────────────────────────────────────────────
    ax_orb = fig.add_subplot(gs_outer[0])
    _style_panel(ax_orb, title="ORBITAL TRAJECTORIES")
    ax_orb.set_aspect("equal", adjustable="box")

    lim = np.max(np.abs(traj.reshape(-1,2))) * 1.22
    ax_orb.set_xlim(-lim, lim); ax_orb.set_ylim(-lim, lim)

    _starfield(ax_orb)
    ax_orb.axhline(0, color=BORDER_CLR, lw=0.4, alpha=0.4, zorder=1)
    ax_orb.axvline(0, color=BORDER_CLR, lw=0.4, alpha=0.4, zorder=1)

    trails, trail_rgbs = zip(*[_make_trail_lc(ax_orb, BWC[i], trail_len)
                                for i in range(3)])

    glow1 = [ax_orb.plot([],[],'o',color=BGC[i],ms=20,alpha=0.10,
                          zorder=4,animated=True)[0] for i in range(3)]
    glow2 = [ax_orb.plot([],[],'o',color=BC[i], ms=9, alpha=0.22,
                          zorder=5,animated=True)[0] for i in range(3)]
    dots  = [ax_orb.plot([],[],'o',color=BC[i], ms=5.5,
                          zorder=6,animated=True)[0] for i in range(3)]
    lbls  = [ax_orb.text(0,0,f"  m{i+1}",color=BC[i],
                          fontsize=6.5,fontfamily=MONO,fontweight="bold",
                          zorder=7,animated=True) for i in range(3)]

    com_marker, = ax_orb.plot([],[],'x',color=COL_COM,ms=5,lw=0.8,
                               alpha=0.5,zorder=3,animated=True)

    ts_text = ax_orb.text(
        0.015, 0.015, "t = 0.000", transform=ax_orb.transAxes,
        color=VALUE_CLR, fontsize=7.5, fontfamily=MONO, animated=True,
        bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec=BORDER_CLR,
                  lw=0.5, alpha=0.80))
    pct_text = ax_orb.text(
        0.015, 0.062, "progress  0.0%", transform=ax_orb.transAxes,
        color=LABEL_CLR, fontsize=5.5, fontfamily=MONO, animated=True)

    # ── [0,0]  CONSERVATION SCORE ─────────────────────────────────────────────
    # A single number: "digits of precision kept".  More meaningful than a
    # raw error band that oscillates at integrator frequency.
    ax_sc = fig.add_subplot(gs_r[0, 0])
    _style_panel(ax_sc, title="CONSERVATION  –log₁₀(max|err|)")
    ax_sc.set_xlim(0, T)
    sc_min = max(0, score.min() - 0.5)
    sc_max = score.max() + 0.5
    ax_sc.set_ylim(sc_min, sc_max)
    # reference bands
    for lvl, lbl, col in [(7,"7 digits","#3dd9eb"),(10,"10 digits","#69ff94")]:
        if sc_min < lvl < sc_max:
            _hline_note(ax_sc, lvl, col, lbl)
    poly_sc = _fill_poly(ax_sc, "#3dd9eb", alpha=0.08)
    line_sc, = ax_sc.plot([],[], color="#3dd9eb", lw=1.0, animated=True)

    # live "digits" readout
    score_badge = ax_sc.text(
        0.97, 0.90, "", transform=ax_sc.transAxes,
        color="#3dd9eb", fontsize=7, fontfamily=MONO,
        ha="right", va="top", animated=True,
        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=BORDER_CLR,
                  lw=0.4, alpha=0.85))

    # ── [0,1]  PAIR DISTANCES ─────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs_r[0, 1])
    _style_panel(ax_r, title="PAIR DISTANCES", xlabel="t", ylabel="r")
    ax_r.set_xlim(0, T)
    rmax = max(r12.max(), r13.max(), r23.max()) * 1.12
    ax_r.set_ylim(0, rmax)
    # global close-approach threshold line
    _hline_note(ax_r, rmin_global*2, "#ffffff", "min×2", side="left")
    line_r12, = ax_r.plot([],[], color=COL_R12, lw=0.9, label="r₁₂",
                           animated=True)
    line_r13, = ax_r.plot([],[], color=COL_R13, lw=0.9, label="r₁₃",
                           animated=True)
    line_r23, = ax_r.plot([],[], color=COL_R23, lw=0.9, label="r₂₃",
                           animated=True)
    ax_r.legend(fontsize=5.5, framealpha=0, labelcolor="white",
                loc="upper right", handlelength=1.1,
                labelspacing=0.2, handletextpad=0.35)

    # ── [1,0]  PHASE PORTRAIT ─────────────────────────────────────────────────
    ax_ph = fig.add_subplot(gs_r[1, 0])
    _style_panel(ax_ph, title="PHASE PORTRAIT  x₁ vs ẋ₁",
                 xlabel="x₁", ylabel="ẋ₁")
    ax_ph.set_xlim(x1.min()*1.18, x1.max()*1.18)
    ax_ph.set_ylim(vx1.min()*1.18, vx1.max()*1.18)

    # ghost: plot full attractor at low alpha as a guide
    ax_ph.plot(x1, vx1, color=BC[0], lw=0.4, alpha=0.08, zorder=1)

    line_ph, = ax_ph.plot([],[], color=BC[0], lw=0.7, alpha=0.85,
                           animated=True)
    ph_dot,  = ax_ph.plot([],[],'o', color=BC[0], ms=4, zorder=6,
                           animated=True)

    # ── [1,1]  ANGULAR MOMENTUM DRIFT ────────────────────────────────────────
    ax_L = fig.add_subplot(gs_r[1, 1])
    _style_panel(ax_L, title="ANG. MOMENTUM DRIFT  ΔL/|L₀|", xlabel="t")
    ax_L.set_xlim(0, T)
    lpad = max(abs(dL.min()), abs(dL.max())) * 1.35 or 1e-10
    ax_L.set_ylim(-lpad, lpad)
    ax_L.axhline(0, color=COL_L, lw=0.4, alpha=0.3)
    poly_L  = _fill_poly(ax_L, COL_L, alpha=0.10)
    line_L, = ax_L.plot([],[], color=COL_L, lw=0.9, animated=True)
    # running max-drift badge
    drift_badge = ax_L.text(
        0.97, 0.90, "", transform=ax_L.transAxes,
        color=COL_L, fontsize=6, fontfamily=MONO,
        ha="right", va="top", animated=True,
        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=BORDER_CLR,
                  lw=0.4, alpha=0.85))

    # ── [2, :]  CENTRE-OF-MASS DRIFT ─────────────────────────────────────────
    # Spans both columns – a great sanity-check for equal-mass symmetry.
    ax_com = fig.add_subplot(gs_r[2, :])
    _style_panel(ax_com,
                 title="CENTRE-OF-MASS DRIFT  |com(t) – com(0)|  "
                       "(should stay ≈ 0 for isolated system)",
                 xlabel="t", ylabel="|Δcom|")
    ax_com.set_xlim(0, T)
    com_ymax = max(com_drift.max() * 1.3, 1e-10)
    ax_com.set_ylim(0, com_ymax)
    # colour the band red/green depending on drift magnitude
    _hline_note(ax_com, com_ymax * 0.1, "#ff6b6b", "10% threshold")
    poly_com  = _fill_poly(ax_com, COL_COM, alpha=0.12)
    line_com, = ax_com.plot([],[], color=COL_COM, lw=1.0, animated=True)

    plt.subplots_adjust(left=0.04, right=0.97, top=0.935, bottom=0.07)

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(frame):
        idx = max(frame, 2)

        # ORBIT: trails
        for i in range(3):
            start = max(0, idx - trail_len)
            tx = traj[start:idx, i, 0]
            ty = traj[start:idx, i, 1]
            n  = len(tx)
            if n < 2:
                trails[i].set_segments(np.full((trail_len,2,2), np.nan))
            else:
                pts  = np.stack([tx,ty],axis=1).reshape(-1,1,2)
                segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
                pad  = trail_len - len(segs)
                if pad > 0:
                    segs = np.vstack([np.full((pad,2,2),np.nan), segs])
                trails[i].set_segments(segs)
                rgb = trail_rgbs[i]
                alphas = np.power(np.linspace(0,1,trail_len), 1.8)
                trails[i].set_colors([(rgb[0],rgb[1],rgb[2],a) for a in alphas])

            px, py = traj[idx-1,i,0], traj[idx-1,i,1]
            glow1[i].set_data([px],[py])
            glow2[i].set_data([px],[py])
            dots[i].set_data([px],[py])
            lbls[i].set_position((px,py))

        com_marker.set_data([com[idx-1,0]], [com[idx-1,1]])
        ts_text.set_text(f"t = {t[idx-1]:.3f}")
        pct_text.set_text(f"progress  {(idx-1)/(Nfr-1)*100:.1f}%")

        # CONSERVATION SCORE
        line_sc.set_data(t[:idx], score[:idx])
        poly_sc.set_xy(_poly_xy(t[:idx], score[:idx]))
        score_badge.set_text(f"{score[idx-1]:.1f} digits")

        # PAIR DISTANCES
        line_r12.set_data(t[:idx], r12[:idx])
        line_r13.set_data(t[:idx], r13[:idx])
        line_r23.set_data(t[:idx], r23[:idx])

        # PHASE PORTRAIT
        line_ph.set_data(x1[:idx], vx1[:idx])
        ph_dot.set_data([x1[idx-1]], [vx1[idx-1]])

        # MOMENTUM DRIFT
        line_L.set_data(t[:idx], dL[:idx])
        poly_L.set_xy(_poly_xy(t[:idx], dL[:idx]))
        max_drift = np.max(np.abs(dL[:idx]))
        drift_badge.set_text(f"max |ΔL| = {max_drift:.2e}")

        # COM DRIFT
        line_com.set_data(t[:idx], com_drift[:idx])
        poly_com.set_xy(_poly_xy(t[:idx], com_drift[:idx]))

        return (list(trails) + glow1 + glow2 + dots + lbls
                + [com_marker, ts_text, pct_text,
                   poly_sc,  line_sc,  score_badge,
                   line_r12, line_r13, line_r23,
                   line_ph,  ph_dot,
                   poly_L,   line_L,   drift_badge,
                   poly_com, line_com])

    ani = animation.FuncAnimation(
        fig, update,
        frames=range(2, Nfr, stride),
        interval=16,
        blit=True,
        cache_frame_data=False,
    )

    if save_path:
        writer = animation.FFMpegWriter(fps=30, bitrate=2400,
                                        extra_args=["-crf","17"])
        ani.save(save_path, writer=writer, dpi=110,
                 savefig_kwargs={"facecolor": BG})
        print(f"Saved  →  {save_path}")
    else:
        plt.show()

    return ani