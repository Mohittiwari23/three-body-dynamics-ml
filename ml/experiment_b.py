"""
ml/experiment_b.py
==================
Experiment B — Orbit Type Classification (FIXED)
=================================================

Research question
-----------------
Do normalised physics features (ε, h, μ, q, r₀) generalise better than
raw features (E₀, L₀, μ, r₀) when classifying orbit type across mass ratios?

How does this compare to using static orbital elements (Pinheiro 2025)?

Fixes applied vs. original
---------------------------
1. Quality filter: dL_max ≤ 1e-6 (not dE_max) — retains all orbit classes
   including parabolic (which the old filter destroyed).
2. MEGNO_clean instead of raw MEGNO — clips unphysical negatives.
3. Parabolic class included in training — sufficient samples now exist.
4. Classification report shows per-class breakdown including parabolic/circular.
5. Confusion matrix uses class names correctly via LabelEncoder.

Why classification is still valid despite orbit class being e-derived
---------------------------------------------------------------------
The classification task is: given physics features, predict orbit type.
The boundary at e=1 is sharp. The uncertainty in classification comes
from the model needing to generalise the epsilon sign threshold (and
sub-threshold distinctions within bound orbits) across mass ratios.
The interesting result is which feature representation makes this
generalisation best — raw vs normalised vs orbital elements.

Experiments
-----------
  B1  XGBoost on PHYSICS_NORM (ε, h, μ, q, r₀) — main model
  B2  XGBoost on PHYSICS_RAW (E₀, L₀, μ, r₀)   — ablation
  B3  XGBoost on ORBITAL_ELEM (e, p, a, r_min, r_ratio) — Pinheiro baseline
  B4  XGBoost on ALL_PHYSICS (norm + dynamics)   — full novel feature set

Usage
-----
    python ml/experiment_b.py
    python ml/experiment_b.py --data data/dataset/metadata.csv
"""

from __future__ import annotations

import sys, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize, LabelEncoder
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import (
    PHYSICS_NORM, PHYSICS_RAW, ORBITAL_ELEM, ALL_PHYSICS, DYNAMICS,
    extract, extract_classification_target,
    train_test_generalisation_split, load_dataset,
    CLASS_NAMES,
)
warnings.filterwarnings("ignore")

DEFAULT_CSV = Path("data/dataset/metadata.csv")
OUTPUT_DIR  = Path("outputs/exp_b")
SEED        = 42

XGB_CLF_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = SEED,
    n_jobs           = -1,
    verbosity        = 0,
    eval_metric      = "mlogloss",
)

STYLE = dict(bg="#080810", ax="#0d0d1a", grid="#1e1e2e", text="#c4c4d4",
             lab="#7777aa", a="#38bdf8", b="#fb923c", c="#86efac",
             d="#f87171", e_col="#c084fc")
CLASS_COLORS = ["#38bdf8", "#86efac", "#fb923c", "#f87171"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _evaluate(y_true, y_pred, y_prob, label, le):
    acc = accuracy_score(y_true, y_pred)
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            n_cls = y_prob.shape[1]
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            if y_bin.shape[1] > 1:
                auc = roc_auc_score(y_bin, y_prob,
                                    multi_class="ovr", average="macro")
        except Exception:
            pass
    print(f"    {label:<45}  Acc={acc:.4f}   AUC={'N/A' if np.isnan(auc) else f'{auc:.4f}'}")
    return {"label": label, "Accuracy": acc, "AUC": auc}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(STYLE["ax"])
    ax.set_title(title, color=STYLE["text"], fontsize=9, pad=4,
                 fontfamily="monospace")
    ax.set_xlabel(xlabel, color=STYLE["lab"], fontsize=8)
    ax.set_ylabel(ylabel, color=STYLE["lab"], fontsize=8)
    ax.tick_params(colors="#666688", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor(STYLE["grid"])
    ax.grid(True, color=STYLE["grid"], lw=0.4, alpha=0.7)


def plot_confusion_matrix(y_true, y_pred, class_names, title, path):
    n   = len(class_names)
    cm  = confusion_matrix(y_true, y_pred)
    if cm.shape[0] != n:
        # pad to full n×n if some classes absent
        full_cm = np.zeros((n, n), dtype=int)
        unique = sorted(set(y_true) | set(y_pred))
        for i, ri in enumerate(unique):
            for j, cj in enumerate(unique):
                full_cm[ri, cj] = cm[i, j]
        cm = full_cm

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["ax"])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=30,
                                                  ha="right", fontsize=8,
                                                  color=STYLE["lab"])
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=8,
                                                  color=STYLE["lab"])
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9,
                    color="white" if cm[i, j] > thresh else STYLE["text"])
    ax.set_title(title, color=STYLE["text"], fontsize=9,
                 fontfamily="monospace", pad=5)
    ax.set_xlabel("Predicted", color=STYLE["lab"], fontsize=8)
    ax.set_ylabel("True",      color=STYLE["lab"], fontsize=8)
    ax.tick_params(colors="#666688")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_generalisation_curve(gen_results, title, path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=STYLE["bg"])
    _style(ax, title, "Mass ratio q  (training: q ≤ 0.15)", "Accuracy")
    colors = [STYLE["a"], STYLE["b"], STYLE["c"], STYLE["d"], STYLE["e_col"]]
    for (name, pts), col in zip(gen_results.items(), colors):
        if not pts: continue
        qs   = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        ax.plot(qs, vals, "o-", color=col, lw=1.5, ms=5, label=name)
    ax.axvline(0.15, color="#fde68a", ls="--", lw=0.8, alpha=0.7,
               label="train boundary")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor="#13131f", labelcolor="#bbbbcc")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names, title, path):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, max(3, len(feature_names)*0.4)),
                           facecolor=STYLE["bg"])
    ax.barh([feature_names[i] for i in idx], imp[idx],
            color=STYLE["a"], alpha=0.85)
    _style(ax, title, "Importance (gain)", "Feature")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_accuracy_vs_q(feats, model, le, df, title, path):
    """Per-class accuracy as a function of q bin."""
    q_bins = [(0.001,0.05),(0.05,0.15),(0.15,0.25),(0.25,0.35),
              (0.35,0.55),(0.55,0.80),(0.80,1.00)]
    present_classes = le.classes_
    n_cls = len(present_classes)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, n_cls, figsize=(4*n_cls, 3.5),
                              facecolor=STYLE["bg"])
    if n_cls == 1: axes = [axes]

    for ci, (ax, cls_idx) in enumerate(zip(axes, present_classes)):
        cls_name = CLASS_NAMES[cls_idx]
        qs_mid, accs = [], []
        for lo, hi in q_bins:
            mask = (df["q"] > lo) & (df["q"] <= hi) & \
                   (df["orbit_class"] == cls_idx)
            sub = df[mask]
            if len(sub) < 3: continue
            X_sub = sub[feats].values
            y_enc = le.transform(np.clip(sub["orbit_class"].values,
                                         le.classes_.min(),
                                         le.classes_.max()))
            accs.append(accuracy_score(y_enc, model.predict(X_sub)))
            qs_mid.append((lo + hi) / 2)
        col = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        ax.plot(qs_mid, accs, "o-", color=col, lw=1.5, ms=5)
        ax.axvline(0.15, color="#fde68a", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylim(0, 1.05)
        _style(ax, cls_name.capitalize(), "q", "Per-class accuracy")

    fig.suptitle(title, color=STYLE["text"], fontsize=9,
                 fontfamily="monospace")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(csv_path: Path = DEFAULT_CSV) -> None:
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    print("\n" + "═"*60)
    print("  EXPERIMENT B — Orbit Type Classification")
    print("  Quality filter: dL_max ≤ 1e-6  (all orbit types retained)")
    print("  Target: orbit_class ∈ {circular, elliptical, parabolic, hyperbolic}")
    print("═"*60)

    df = load_dataset(csv_path,
                      apply_quality_filter=True,
                      dL_max_threshold=1e-6)

    print("\n  Class distribution:")
    for name in CLASS_NAMES:
        n = (df["orbit_name"] == name).sum()
        if n > 0:
            print(f"    {name:<12}: {n:4d}  ({100*n/len(df):.1f}%)")

    # ── Splits ─────────────────────────────────────────────────────────────
    df_tr, df_id, df_ood = train_test_generalisation_split(df, seed=SEED)
    print(f"\n  Train: {len(df_tr):4d}  In-dist: {len(df_id):4d}  OOD: {len(df_ood):4d}")

    # LabelEncoder fitted on training classes
    le = LabelEncoder()
    le.fit(df_tr["orbit_class"].values)
    n_cls = len(le.classes_)
    print(f"  Classes in training: {[CLASS_NAMES[c] for c in le.classes_]}")

    def _enc(df_): return le.transform(np.clip(
        df_["orbit_class"].values, le.classes_.min(), le.classes_.max()))

    y_tr  = _enc(df_tr)
    y_id  = _enc(df_id)
    y_ood = _enc(df_ood)

    all_results  = []
    gen_results  = {}
    trained_models = {}

    # ── Train + evaluate helper ────────────────────────────────────────────
    def _run_xgb(name, feats, df_tr_, df_id_, df_ood_):
        print(f"\n── {name} ──")
        X_tr  = extract(df_tr_,  feats)
        X_id  = extract(df_id_,  feats)
        X_ood = extract(df_ood_, feats)
        y_tr_ = _enc(df_tr_)
        y_id_ = _enc(df_id_)
        y_ood_= _enc(df_ood_)

        clf = xgb.XGBClassifier(
            **XGB_CLF_PARAMS,
            objective="multi:softprob" if n_cls > 2 else "binary:logistic",
            num_class=n_cls if n_cls > 2 else None,
        )
        clf.fit(X_tr, y_tr_)

        r1 = _evaluate(y_id_,  clf.predict(X_id),  clf.predict_proba(X_id),
                       f"{name} | in-dist", le)
        r2 = _evaluate(y_ood_, clf.predict(X_ood), clf.predict_proba(X_ood),
                       f"{name} | OOD",     le)
        all_results.extend([r1, r2])
        return clf

    # ── B1: PHYSICS_NORM ───────────────────────────────────────────────────
    m_b1 = _run_xgb("B1 PHYSICS_NORM", PHYSICS_NORM, df_tr, df_id, df_ood)
    trained_models["B1 PHYSICS_NORM"] = (PHYSICS_NORM, m_b1)
    gen_results["B1 PHYSICS_NORM"] = []
    joblib.dump(m_b1, out / "models" / "model_b1_physics_norm.joblib")

    # ── B2: PHYSICS_RAW ────────────────────────────────────────────────────
    m_b2 = _run_xgb("B2 PHYSICS_RAW (ablation)", PHYSICS_RAW,
                     df_tr, df_id, df_ood)
    trained_models["B2 PHYSICS_RAW"] = (PHYSICS_RAW, m_b2)
    gen_results["B2 PHYSICS_RAW"] = []

    # ── B3: ORBITAL_ELEM — Pinheiro 2025 baseline ─────────────────────────
    # Orbital elements undefined for unbound orbits (a, r_ratio = NaN)
    # Keep bound orbits only for this comparison
    df_tr3  = df_tr.dropna(subset=ORBITAL_ELEM)
    df_id3  = df_id.dropna(subset=ORBITAL_ELEM)
    df_ood3 = df_ood.dropna(subset=ORBITAL_ELEM)
    print(f"\n── B3 ORBITAL_ELEM (Pinheiro baseline — bound orbits only) ──")
    print(f"  Samples: tr={len(df_tr3)}, id={len(df_id3)}, ood={len(df_ood3)}")
    if len(df_tr3) >= 20:
        le3 = LabelEncoder(); le3.fit(df_tr3["orbit_class"].values)
        def _enc3(df_): return le3.transform(np.clip(
            df_["orbit_class"].values, le3.classes_.min(), le3.classes_.max()))
        n3 = len(le3.classes_)
        clf3 = xgb.XGBClassifier(
            **XGB_CLF_PARAMS,
            objective="multi:softprob" if n3 > 2 else "binary:logistic",
            num_class=n3 if n3 > 2 else None,
        )
        clf3.fit(extract(df_tr3, ORBITAL_ELEM), _enc3(df_tr3))
        r1 = _evaluate(_enc3(df_id3), clf3.predict(extract(df_id3, ORBITAL_ELEM)),
                       clf3.predict_proba(extract(df_id3, ORBITAL_ELEM)),
                       "B3 ORBITAL_ELEM | in-dist", le3)
        r2 = _evaluate(_enc3(df_ood3), clf3.predict(extract(df_ood3, ORBITAL_ELEM)),
                       clf3.predict_proba(extract(df_ood3, ORBITAL_ELEM)),
                       "B3 ORBITAL_ELEM | OOD", le3)
        all_results.extend([r1, r2])
        trained_models["B3 ORBITAL_ELEM"] = (ORBITAL_ELEM, clf3)
        gen_results["B3 ORBITAL_ELEM"] = []

    # ── B4: ALL_PHYSICS ────────────────────────────────────────────────────
    m_b4 = _run_xgb("B4 ALL_PHYSICS (norm + dynamics)", ALL_PHYSICS,
                     df_tr, df_id, df_ood)
    trained_models["B4 ALL_PHYSICS"] = (ALL_PHYSICS, m_b4)
    gen_results["B4 ALL_PHYSICS"] = []
    joblib.dump(m_b4, out / "models" / "model_b4_all_physics.joblib")

    # ── Generalisation curve ───────────────────────────────────────────────
    print("\n── Mass ratio generalisation test ──")
    q_bins = [(0.001,0.05),(0.05,0.15),(0.15,0.25),(0.25,0.35),
              (0.35,0.55),(0.55,0.80),(0.80,1.00)]
    for lo, hi in q_bins:
        sub = df[(df["q"] > lo) & (df["q"] <= hi)]
        if len(sub) < 5: continue
        q_mid = (lo + hi) / 2
        y_sub = _enc(sub)
        for name, (feats, model) in trained_models.items():
            if name not in gen_results: continue
            sub_ = sub.dropna(subset=feats)
            if len(sub_) < 3: continue
            y_sub_ = _enc(sub_)
            acc = accuracy_score(y_sub_, model.predict(extract(sub_, feats)))
            gen_results[name].append((q_mid, acc))
            print(f"  {name:<25}  q∈({lo:.3f},{hi:.3f}]  "
                  f"n={len(sub_):4d}  acc={acc:.4f}")

    # ── Classification report (B1, in-dist) ───────────────────────────────
    print("\n── B1 Full Classification Report (in-distribution) ──")
    y_pred_b1 = m_b1.predict(extract(df_id, PHYSICS_NORM))
    present   = sorted(set(y_id) | set(y_pred_b1))
    names_present = [CLASS_NAMES[le.classes_[i]] for i in present]
    print(classification_report(y_id, y_pred_b1, labels=present,
                                 target_names=names_present, zero_division=0))

    # ── Save results ───────────────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(out / "results.csv", index=False)
    print(f"\n  Saved: {out / 'results.csv'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    # Confusion matrices (B1)
    full_names = [CLASS_NAMES[c] for c in range(max(le.classes_)+1)
                  if c in le.classes_]
    plot_confusion_matrix(
        y_id,  y_pred_b1, full_names,
        "B1 Confusion Matrix — in-distribution",
        out / "plots" / "confusion_b1_id.png",
    )
    plot_confusion_matrix(
        y_ood, m_b1.predict(extract(df_ood, PHYSICS_NORM)), full_names,
        "B1 Confusion Matrix — OOD (q ≥ 0.35)",
        out / "plots" / "confusion_b1_ood.png",
    )

    # Generalisation curves
    plot_generalisation_curve(
        gen_results,
        "Experiment B — Accuracy vs mass ratio q\n"
        "Key: B1 (normalised) vs B2 (raw) vs B3 (Pinheiro) vs B4 (all)",
        out / "plots" / "generalisation_accuracy.png",
    )

    # Feature importance
    plot_feature_importance(m_b1, PHYSICS_NORM,
        "B1 Feature Importance — PHYSICS_NORM",
        out / "plots" / "feature_importance_b1.png")
    plot_feature_importance(m_b4, ALL_PHYSICS,
        "B4 Feature Importance — ALL_PHYSICS (novel feature set)",
        out / "plots" / "feature_importance_b4.png")

    # Per-class accuracy vs q
    plot_per_class_accuracy_vs_q(
        PHYSICS_NORM, m_b1, le, df,
        "B1 — Per-class accuracy vs mass ratio",
        out / "plots" / "per_class_accuracy_vs_q.png",
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EXPERIMENT B COMPLETE")
    print("─"*60)
    _print_table(all_results)
    _interpret(all_results)
    print(f"\n  Output: {out}/")


def _print_table(results):
    print("\n  ┌────────────────────────────────────────────────┬────────┬────────┐")
    print(  "  │ Model                                          │  Acc.  │  AUC   │")
    print(  "  ├────────────────────────────────────────────────┼────────┼────────┤")
    for r in results:
        lab  = r["label"][:48]
        auc  = "  N/A " if np.isnan(r["AUC"]) else f"{r['AUC']:.4f}"
        print(f"  │ {lab:<48} │ {r['Accuracy']:.4f} │ {auc} │")
    print(  "  └────────────────────────────────────────────────┴────────┴────────┘")


def _interpret(results):
    def _get(label_substr):
        return next((r for r in results if label_substr in r["label"]), None)

    b1_ood = _get("B1 PHYSICS_NORM | OOD")
    b2_ood = _get("B2 PHYSICS_RAW (ablation) | OOD")
    b3_ood = _get("B3 ORBITAL_ELEM | OOD")

    print("\n  KEY FINDINGS:")
    if b1_ood and b2_ood:
        gap = b1_ood["Accuracy"] - b2_ood["Accuracy"]
        direction = "normalised > raw" if gap > 0 else "raw ≥ normalised"
        print(f"  B1 vs B2 OOD gap: {gap:+.4f}  ({direction})")
    if b1_ood and b3_ood:
        gap = b1_ood["Accuracy"] - b3_ood["Accuracy"]
        direction = "PHYSICS_NORM > ORBITAL_ELEM" if gap > 0 else "ORBITAL_ELEM ≥ PHYSICS_NORM"
        print(f"  B1 vs B3 OOD gap: {gap:+.4f}  ({direction})")
    print("  (Positive gaps support the paper's claim for normalised physics features)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_CSV)
    args = p.parse_args()
    if not args.data.exists():
        print(f"ERROR: {args.data} not found. Run: python ml/dataset_generator.py")
        sys.exit(1)
    run(csv_path=args.data)