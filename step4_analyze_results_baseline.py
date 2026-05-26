import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    ANALYSIS_DIR,
    DIMENSIONS,
    MODEL_SLUG,
    MODEL_TO_EVALUATE,
    REPO_ROOT,
    baseline_results_path,
)

INPUT_JSON = baseline_results_path()
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ANALYSIS_DIR / f"{MODEL_SLUG}_top100_baseline_{_TIMESTAMP}"

DIM_LABELS = {
    "X1_discovery":    "X1 Discovery",
    "X2_prediction":   "X2 Prediction",
    "X3_diagnosis":    "X3 Diagnosis",
    "X4_intervention": "X4 Intervention",
}
DIM_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Data loading & flattening ──────────────────────────────────────────────────

def load_flat_records(path: Path) -> pd.DataFrame:
    """
    Flatten results JSON into one row per question with columns:
      image_id, cluster, dimension, DCR, r1, r2, r3, r1p, r2p, r3p
    """
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for img in dataset:
        image_id = img["image_id"]
        cluster  = img.get("primary_cluster", "other")
        for dim in DIMENSIONS:
            for q in img["questions"].get(dim, []):
                er = q.get("evaluation_results")
                if er is None:
                    continue
                raw = er["raw_scores"]
                cas = er["cascaded_scores"]
                rows.append({
                    "image_id":  image_id,
                    "cluster":   cluster,
                    "dimension": dim,
                    "DCR":       er["DCR"],
                    "r1":        raw["r1"],
                    "r2":        raw["r2"],
                    "r3":        raw["r3"],
                    "r1p":       cas["r1_prime"],
                    "r2p":       cas["r2_prime"],
                    "r3p":       cas["r3_prime"],
                })
    return pd.DataFrame(rows)


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    sep = "=" * 65

    print(sep)
    print(f"  ICCR Benchmark Results (Top-100 baseline)  |  model: {MODEL_TO_EVALUATE}")
    print(sep)

    n = len(df)
    print(f"\n  Total questions scored : {n}")
    print(f"  Images evaluated       : {df['image_id'].nunique()}")

    # ── Overall ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  OVERALL")
    print(f"{'─' * 65}")
    print(f"  Average DCR          : {df['DCR'].mean():.3f}")
    print(f"  Median  DCR          : {df['DCR'].median():.3f}")
    print(f"  Std     DCR          : {df['DCR'].std():.3f}")
    print(f"  Full-credit (DCR=1)  : {(df['DCR']==1.0).mean():.1%}  "
          f"({(df['DCR']==1.0).sum()}/{n})")
    print(f"  Zero-credit (DCR=0)  : {(df['DCR']==0.0).mean():.1%}  "
          f"({(df['DCR']==0.0).sum()}/{n})")
    print(f"  L1 pass rate         : {df['r1p'].mean():.1%}  "
          f"(entity perception gate)")
    print(f"  L2 pass rate         : {df['r2p'].mean():.1%}  "
          f"(interaction grounding)")
    print(f"  L3 pass rate         : {df['r3p'].mean():.1%}  "
          f"(causal resolution)")

    # ── Per-dimension ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  {'Dimension':<22} {'Avg DCR':>8}  {'L1%':>6}  {'L2%':>6}  {'L3%':>6}  {'n':>4}")
    print(f"{'─' * 65}")
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            continue
        print(
            f"  {DIM_LABELS[dim]:<22} "
            f"{sub['DCR'].mean():>8.3f}  "
            f"{sub['r1p'].mean():>6.1%}  "
            f"{sub['r2p'].mean():>6.1%}  "
            f"{sub['r3p'].mean():>6.1%}  "
            f"{len(sub):>4}"
        )

    # ── Per-cluster ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  {'Cluster':<18} {'Avg DCR':>8}  {'n_q':>4}  {'n_img':>5}")
    print(f"{'─' * 65}")
    cluster_stats = (
        df.groupby("cluster")
        .agg(avg_dcr=("DCR", "mean"), n_q=("DCR", "count"),
             n_img=("image_id", "nunique"))
        .sort_values("avg_dcr", ascending=False)
    )
    for cluster, row in cluster_stats.iterrows():
        print(f"  {cluster:<18} {row['avg_dcr']:>8.3f}  {int(row['n_q']):>4}  {int(row['n_img']):>5}")

    # ── DCR distribution ──────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  DCR DISTRIBUTION")
    print(f"{'─' * 65}")
    for val, label in [(0.0, "0.000 (complete fail)"),
                       (0.333, "0.333 (entity only)"),
                       (0.667, "0.667 (entity + interaction)"),
                       (1.000, "1.000 (full credit)")]:
        cnt = (df["DCR"].round(3) == round(val, 3)).sum()
        bar = "█" * int(cnt / n * 40)
        print(f"  {label:<30}  {bar:<40}  {cnt:>3} ({cnt/n:.1%})")

    print(f"\n{sep}")


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, suffix: str) -> None:
    path = OUT_DIR / f"{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# ── Figure 1: Average DCR per dimension ───────────────────────────────────────

def fig_dcr_per_dimension(df: pd.DataFrame) -> None:
    means = [df[df["dimension"] == d]["DCR"].mean() for d in DIMENSIONS]
    stds  = [df[df["dimension"] == d]["DCR"].std()  for d in DIMENSIONS]
    labels = [DIM_LABELS[d] for d in DIMENSIONS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, color=DIM_COLORS, width=0.55,
                  yerr=stds, capsize=5, error_kw={"linewidth": 1.2})

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Average DCR", fontsize=11)
    ax.set_title(f"Average DCR per Dimension  ({MODEL_TO_EVALUATE})", fontsize=13, pad=12)
    ax.axhline(df["DCR"].mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Overall avg = {df['DCR'].mean():.3f}")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig1_dcr_per_dimension")


# ── Figure 2: L1 / L2 / L3 pass rates per dimension ──────────────────────────

def fig_level_pass_rates(df: pd.DataFrame) -> None:
    labels = [DIM_LABELS[d] for d in DIMENSIONS]
    l1 = [df[df["dimension"] == d]["r1p"].mean() for d in DIMENSIONS]
    l2 = [df[df["dimension"] == d]["r2p"].mean() for d in DIMENSIONS]
    l3 = [df[df["dimension"] == d]["r3p"].mean() for d in DIMENSIONS]

    x   = np.arange(len(DIMENSIONS))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - w, l1, w, label="L1 Perception",   color="#4C72B0")
    b2 = ax.bar(x,     l2, w, label="L2 Interaction",  color="#55A868")
    b3 = ax.bar(x + w, l3, w, label="L3 Causal",       color="#C44E52")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Pass Rate (cascaded)", fontsize=11)
    ax.set_title(f"L1 / L2 / L3 Pass Rates per Dimension  ({MODEL_TO_EVALUATE})",
                 fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig2_level_pass_rates")


# ── Figure 3: DCR score distribution (histogram) ─────────────────────────────

def fig_dcr_distribution(df: pd.DataFrame) -> None:
    buckets = {0.0: 0, 0.333: 0, 0.667: 0, 1.0: 0}
    for v in df["DCR"]:
        key = min(buckets.keys(), key=lambda k: abs(k - v))
        buckets[key] += 1

    labels = ["0.000\n(Complete fail)", "0.333\n(Entity only)",
              "0.667\n(Entity+Interact)", "1.000\n(Full credit)"]
    counts = list(buckets.values())
    total  = sum(counts)
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, counts, color=colors, width=0.55)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cnt}\n({cnt/total:.1%})", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(f"DCR Score Distribution  ({MODEL_TO_EVALUATE})", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig3_dcr_distribution")


# ── Figure 4: Average DCR per action cluster ──────────────────────────────────

def fig_dcr_per_cluster(df: pd.DataFrame) -> None:
    cluster_dcr = (
        df.groupby("cluster")["DCR"]
        .mean()
        .sort_values(ascending=True)
    )

    overall = df["DCR"].mean()
    colors  = ["#C44E52" if v < overall else "#4C72B0" for v in cluster_dcr.values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(cluster_dcr) * 0.45)))
    bars = ax.barh(cluster_dcr.index, cluster_dcr.values, color=colors, height=0.6)

    for bar, val in zip(bars, cluster_dcr.values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.axvline(overall, color="gray", linestyle="--", linewidth=1,
               label=f"Overall avg = {overall:.3f}")
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Average DCR", fontsize=11)
    ax.set_title(f"Average DCR per Action Cluster  ({MODEL_TO_EVALUATE})", fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig4_dcr_per_cluster")


# ── Figure 5: Per-image DCR heatmap ───────────────────────────────────────────

def fig_heatmap(df: pd.DataFrame) -> None:
    # Pivot: rows = images sorted by overall DCR desc, columns = dimensions
    pivot = df.pivot_table(
        index="image_id", columns="dimension", values="DCR", aggfunc="mean"
    )[DIMENSIONS]
    pivot = pivot.reindex(
        pivot.mean(axis=1).sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(7, max(5, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=7)
    ax.set_title(f"Per-image Mean DCR Heatmap  ({MODEL_TO_EVALUATE})", fontsize=12, pad=10)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(DIMENSIONS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean DCR", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig5_heatmap")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_JSON.exists():
        print(f"[Fatal] {INPUT_JSON.name} not found. Run step3_judge_scoring_baseline.py first.")
        return

    # Create timestamped output folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder (repo-relative): {OUT_DIR.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}")

    print(f"Loading {INPUT_JSON.name} …")
    df = load_flat_records(INPUT_JSON)
    print(f"  {len(df)} question records loaded.\n")

    # ── Console report + save to text file ──────────────────────────────────
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_report(df)
    sys.stdout = old_stdout
    report_text = buf.getvalue()

    print(report_text, end="")          # still show in terminal

    report_path = OUT_DIR / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved: report.txt\n")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("Generating figures …")
    fig_dcr_per_dimension(df)
    fig_level_pass_rates(df)
    fig_dcr_distribution(df)
    fig_dcr_per_cluster(df)
    fig_heatmap(df)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
