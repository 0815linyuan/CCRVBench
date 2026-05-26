from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from config import (
    ANALYSIS_DIR,
    MODEL_SLUG,
    MODEL_TO_EVALUATE,
    MODEL_OUTPUTS_DIR,
    REPO_ROOT,
    baseline_results_path,
    existing_path,
)

INPUT_JSON = MODEL_OUTPUTS_DIR / f"results_{MODEL_SLUG}_main_set_top100.json"
BASELINE_JSON = existing_path(baseline_results_path(), MODEL_OUTPUTS_DIR / f"results_{MODEL_SLUG}.json")

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ANALYSIS_DIR / f"{MODEL_SLUG}_main_set_top100_{_TIMESTAMP}"

DIM_LABELS = {
    "X1_discovery": "X1 Discovery",
    "X2_prediction": "X2 Prediction",
    "X3_diagnosis": "X3 Diagnosis",
    "X4_intervention": "X4 Intervention",
}


def constraint_major(constraint_id: str) -> str:
    if not constraint_id:
        return "unknown"
    return constraint_id.split(".")[0]


def load_baseline_lookup(path: Path) -> dict[tuple[int, str, int], float]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    lookup = {}
    for img in dataset:
        image_id = img.get("image_id")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                er = q.get("evaluation_results")
                if er and "DCR" in er:
                    lookup[(image_id, dim, q_idx)] = er["DCR"]
    return lookup


def load_flat_records(path: Path, baseline_lookup: dict[tuple[int, str, int], float]) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for img in dataset:
        image_id = img.get("image_id")
        cluster = img.get("primary_cluster", "other")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    er = main.get("main_set_evaluation_results")
                    if not er:
                        continue

                    raw = er.get("raw_scores", {})
                    cas = er.get("cascaded_scores", {})
                    constraint_id = main.get("constraint_id", "")
                    baseline_dcr = baseline_lookup.get((image_id, dim, q_idx))

                    rows.append({
                        "image_id": image_id,
                        "cluster": cluster,
                        "dimension": dim,
                        "dimension_label": DIM_LABELS.get(dim, dim),
                        "q_idx": q_idx,
                        "variant_idx": var_idx,
                        "constraint_id": constraint_id,
                        "constraint_group": constraint_major(constraint_id),
                        "constraint_name": main.get("constraint_name", ""),
                        "DCR": er["DCR"],
                        "CSR": er["CSR"],
                        "effective_DCR": er["effective_DCR"],
                        "baseline_DCR": baseline_dcr,
                        "sample_CDI": None if baseline_dcr is None else baseline_dcr - er["DCR"],
                        "sample_effective_CDI": None if baseline_dcr is None else baseline_dcr - er["effective_DCR"],
                        "r1": raw.get("r1"),
                        "r2": raw.get("r2"),
                        "r3": raw.get("r3"),
                        "r1p": cas.get("r1_prime"),
                        "r2p": cas.get("r2_prime"),
                        "r3p": cas.get("r3_prime"),
                    })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        compliant = sub[sub["CSR"] == 1]
        row.update({
            "n": len(sub),
            "DCR": sub["DCR"].mean(),
            "CSR": sub["CSR"].mean(),
            "effective_DCR": sub["effective_DCR"].mean(),
            "compliant_n": len(compliant),
            "compliant_DCR": compliant["DCR"].mean() if len(compliant) else float("nan"),
            "baseline_DCR": sub["baseline_DCR"].mean(),
            "r1p": sub["r1p"].mean(),
            "r2p": sub["r2p"].mean(),
            "r3p": sub["r3p"].mean(),
        })
        if pd.notna(row["baseline_DCR"]):
            row["CDI"] = row["baseline_DCR"] - row["DCR"]
            row["compliant_CDI"] = row["baseline_DCR"] - row["compliant_DCR"]
            row["effective_CDI"] = row["baseline_DCR"] - row["effective_DCR"]
        else:
            row["CDI"] = float("nan")
            row["compliant_CDI"] = float("nan")
            row["effective_CDI"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "  (empty)\n"
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return view.to_string(index=False) + "\n"


def build_report(df: pd.DataFrame, baseline_available: bool) -> str:
    sep = "=" * 78
    overall = aggregate(df, [])
    by_x = aggregate(df, ["dimension", "dimension_label"]).sort_values("dimension")
    by_y = aggregate(df, ["constraint_group", "constraint_id"]).sort_values(
        ["constraint_group", "constraint_id"]
    )
    by_xy = aggregate(df, ["dimension", "constraint_id"]).sort_values(["dimension", "constraint_id"])

    lines = [
        sep,
        f"  ICCR Constrained Main-Set (Top-100, full X×Y) | model: {MODEL_TO_EVALUATE}",
        sep,
        "",
        f"  Total scored constrained variants : {len(df)}",
        f"  Images evaluated                  : {df['image_id'].nunique()}",
        f"  Baseline for CDI                  : {'available' if baseline_available else 'not found'}",
        "",
        "OVERALL",
        format_table(
            overall,
            ["n", "DCR", "CSR", "effective_DCR", "r1p", "r2p", "r3p", "compliant_n", "compliant_DCR",
             "baseline_DCR", "CDI", "compliant_CDI", "effective_CDI"],
        ),
        "",
        "BY X DIMENSION",
        format_table(
            by_x,
            ["dimension_label", "n", "DCR", "CSR", "r1p", "r2p", "r3p", "effective_DCR", "compliant_DCR",
             "baseline_DCR", "effective_CDI"],
        ),
        "",
        "BY Y CONSTRAINT",
        format_table(
            by_y,
            ["constraint_id", "n", "DCR", "CSR", "r1p", "r2p", "r3p", "effective_DCR", "compliant_DCR",
             "baseline_DCR", "effective_CDI"],
        ),
        "",
        "BY X x Y",
        format_table(
            by_xy,
            ["dimension", "constraint_id", "n", "DCR", "CSR", "r1p", "r2p", "r3p", "effective_DCR",
             "compliant_DCR", "effective_CDI"],
        ),
        sep,
    ]
    return "\n".join(lines)


def save_bar_metric(df: pd.DataFrame, x_col: str, metric: str, filename: str, title: str) -> None:
    summary = df.groupby(x_col)[metric].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(summary.index.astype(str), summary.values, color="#4C72B0", width=0.6)
    for bar, val in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric)
    ax.set_title(title)
    if metric == "CSR":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def save_xy_heatmap(df: pd.DataFrame, metric: str, filename: str) -> None:
    pivot = df.pivot_table(index="dimension", columns="constraint_id", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([DIM_LABELS.get(v, v) for v in pivot.index])
    ax.set_title(f"{metric} by X Dimension and Y Constraint")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def main() -> None:
    if not INPUT_JSON.exists():
        print(f"[Fatal] {INPUT_JSON.name} not found. Run step3_judge_main_set.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rel_out = OUT_DIR.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    print(f"\nOutput folder (repo-relative): {rel_out}")

    baseline_lookup = load_baseline_lookup(BASELINE_JSON)
    df = load_flat_records(INPUT_JSON, baseline_lookup)
    print(f"Loaded {len(df)} scored constrained records.")

    if df.empty:
        print("No scored records found. Nothing to analyze.")
        return

    df.to_csv(OUT_DIR / "main_set_flat_records.csv", index=False, encoding="utf-8-sig")
    aggregate(df, ["dimension", "constraint_id"]).to_csv(
        OUT_DIR / "main_set_xy_summary.csv", index=False, encoding="utf-8-sig"
    )
    aggregate(df, ["constraint_id"]).to_csv(
        OUT_DIR / "main_set_y_summary.csv", index=False, encoding="utf-8-sig"
    )

    report_text = build_report(df, bool(baseline_lookup))
    print(report_text)
    (OUT_DIR / "report_main_set.txt").write_text(report_text, encoding="utf-8")
    print("  Saved: report_main_set.txt")

    print("\nGenerating figures ...")
    save_bar_metric(df, "dimension_label", "DCR", "fig1_dcr_by_x.png", "Constrained DCR by X Dimension")
    save_bar_metric(df, "constraint_id", "CSR", "fig2_csr_by_y.png", "Constraint Success Rate by Y")
    save_bar_metric(
        df,
        "constraint_id",
        "effective_DCR",
        "fig3_effective_dcr_by_y.png",
        "Effective DCR by Y Constraint",
    )
    save_xy_heatmap(df, "effective_DCR", "fig4_xy_effective_dcr_heatmap.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
