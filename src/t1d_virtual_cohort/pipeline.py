from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .diurnal import compute_diurnal_diagnostics
from .io import read_real_summary, read_virtual_summary
from .matching import feature_ablation, match_members
from .plotting import (
    plot_ablation,
    plot_agreement_grid,
    plot_diurnal,
    plot_matching_overview,
)
from .statistics import cluster_bootstrap_mean_ci, paired_tost


METRIC_LABELS = {
    "mean_glucose_mgdl": "Mean glucose (mg/dL)",
    "cv_percent": r"CV (\%)",
    "tir_percent": r"TIR (\%)",
    "gmi_percent": r"GMI (\%)",
    "tbr_percent": r"TBR (\%)",
    "tar_percent": r"TAR (\%)",
    "lbgi": "LBGI",
    "hbgi": "HBGI",
    "nocturnal_lbgi": "Nocturnal LBGI",
    "nocturnal_hbgi": "Nocturnal HBGI",
    "dawn_lbgi": "Dawn LBGI",
    "dawn_hbgi": "Dawn HBGI",
}


def _add_robust_verdict(result: dict) -> None:
    margin = result["margin"]
    bootstrap_equivalent = (
        result["cluster_bootstrap_ci90_low"] > -margin
        and result["cluster_bootstrap_ci90_high"] < margin
    )
    result["cluster_bootstrap_equivalent"] = bool(bootstrap_equivalent)
    if result["equivalent"] and bootstrap_equivalent:
        result["robust_verdict"] = "Robustly equivalent"
    elif result["equivalent"]:
        result["robust_verdict"] = "Not robust"
    else:
        result["robust_verdict"] = "Not equivalent"


def _tost_rows(
    matches: pd.DataFrame, margins: dict, replicates: int, seed: int
) -> pd.DataFrame:
    rows = []
    for metric, margin in margins.items():
        real_col = f"real_{metric}"
        virtual_col = f"virtual_{metric}"
        if real_col not in matches or virtual_col not in matches:
            continue
        result = paired_tost(matches[real_col], matches[virtual_col], margin)
        difference = matches[real_col] - matches[virtual_col]
        bootstrap = cluster_bootstrap_mean_ci(
            difference, matches["member_key"], replicates=replicates, seed=seed
        )
        result.update(
            {
                "metric": metric,
                "cluster_bootstrap_ci90_low": bootstrap[0],
                "cluster_bootstrap_ci90_high": bootstrap[1],
            }
        )
        _add_robust_verdict(result)
        rows.append(result)
    return pd.DataFrame(rows)


def _diurnal_rows(
    diurnal: pd.DataFrame, margins: dict, replicates: int, seed: int
) -> pd.DataFrame:
    qualified = diurnal[diurnal["qualified"]]
    rows = []
    for window in ["nocturnal", "dawn"]:
        for metric in ["lbgi", "hbgi"]:
            result = paired_tost(
                qualified[f"real_{window}_{metric}"],
                qualified[f"virtual_{window}_{metric}"],
                margins[metric],
            )
            difference = (
                qualified[f"real_{window}_{metric}"]
                - qualified[f"virtual_{window}_{metric}"]
            )
            bootstrap = cluster_bootstrap_mean_ci(
                difference,
                qualified["member_key"],
                replicates=replicates,
                seed=seed,
            )
            result.update(
                {
                    "metric": f"{window}_{metric}",
                    "cluster_bootstrap_ci90_low": bootstrap[0],
                    "cluster_bootstrap_ci90_high": bootstrap[1],
                }
            )
            _add_robust_verdict(result)
            rows.append(result)
    return pd.DataFrame(rows)


def _diurnal_threshold_sensitivity(
    diurnal: pd.DataFrame,
    margins: dict,
    thresholds: list[float],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    tables = []
    for threshold in thresholds:
        threshold_data = diurnal.copy()
        threshold_data["qualified"] = (
            (threshold_data["nocturnal_completeness"] >= threshold)
            & (threshold_data["dawn_completeness"] >= threshold)
        )
        table = _diurnal_rows(threshold_data, margins, replicates, seed)
        table.insert(0, "completeness_threshold", threshold)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _diurnal_distribution_summary(diurnal: pd.DataFrame) -> pd.DataFrame:
    qualified = diurnal[diurnal["qualified"]]
    rows = []
    for window in ["nocturnal", "dawn"]:
        for metric in ["lbgi", "hbgi"]:
            real = qualified[f"real_{window}_{metric}"]
            virtual = qualified[f"virtual_{window}_{metric}"]
            difference = real - virtual
            for source, values in [
                ("real", real),
                ("virtual", virtual),
                ("real_minus_virtual", difference),
            ]:
                rows.append(
                    {
                        "window": window,
                        "metric": metric,
                        "source": source,
                        "n": int(values.notna().sum()),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "median": float(values.median()),
                        "q1": float(values.quantile(0.25)),
                        "q3": float(values.quantile(0.75)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def _format_p(value: float) -> str:
    if value < 1e-16:
        return r"$<10^{-16}$"
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def _write_tost_latex(table: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Metric & Bias & TOST 90\% CI & Cluster 90\% CI & Final verdict \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        metric = METRIC_LABELS.get(row.metric, str(row.metric).replace("_", " ").title())
        lines.append(
            f"{metric} & {row.bias_real_minus_virtual:+.2f} & "
            f"({row.ci90_low:+.2f}, {row.ci90_high:+.2f}) & "
            f"({row.cluster_bootstrap_ci90_low:+.2f}, "
            f"{row.cluster_bootstrap_ci90_high:+.2f}) & "
            f"{row.robust_verdict} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cohort_latex(real: pd.DataFrame, path: Path) -> None:
    labels = [
        ("mean_glucose_mgdl", "Mean glucose (mg/dL)"),
        ("gmi_percent", "GMI (\\%)"),
        ("tir_percent", "TIR (\\%)"),
        ("tbr_percent", "TBR (\\%)"),
        ("tar_percent", "TAR (\\%)"),
        ("cv_percent", "CV (\\%)"),
        ("lbgi", "LBGI"),
        ("hbgi", "HBGI"),
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{Glycemic characteristics of the Stage-1 cohort ($N={len(real)}$).}}",
        r"\label{tab:cohort}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Metric & Mean $\pm$ SD & Median [IQR] \\",
        r"\midrule",
    ]
    for metric, label in labels:
        if metric not in real:
            continue
        values = real[metric].dropna()
        lines.append(
            f"{label} & {values.mean():.2f} $\\pm$ {values.std(ddof=1):.2f} & "
            f"{values.median():.2f} [{values.quantile(.25):.2f}, {values.quantile(.75):.2f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_margin_latex(margins: dict, path: Path) -> None:
    rationales = {
        "mean_glucose_mgdl": "Population mean-glucose tolerance",
        "cv_percent": "Variability tolerance around consensus CV use",
        "tir_percent": "Population-level range tolerance",
        "gmi_percent": "Derived average-glycemia tolerance",
        "tbr_percent": "Safety-sensitive low-range tolerance",
        "tar_percent": "Population-level high-range tolerance",
        "lbgi": "Operational low-risk-index tolerance",
        "hbgi": "Operational high-risk-index tolerance",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Study-specified equivalence margins. Percentage metrics use percentage points.}",
        r"\label{tab:margins}",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Metric & Margin & Interpretation \\",
        r"\midrule",
    ]
    for metric, margin in margins.items():
        label = METRIC_LABELS.get(metric, metric)
        unit = " mg/dL" if metric == "mean_glucose_mgdl" else ""
        lines.append(
            f"{label} & $\\pm${margin:g}{unit} & {rationales[metric]} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    real_summary_path,
    virtual_summary_path,
    output_dir,
    config_path,
    real_trace_dir=None,
    virtual_trace_dir=None,
) -> dict:
    output = Path(output_dir)
    figures = output / "figures"
    tables = output / "tables"
    private = output / "private"
    for directory in [figures, tables, private]:
        directory.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    real = read_real_summary(real_summary_path)
    virtual = read_virtual_summary(virtual_summary_path)
    features = config["matching"]["features"]
    matches = match_members(
        real, virtual, features, config["matching"]["normalization"]
    )
    matches.to_csv(private / "matches_3d.csv", index=False)
    ablation = feature_ablation(real, virtual, features)
    ablation.to_csv(tables / "feature_ablation.csv", index=False)
    bootstrap = config["bootstrap"]
    aggregate = _tost_rows(
        matches,
        config["equivalence_margins"],
        bootstrap["replicates"],
        bootstrap["seed"],
    )
    aggregate.to_csv(tables / "aggregate_equivalence.csv", index=False)
    utilization = (
        matches.groupby("scenario")
        .size()
        .rename("n_assignments")
        .reset_index()
        .sort_values("scenario")
    )
    utilization.to_csv(tables / "scenario_utilization.csv", index=False)

    excellent = float(config["matching"]["excellent_distance"])
    good = float(config["matching"]["good_distance"])
    summary = {
        "stage1_n": int(len(real)),
        "virtual_profiles": int(len(virtual)),
        "base_virtual_subjects": int(virtual["virtual_id"].nunique()),
        "selected_profiles": int(matches["member_key"].nunique()),
        "mean_matching_distance": float(matches["distance"].mean()),
        "median_matching_distance": float(matches["distance"].median()),
        "excellent_percent": float((matches["distance"] < excellent).mean() * 100),
        "good_percent": float(
            ((matches["distance"] >= excellent) & (matches["distance"] < good)).mean()
            * 100
        ),
        "moderate_percent": float((matches["distance"] >= good).mean() * 100),
        "matching_features": features,
    }

    plot_matching_overview(matches, figures, excellent, good)
    plot_agreement_grid(
        matches,
        ["mean_glucose_mgdl", "cv_percent", "tir_percent"],
        figures,
        "fig2_anchor_agreement",
    )
    independent = [
        metric
        for metric in ["tbr_percent", "tar_percent", "lbgi", "hbgi"]
        if f"real_{metric}" in matches
    ]
    if independent:
        plot_agreement_grid(matches, independent, figures, "fig5_independent_metrics")
    plot_ablation(ablation, figures)

    _write_cohort_latex(real, tables / "table_cohort.tex")
    _write_margin_latex(
        config["equivalence_margins"], tables / "table_equivalence_margins.tex"
    )
    _write_tost_latex(
        aggregate,
        tables / "table_aggregate_equivalence.tex",
        "Aggregate agreement under 3D matching (real minus virtual).",
        "tab:aggregate",
    )

    if real_trace_dir and virtual_trace_dir:
        qc = config["quality_control"]
        diurnal = compute_diurnal_diagnostics(
            matches,
            real_trace_dir,
            virtual_trace_dir,
            qc["diurnal_minimum_completeness"],
            qc["expected_interval_minutes"],
            tuple(qc["nocturnal_window"]),
            tuple(qc["dawn_window"]),
        )
        diurnal.to_csv(private / "diurnal_subject_level.csv", index=False)
        diurnal_table = _diurnal_rows(
            diurnal,
            config["diurnal_margins"],
            bootstrap["replicates"],
            bootstrap["seed"],
        )
        diurnal_table.to_csv(tables / "diurnal_equivalence.csv", index=False)
        sensitivity = _diurnal_threshold_sensitivity(
            diurnal,
            config["diurnal_margins"],
            [0.60, 0.70, 0.80, 0.90],
            bootstrap["replicates"],
            bootstrap["seed"],
        )
        sensitivity.to_csv(
            tables / "diurnal_threshold_sensitivity.csv", index=False
        )
        distributions = _diurnal_distribution_summary(diurnal)
        distributions.to_csv(
            tables / "diurnal_distribution_summary.csv", index=False
        )
        summary["diurnal_n"] = int(diurnal["qualified"].sum())
        plot_diurnal(diurnal, figures)
        _write_tost_latex(
            diurnal_table,
            tables / "table_diurnal_equivalence.tex",
            "Coverage-qualified diurnal risk agreement (real minus virtual).",
            "tab:diurnal",
        )

    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
