#!/usr/bin/env python3
"""Generate a professional PDF report for step-response comparison."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def rank_series(values, ascending=True):
    # Stable ranking with NaN-aware behavior (NaN goes to worst rank)
    s = values.copy()
    if ascending:
        s = s.fillna(np.inf)
        return s.rank(method="min", ascending=True).astype(int)
    s = s.fillna(-np.inf)
    return s.rank(method="min", ascending=False).astype(int)


def fmt_num(x, digits=3, na_text="Not Available"):
    if pd.isna(x):
        return na_text
    return f"{x:.{digits}f}"


def generate_pdf(base_path, lap=2):
    summary_path = os.path.join(base_path, f"point_to_point_summary_lap{lap}.csv")
    details_path = os.path.join(base_path, f"point_to_point_details_lap{lap}.csv")
    tracking_plot_path = os.path.join(base_path, f"controller_tracking_comparison_lap{lap}_time.png")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not os.path.exists(details_path):
        raise FileNotFoundError(f"Missing details CSV: {details_path}")

    summary = pd.read_csv(summary_path)
    details = pd.read_csv(details_path)

    report_path = os.path.join(base_path, f"Step_Response_Comparison_Lap{lap}.pdf")

    # Composite score: lower is better for all included metrics.
    score_cols = [
        "x_rise_time_mean_s",
        "y_rise_time_mean_s",
        "x_overshoot_mean_pct",
        "y_overshoot_mean_pct",
        "x_settling_time_mean_s",
        "y_settling_time_mean_s",
    ]

    ranked = summary.copy()
    score = pd.Series(0.0, index=ranked.index)
    valid_counts = pd.Series(0, index=ranked.index)
    for c in score_cols:
        if c in ranked.columns:
            r = rank_series(ranked[c], ascending=True)
            score += r
            valid_counts += ranked[c].notna().astype(int)

    ranked["step_response_score"] = score / valid_counts.replace(0, np.nan)
    ranked = ranked.sort_values("step_response_score", ascending=True, na_position="last")

    with PdfPages(report_path) as pdf:
        # PAGE 1: Cover + executive summary
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Controller Step-Response Comparison Report", fontsize=22, fontweight="bold", y=0.96)

        ax = fig.add_subplot(111)
        ax.axis("off")

        subtitle = (
            f"Lap: {lap} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Metrics: Mean Rise Time, Overshoot, Settling Time (X and Y axes), plus tracking error context"
        )
        ax.text(0.5, 0.84, subtitle, ha="center", va="center", fontsize=11)

        glossary = (
            "Interpretation notes: \n"
            "- Not Settled: response did not enter and remain within the ±2% settling band in the evaluation window.\n"
            "- Not Available: metric could not be computed (e.g., transition too short or no valid crossing)."
        )
        ax.text(0.08, 0.77, glossary, ha="left", va="top", fontsize=9)

        top = ranked.iloc[0]
        exec_lines = [
            "Executive Summary",
            f"Best overall mean step-response score: {top['controller']} (score={fmt_num(top['step_response_score'], 2)})",
            f"Fastest mean Y rise time: {summary.loc[summary['y_rise_time_mean_s'].idxmin(), 'controller']} ({fmt_num(summary['y_rise_time_mean_s'].min())} s)",
            f"Lowest mean Y overshoot: {summary.loc[summary['y_overshoot_mean_pct'].idxmin(), 'controller']} ({fmt_num(summary['y_overshoot_mean_pct'].min())} %)",
        ]

        y0 = 0.60
        for i, line in enumerate(exec_lines):
            size = 14 if i == 0 else 11
            weight = "bold" if i == 0 else "normal"
            ax.text(0.08, y0 - i * 0.08, line, fontsize=size, fontweight=weight, ha="left")

        # Two stacked summary tables for readability.
        ax.text(0.04, 0.29, "Step-Response Metrics", fontsize=11, fontweight="bold", ha="left")
        step_cols = [
            "controller",
            "x_rise_time_mean_s",
            "y_rise_time_mean_s",
            "x_overshoot_mean_pct",
            "y_overshoot_mean_pct",
            "x_settling_time_mean_s",
            "y_settling_time_mean_s",
            "step_response_score",
        ]
        step_table_df = ranked[step_cols].copy()
        for c in step_cols[1:]:
            if c in ["x_settling_time_mean_s", "y_settling_time_mean_s"]:
                step_table_df[c] = step_table_df[c].apply(lambda v: fmt_num(v, na_text="Not Settled"))
            else:
                step_table_df[c] = step_table_df[c].apply(fmt_num)
        step_table_df.columns = [
            "Controller",
            "X Rise [s]",
            "Y Rise [s]",
            "X Overshoot [%]",
            "Y Overshoot [%]",
            "X Settling [s]",
            "Y Settling [s]",
            "Score",
        ]

        step_table = ax.table(
            cellText=step_table_df.values,
            colLabels=step_table_df.columns,
            cellLoc="center",
            bbox=[0.04, 0.15, 0.92, 0.13],
        )
        step_table.auto_set_font_size(False)
        step_table.set_fontsize(8)
        for j in range(len(step_table_df.columns)):
            step_table[(0, j)].set_facecolor("#1f4e79")
            step_table[(0, j)].set_text_props(color="white", weight="bold")
        for i in range(1, len(step_table_df) + 1):
            if i % 2 == 0:
                for j in range(len(step_table_df.columns)):
                    step_table[(i, j)].set_facecolor("#f4f7fb")

        ax.text(0.04, 0.13, "Tracking Error Context", fontsize=11, fontweight="bold", ha="left")
        err_cols = [
            "controller",
            "lap_time_s",
            "position_rmse_m",
            "position_mean_error_m",
            "position_max_error_m",
        ]
        err_table_df = ranked[err_cols].copy()
        for c in err_cols[1:]:
            err_table_df[c] = err_table_df[c].apply(fmt_num)
        err_table_df.columns = [
            "Controller",
            "Lap Time [s]",
            "RMSE [m]",
            "Mean Error [m]",
            "Max Error [m]",
        ]

        err_table = ax.table(
            cellText=err_table_df.values,
            colLabels=err_table_df.columns,
            cellLoc="center",
            bbox=[0.04, 0.03, 0.92, 0.09],
        )
        err_table.auto_set_font_size(False)
        err_table.set_fontsize(8)
        for j in range(len(err_table_df.columns)):
            err_table[(0, j)].set_facecolor("#1f4e79")
            err_table[(0, j)].set_text_props(color="white", weight="bold")
        for i in range(1, len(err_table_df) + 1):
            if i % 2 == 0:
                for j in range(len(err_table_df.columns)):
                    err_table[(i, j)].set_facecolor("#f4f7fb")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 2: Mean step-response bars
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle("Mean Step-Response Metrics by Controller", fontsize=16, fontweight="bold")

        plot_specs = [
            ("x_rise_time_mean_s", "Mean Rise Time X [s]"),
            ("y_rise_time_mean_s", "Mean Rise Time Y [s]"),
            ("x_overshoot_mean_pct", "Mean Overshoot X [%]"),
            ("y_overshoot_mean_pct", "Mean Overshoot Y [%]"),
        ]

        colors = ["#4e79a7", "#e15759", "#f28e2b"]
        for ax, (metric, title) in zip(axs.flat, plot_specs):
            vals = summary[metric]
            bars = ax.bar(summary["controller"], vals, color=colors)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=15)
            for b, v in zip(bars, vals):
                label = "Not Settled" if pd.isna(v) and "Settling" in title else ("Not Available" if pd.isna(v) else f"{v:.3f}")
                y = 0 if pd.isna(v) else v
                ax.text(b.get_x() + b.get_width() / 2, y + 0.01, label, ha="center", va="bottom", fontsize=8)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 3: Settling + transition coverage + context error metrics
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle("Settling Performance and Transition Coverage", fontsize=16, fontweight="bold")

        for ax, metric, title in [
            (axs[0, 0], "x_settling_time_mean_s", "Mean Settling Time X [s]"),
            (axs[0, 1], "y_settling_time_mean_s", "Mean Settling Time Y [s]"),
            (axs[1, 0], "position_rmse_m", "Position RMSE [m]"),
            (axs[1, 1], "position_max_error_m", "Max Position Error [m]"),
        ]:
            vals = summary[metric]
            bars = ax.bar(summary["controller"], vals, color=colors)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=15)
            for b, v in zip(bars, vals):
                label = "Not Settled" if pd.isna(v) and "Settling" in title else ("Not Available" if pd.isna(v) else f"{v:.3f}")
                y = 0 if pd.isna(v) else v
                ax.text(b.get_x() + b.get_width() / 2, y + 0.01, label, ha="center", va="bottom", fontsize=8)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 4: Detailed transition table excerpt
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Transition-Level Step-Response Details (Lap 2)", fontsize=16, fontweight="bold")
        ax = fig.add_subplot(111)
        ax.axis("off")

        details_view = details.copy()
        keep_cols = [
            "controller",
            "axis",
            "transition_id",
            "rise_time_s",
            "overshoot_pct",
            "settling_time_s",
            "steady_state_error",
        ]
        details_view = details_view[keep_cols]

        # show first 18 rows to keep page readable
        details_head = details_view.head(18).copy()
        for c in ["rise_time_s", "overshoot_pct", "steady_state_error"]:
            details_head[c] = details_head[c].apply(fmt_num)
        details_head["settling_time_s"] = details_head["settling_time_s"].apply(lambda v: fmt_num(v, na_text="Not Settled"))

        details_head.columns = [
            "Controller",
            "Axis",
            "Transition",
            "Rise [s]",
            "Overshoot [%]",
            "Settling [s]",
            "Steady-State Error",
        ]

        table = ax.table(
            cellText=details_head.values,
            colLabels=details_head.columns,
            cellLoc="center",
            bbox=[0.02, 0.05, 0.96, 0.84],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for j in range(len(details_head.columns)):
            table[(0, j)].set_facecolor("#1f4e79")
            table[(0, j)].set_text_props(color="white", weight="bold")

        for i in range(1, len(details_head) + 1):
            if i % 2 == 0:
                for j in range(len(details_head.columns)):
                    table[(i, j)].set_facecolor("#f4f7fb")

        ax.text(
            0.02,
            0.92,
            "Note: 'Not Settled' means the response did not settle in the evaluated transition window.",
            fontsize=9,
            ha="left",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 5: Optional tracking plot as appendix
        if os.path.exists(tracking_plot_path):
            img = plt.imread(tracking_plot_path)
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle("Appendix: Lap-2 Tracking Comparison (Time Domain)", fontsize=14, fontweight="bold")
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate professional step-response PDF report.")
    parser.add_argument(
        "--base-path",
        default=os.path.expanduser("~/sim_ws/src/neural_network_control/neural_network_control/"),
    )
    parser.add_argument("--lap", type=int, default=2)
    args = parser.parse_args()

    report_path = generate_pdf(os.path.expanduser(args.base_path), lap=args.lap)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    import argparse

    main()
