import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def resolve_controller_csv(base_path, latest_name, lap, legacy_name=None):
    stem, ext = os.path.splitext(latest_name)
    lap_candidate = os.path.join(base_path, f"{stem}_{lap}{ext}")
    if os.path.exists(lap_candidate):
        return lap_candidate

    latest_candidate = os.path.join(base_path, latest_name)
    if os.path.exists(latest_candidate):
        return latest_candidate

    if legacy_name is not None:
        legacy_candidate = os.path.join(base_path, legacy_name)
        if os.path.exists(legacy_candidate):
            return legacy_candidate

    return None


def load_xy_and_time(csv_path, x_col="x", y_col="y"):
    if csv_path is None or not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        return None, None

    xy = df[[x_col, y_col]].dropna().to_numpy()
    lap_time = None
    if "lap_time_s" in df.columns:
        t = pd.to_numeric(df["lap_time_s"], errors="coerce").dropna()
        if not t.empty:
            lap_time = float(t.iloc[0])

    return xy, lap_time


def build_time_axis(num_samples, lap_time):
    if num_samples < 2:
        return np.array([0.0])
    if lap_time is not None and lap_time > 0.0:
        return np.linspace(0.0, float(lap_time), num_samples)
    return np.linspace(0.0, float(num_samples - 1), num_samples)


def point_ahead_on_polyline(points_xy, distance_m, lateral_offset_m=0.0):
    if len(points_xy) == 0:
        raise ValueError("No points available to compute forward target point")
    if len(points_xy) == 1 or distance_m <= 0.0:
        return points_xy[0].astype(float)

    diffs = np.diff(points_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])

    target_s = min(float(distance_m), float(s[-1]))
    x = np.interp(target_s, s, points_xy[:, 0])
    y = np.interp(target_s, s, points_xy[:, 1])
    target = np.array([x, y], dtype=float)

    if abs(lateral_offset_m) > 1e-9:
        seg_idx = int(np.searchsorted(s, target_s, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(diffs) - 1))
        tangent = diffs[seg_idx]
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-9:
            tangent = tangent / t_norm
            normal_left = np.array([-tangent[1], tangent[0]], dtype=float)
            target = target + float(lateral_offset_m) * normal_left

    return target


def first_index(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size > 0 else None


def evaluate_axis_step(signal, target, t, rel_tol=0.05, abs_tol=0.2):
    y0 = float(signal[0])
    amplitude = abs(target - y0)
    eps = 1e-9

    error = signal - target
    direction = 1.0 if (target - y0) >= 0.0 else -1.0

    if amplitude < eps:
        settled_mask = np.abs(error) <= abs_tol
        settling = 0.0 if np.all(settled_mask) else np.nan
        return {
            "rise_time_s": 0.0,
            "overshoot_pct": 0.0,
            "settling_time_s": settling,
            "steady_state_error": float(error[-1]),
            "time_to_hit_abs_tol_s": 0.0 if settled_mask[0] else np.nan,
            "max_abs_error": float(np.max(np.abs(error))),
        }

    normalized = direction * (signal - y0) / amplitude

    i10 = first_index(normalized >= 0.1)
    i90 = first_index(normalized >= 0.9)
    rise_time = np.nan
    if i10 is not None and i90 is not None and i90 >= i10:
        rise_time = float(t[i90] - t[i10])

    overshoot_pct = float(max(0.0, (np.nanmax(normalized) - 1.0) * 100.0))

    settle_mask = (np.abs(normalized - 1.0) <= rel_tol) | (np.abs(error) <= abs_tol)
    settling_time = np.nan
    for k in range(len(settle_mask)):
        if np.all(settle_mask[k:]):
            settling_time = float(t[k])
            break

    hit_idx = first_index(np.abs(error) <= abs_tol)
    hit_time = np.nan if hit_idx is None else float(t[hit_idx])

    return {
        "rise_time_s": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time,
        "steady_state_error": float(error[-1]),
        "time_to_hit_abs_tol_s": hit_time,
        "max_abs_error": float(np.max(np.abs(error))),
    }


def evaluate_distance_response(path_xy, target_xy, t, dist_tol=0.3):
    dist = np.linalg.norm(path_xy - target_xy, axis=1)
    d0 = float(dist[0])

    hit_idx = first_index(dist <= dist_tol)
    settle_time = np.nan
    for k in range(len(dist)):
        if np.all(dist[k:] <= dist_tol):
            settle_time = float(t[k])
            break

    idx90 = first_index(dist <= 0.9 * d0) if d0 > 1e-9 else 0
    idx10 = first_index(dist <= 0.1 * d0) if d0 > 1e-9 else 0
    reduction_rise = np.nan
    if idx90 is not None and idx10 is not None and idx10 >= idx90:
        reduction_rise = float(t[idx10] - t[idx90])

    min_idx = int(np.argmin(dist))

    return {
        "initial_distance_m": d0,
        "min_distance_m": float(dist[min_idx]),
        "time_to_min_distance_s": float(t[min_idx]),
        "time_to_hit_dist_tol_s": np.nan if hit_idx is None else float(t[hit_idx]),
        "distance_settling_time_s": settle_time,
        "distance_reduction_rise_time_s": reduction_rise,
        "distance_series": dist,
    }


def fmt(v, digits=3, na_text="Not Available"):
    if pd.isna(v):
        return na_text
    return f"{v:.{digits}f}"


def generate_pdf(report_path, summary_df, target_xy, lap, axis_tol, abs_tol, dist_tol, series_data):
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")

        fig.suptitle("Single-Target Step-Response Comparison", fontsize=20, fontweight="bold", y=0.96)
        ax.text(
            0.5,
            0.88,
            f"Lap {lap} | Target Point: ({target_xy[0]:.3f}, {target_xy[1]:.3f}) m | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            fontsize=11,
        )
        ax.text(
            0.5,
            0.84,
            f"Criteria: axis rel tol={axis_tol:.3f}, axis abs tol={abs_tol:.3f} m, distance tol={dist_tol:.3f} m",
            ha="center",
            fontsize=10,
        )

        table_cols = [
            "controller",
            "lap_time_s",
            "x_rise_time_s",
            "x_overshoot_pct",
            "x_settling_time_s",
            "y_rise_time_s",
            "y_overshoot_pct",
            "y_settling_time_s",
            "min_distance_m",
            "time_to_hit_dist_tol_s",
        ]
        table_df = summary_df[table_cols].copy()
        for c in table_cols[1:]:
            if "settling" in c:
                table_df[c] = table_df[c].apply(lambda v: fmt(v, na_text="Not Settled"))
            else:
                table_df[c] = table_df[c].apply(fmt)

        table_df.columns = [
            "Controller",
            "Lap Time [s]",
            "X Rise [s]",
            "X Over [%]",
            "X Settle [s]",
            "Y Rise [s]",
            "Y Over [%]",
            "Y Settle [s]",
            "Min Dist [m]",
            "Hit Dist Tol [s]",
        ]

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            bbox=[0.03, 0.50, 0.94, 0.28],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for j in range(len(table_df.columns)):
            table[(0, j)].set_facecolor("#1f4e79")
            table[(0, j)].set_text_props(color="white", weight="bold")
        for i in range(1, len(table_df) + 1):
            if i % 2 == 0:
                for j in range(len(table_df.columns)):
                    table[(i, j)].set_facecolor("#f4f7fb")

        ax.text(
            0.03,
            0.44,
            "Notes: 'Not Settled' means the signal did not enter and remain inside tolerance until the end of lap.",
            fontsize=9,
            ha="left",
        )
        ax.text(
            0.03,
            0.40,
            "Lower values are better for rise time, settling time, overshoot, and distance metrics.",
            fontsize=9,
            ha="left",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axs = plt.subplots(3, 1, figsize=(11.69, 8.27), sharex=False)
        fig.suptitle("Target-Point Response Signals", fontsize=16, fontweight="bold")

        colors = {
            "Pure Pursuit": "#4e79a7",
            "PINN Drive": "#e15759",
            "NN AI Controller": "#f28e2b",
        }

        for row in series_data:
            name = row["controller"]
            t = row["time"]
            xy = row["xy"]
            axs[0].plot(t, xy[:, 0], color=colors.get(name, None), label=name, linewidth=2.0)
            axs[1].plot(t, xy[:, 1], color=colors.get(name, None), label=name, linewidth=2.0)
            axs[2].plot(t, row["dist"], color=colors.get(name, None), label=name, linewidth=2.0)

        axs[0].axhline(target_xy[0], color="black", linestyle="--", linewidth=1.5, label="Target X")
        axs[1].axhline(target_xy[1], color="black", linestyle="--", linewidth=1.5, label="Target Y")
        axs[2].axhline(dist_tol, color="black", linestyle="--", linewidth=1.5, label="Distance tolerance")

        axs[0].set_ylabel("X [m]")
        axs[1].set_ylabel("Y [m]")
        axs[2].set_ylabel("Distance to target [m]")
        axs[2].set_xlabel("Time [s]")

        for ax in axs:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Single combined step-response plot (all controllers on one graph).
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        fig.suptitle("Step Response vs Time (Single Graph)", fontsize=16, fontweight="bold")

        for row in series_data:
            name = row["controller"]
            t = row["time"]
            d = row["dist"]
            d0 = float(d[0]) if len(d) > 0 else 0.0
            if d0 > 1e-9:
                response = 1.0 - (d / d0)
            else:
                response = np.ones_like(d)
            ax.plot(t, response, linewidth=2.2, label=name)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Target (normalized)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Normalized response (1 - d(t)/d0)")
        ax.set_ylim([-0.1, 1.2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        ax.text(
            0.01,
            0.02,
            "Higher and faster is better. 1.0 means the controller reached zero distance error relative to its initial distance.",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single-target step-response comparison PDF for three controllers.")
    parser.add_argument(
        "--base-path",
        default=os.path.expanduser("~/sim_ws/src/neural_network_control/neural_network_control/"),
    )
    parser.add_argument("--lap", type=int, default=2)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument(
        "--target-ahead-distance",
        type=float,
        default=3.0,
        help="If target-x/target-y are not provided, choose a target this many meters ahead on waypoints (default: 3.0).",
    )
    parser.add_argument(
        "--target-lateral-offset",
        type=float,
        default=0.0,
        help="Optional lateral offset in meters from the forward target point (positive = left of path tangent).",
    )
    parser.add_argument("--axis-rel-tol", type=float, default=0.05)
    parser.add_argument("--axis-abs-tol", type=float, default=0.20)
    parser.add_argument("--dist-tol", type=float, default=0.30)
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Defaults to target_point_step_response_summary_lap<N>.csv under base-path.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Defaults to Target_Point_Step_Response_Comparison_Lap<N>.pdf under base-path.",
    )
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)

    waypoints_path = os.path.join(base_path, "waypoints.csv")
    if not os.path.exists(waypoints_path):
        raise FileNotFoundError(f"Missing waypoint file: {waypoints_path}")

    waypoints_df = pd.read_csv(waypoints_path)
    if "x" not in waypoints_df.columns or "y" not in waypoints_df.columns:
        raise ValueError("waypoints.csv must contain x and y columns")

    if args.target_x is None or args.target_y is None:
        waypoint_xy = waypoints_df[["x", "y"]].to_numpy(dtype=float)
        target_xy = point_ahead_on_polyline(
            waypoint_xy,
            args.target_ahead_distance,
            lateral_offset_m=args.target_lateral_offset,
        )
    else:
        target_xy = np.array([float(args.target_x), float(args.target_y)], dtype=float)

    controller_specs = [
        ("Pure Pursuit", "pure_pursuit_reference_lap.csv", "pp_reference_lap.csv", "x", "y"),
        ("PINN Drive", "pinn_reference_lap.csv", None, "x", "y"),
        ("NN AI Controller", "nn_ai_reference_lap.csv", None, "x", "y"),
    ]

    rows = []
    series_data = []

    for name, latest_name, legacy_name, x_col, y_col in controller_specs:
        csv_path = resolve_controller_csv(base_path, latest_name, args.lap, legacy_name)
        xy, lap_time = load_xy_and_time(csv_path, x_col=x_col, y_col=y_col)
        if xy is None and legacy_name is not None and csv_path is not None and csv_path.endswith(legacy_name):
            xy, lap_time = load_xy_and_time(csv_path, x_col="pp_x", y_col="pp_y")

        if xy is None:
            raise FileNotFoundError(f"Could not load lap data for {name}")

        if lap_time is None or lap_time <= 0.0:
            lap_time = float(len(xy) - 1)

        t = build_time_axis(len(xy), lap_time)

        x_metrics = evaluate_axis_step(
            xy[:, 0],
            target_xy[0],
            t,
            rel_tol=args.axis_rel_tol,
            abs_tol=args.axis_abs_tol,
        )
        y_metrics = evaluate_axis_step(
            xy[:, 1],
            target_xy[1],
            t,
            rel_tol=args.axis_rel_tol,
            abs_tol=args.axis_abs_tol,
        )
        d_metrics = evaluate_distance_response(xy, target_xy, t, dist_tol=args.dist_tol)

        rows.append(
            {
                "controller": name,
                "lap": args.lap,
                "target_x": float(target_xy[0]),
                "target_y": float(target_xy[1]),
                "lap_time_s": float(lap_time),
                "x_rise_time_s": x_metrics["rise_time_s"],
                "x_overshoot_pct": x_metrics["overshoot_pct"],
                "x_settling_time_s": x_metrics["settling_time_s"],
                "x_steady_state_error": x_metrics["steady_state_error"],
                "x_time_to_hit_abs_tol_s": x_metrics["time_to_hit_abs_tol_s"],
                "y_rise_time_s": y_metrics["rise_time_s"],
                "y_overshoot_pct": y_metrics["overshoot_pct"],
                "y_settling_time_s": y_metrics["settling_time_s"],
                "y_steady_state_error": y_metrics["steady_state_error"],
                "y_time_to_hit_abs_tol_s": y_metrics["time_to_hit_abs_tol_s"],
                "initial_distance_m": d_metrics["initial_distance_m"],
                "min_distance_m": d_metrics["min_distance_m"],
                "time_to_min_distance_s": d_metrics["time_to_min_distance_s"],
                "time_to_hit_dist_tol_s": d_metrics["time_to_hit_dist_tol_s"],
                "distance_settling_time_s": d_metrics["distance_settling_time_s"],
                "distance_reduction_rise_time_s": d_metrics["distance_reduction_rise_time_s"],
            }
        )

        series_data.append(
            {
                "controller": name,
                "time": t,
                "xy": xy,
                "dist": d_metrics["distance_series"],
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_output = args.summary_output or os.path.join(
        base_path, f"target_point_step_response_summary_lap{args.lap}.csv"
    )
    report_output = args.report_output or os.path.join(
        base_path, f"Target_Point_Step_Response_Comparison_Lap{args.lap}.pdf"
    )

    summary_df.to_csv(summary_output, index=False)
    generate_pdf(
        report_output,
        summary_df,
        target_xy,
        args.lap,
        args.axis_rel_tol,
        args.axis_abs_tol,
        args.dist_tol,
        series_data,
    )

    print(f"Target point used: ({target_xy[0]:.6f}, {target_xy[1]:.6f})")
    print(f"Saved summary: {summary_output}")
    print(f"Saved report: {report_output}")


if __name__ == "__main__":
    main()