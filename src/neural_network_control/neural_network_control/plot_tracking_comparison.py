import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_controller_csv(base_path, latest_name, lap, legacy_name=None):
    if lap is not None:
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
        lap_time_series = pd.to_numeric(df["lap_time_s"], errors="coerce").dropna()
        if not lap_time_series.empty:
            lap_time = float(lap_time_series.iloc[0])

    return xy, lap_time


def close_loop(points):
    if len(points) < 2:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def desired_from_progress(waypoints_closed, num_samples):
    if num_samples < 2:
        return waypoints_closed[:1]

    diffs = np.diff(waypoints_closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])

    s_uniform = np.linspace(0.0, s[-1], num_samples)
    x_uniform = np.interp(s_uniform, s, waypoints_closed[:, 0])
    y_uniform = np.interp(s_uniform, s, waypoints_closed[:, 1])
    return np.column_stack([x_uniform, y_uniform])


def compute_signed_cte(path_xy, waypoints):
    if len(path_xy) == 0:
        return np.array([])

    ctes = np.zeros(len(path_xy))
    n = len(waypoints)
    for i, pt in enumerate(path_xy):
        deltas = waypoints - pt
        dists = np.linalg.norm(deltas, axis=1)
        idx = int(np.argmin(dists))
        next_idx = (idx + 1) % n

        w0 = waypoints[idx]
        w1 = waypoints[next_idx]
        tangent = w1 - w0
        rel = pt - w0

        cross_z = tangent[0] * rel[1] - tangent[1] * rel[0]
        sign = 1.0 if cross_z >= 0.0 else -1.0
        ctes[i] = sign * dists[idx]

    return ctes


def format_controller_label(name, lap_time):
    if lap_time is None:
        return name
    return f"{name} ({lap_time:.3f}s)"


def build_time_axis(num_samples, lap_time):
    if num_samples < 2:
        return np.array([0.0])
    if lap_time is not None and lap_time > 0.0:
        return np.linspace(0.0, float(lap_time), num_samples)
    return np.linspace(0.0, float(num_samples - 1), num_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Plot X/Y/CTE tracking comparison for lap 2 with three controllers."
    )
    parser.add_argument(
        "--base-path",
        default=os.path.expanduser("~/sim_ws/src/neural_network_control/neural_network_control/"),
        help="Directory containing waypoints and controller lap CSV files.",
    )
    parser.add_argument(
        "--lap",
        type=int,
        default=2,
        help="Lap number to load for each controller. Default: 2.",
    )
    parser.add_argument(
        "--output",
        default="controller_tracking_comparison_lap2_time.png",
        help="Output PNG name (saved under base path unless absolute).",
    )
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    waypoints_path = os.path.join(base_path, "waypoints.csv")

    pp_path = resolve_controller_csv(
        base_path,
        latest_name="pure_pursuit_reference_lap.csv",
        lap=args.lap,
        legacy_name="pp_reference_lap.csv",
    )
    pinn_path = resolve_controller_csv(
        base_path,
        latest_name="pinn_reference_lap.csv",
        lap=args.lap,
    )
    nn_path = resolve_controller_csv(
        base_path,
        latest_name="nn_ai_reference_lap.csv",
        lap=args.lap,
    )

    if not os.path.exists(waypoints_path):
        raise FileNotFoundError(f"Missing desired path file: {waypoints_path}")

    waypoints_df = pd.read_csv(waypoints_path)
    if "x" not in waypoints_df.columns or "y" not in waypoints_df.columns:
        raise ValueError("waypoints.csv must contain x and y columns")
    waypoints = waypoints_df[["x", "y"]].to_numpy()
    waypoints_closed = close_loop(waypoints)

    pure_pursuit, pp_time = load_xy_and_time(pp_path)
    if pure_pursuit is None and pp_path is not None and pp_path.endswith("pp_reference_lap.csv"):
        pure_pursuit, pp_time = load_xy_and_time(pp_path, x_col="pp_x", y_col="pp_y")

    pinn_drive, pinn_time = load_xy_and_time(pinn_path)
    nn_ai, nn_time = load_xy_and_time(nn_path)

    missing = []
    if pure_pursuit is None:
        missing.append("Pure Pursuit")
    if pinn_drive is None:
        missing.append("PINN Drive")
    if nn_ai is None:
        missing.append("NN AI Controller")
    if missing:
        raise FileNotFoundError(
            "Missing lap trajectory CSVs for: " + ", ".join(missing)
        )

    controllers = [
        ("Pure Pursuit", pure_pursuit, pp_time, "blue"),
        ("PINN Drive", pinn_drive, pinn_time, "red"),
        ("NN AI Controller", nn_ai, nn_time, "orange"),
    ]

    available_lap_times = [t for _, _, t, _ in controllers if t is not None and t > 0.0]
    nominal_lap_time = max(available_lap_times) if available_lap_times else None

    desired_dense = desired_from_progress(waypoints_closed, 1000)
    desired_time = build_time_axis(len(desired_dense), nominal_lap_time)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # X subplot
    axs[0].plot(desired_time, desired_dense[:, 0], color="green", linewidth=2.5, label="Desired X")
    for name, path_xy, lap_time, color in controllers:
        t = build_time_axis(len(path_xy), lap_time)
        axs[0].plot(
            t,
            path_xy[:, 0],
            color=color,
            linewidth=2.0,
            label=format_controller_label(name, lap_time),
        )
    axs[0].set_ylabel("X [m]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    # Y subplot
    axs[1].plot(desired_time, desired_dense[:, 1], color="green", linewidth=2.5, label="Desired Y")
    for name, path_xy, _, color in controllers:
        lap_time = next(t for n, _, t, _ in controllers if n == name)
        t = build_time_axis(len(path_xy), lap_time)
        axs[1].plot(t, path_xy[:, 1], color=color, linewidth=2.0)
    axs[1].set_ylabel("Y [m]")
    axs[1].grid(True, alpha=0.3)

    # CTE subplot
    for name, path_xy, _, color in controllers:
        lap_time = next(t for n, _, t, _ in controllers if n == name)
        t = build_time_axis(len(path_xy), lap_time)
        cte = compute_signed_cte(path_xy, waypoints)
        axs[2].plot(t, cte, color=color, linewidth=2.0, label=name)
    axs[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("CTE [m]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="best")

    fig.suptitle(f"Tracking Comparison | Lap {args.lap} | X / Y / CTE vs Time", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_path, output_path)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
