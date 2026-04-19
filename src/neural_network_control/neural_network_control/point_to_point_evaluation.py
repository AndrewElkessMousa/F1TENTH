import argparse
import os

import numpy as np
import pandas as pd


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


def close_loop(points):
    if len(points) < 2:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def desired_from_arclength(waypoints_closed, num_samples):
    if num_samples < 2:
        return waypoints_closed[:1]

    diffs = np.diff(waypoints_closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])

    s_uniform = np.linspace(0.0, s[-1], num_samples)
    x_uniform = np.interp(s_uniform, s, waypoints_closed[:, 0])
    y_uniform = np.interp(s_uniform, s, waypoints_closed[:, 1])
    return np.column_stack([x_uniform, y_uniform])


def segment_transitions(signal, min_delta=1.0, eps=1e-3, min_samples=20):
    d = np.diff(signal)
    sign = np.zeros_like(d, dtype=int)
    sign[d > eps] = 1
    sign[d < -eps] = -1

    segments = []
    i = 0
    n = len(sign)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue

        sgn = sign[i]
        start = i
        i += 1
        while i < n and sign[i] == sgn:
            i += 1
        end = i

        seg_start = start
        seg_end = end
        if (seg_end - seg_start + 1) < min_samples:
            continue

        delta = signal[seg_end] - signal[seg_start]
        if abs(delta) < min_delta:
            continue

        segments.append((seg_start, seg_end, sgn))

    return segments


def evaluate_step_response(
    desired,
    actual,
    time_axis,
    segments,
    settle_tol=0.05,
    settle_window_ratio=0.75,
    settle_min_samples=30,
    settle_abs_band=None,
):
    rows = []

    for idx, (start, end, sgn) in enumerate(segments, start=1):
        seg_len = end - start + 1
        window_end = min(
            len(actual) - 1,
            end + max(settle_min_samples, int(settle_window_ratio * seg_len)),
        )

        y0 = desired[start]
        yf = desired[end]
        amplitude = abs(yf - y0)
        if amplitude < 1e-9:
            continue

        direction = 1.0 if (yf - y0) >= 0.0 else -1.0

        t = time_axis[start : window_end + 1] - time_axis[start]
        y = actual[start : window_end + 1]

        normalized = direction * (y - y0) / amplitude

        rise_time = np.nan
        idx10 = np.where(normalized >= 0.1)[0]
        idx90 = np.where(normalized >= 0.9)[0]
        if idx10.size > 0 and idx90.size > 0:
            i10 = int(idx10[0])
            i90_candidates = idx90[idx90 >= i10]
            if i90_candidates.size > 0:
                i90 = int(i90_candidates[0])
                rise_time = float(t[i90] - t[i10])

        overshoot_pct = float(max(0.0, (np.nanmax(normalized) - 1.0) * 100.0))

        settling_time = np.nan
        if settle_abs_band is not None and settle_abs_band > 0:
            settle_mask = (np.abs(normalized - 1.0) <= settle_tol) | (np.abs(y - yf) <= settle_abs_band)
        else:
            settle_mask = np.abs(normalized - 1.0) <= settle_tol

        for k in range(len(normalized)):
            if np.all(settle_mask[k:]):
                settling_time = float(t[k])
                break

        steady_state_error = float(y[-1] - yf)
        peak_error = float(np.max(np.abs(y - np.linspace(y0, yf, len(y)))))

        rows.append(
            {
                "transition_id": idx,
                "start_idx": int(start),
                "end_idx": int(end),
                "rise_time_s": rise_time,
                "overshoot_pct": overshoot_pct,
                "settling_time_s": settling_time,
                "steady_state_error": steady_state_error,
                "peak_following_error": peak_error,
            }
        )

    return pd.DataFrame(rows)


def summarize_transition_metrics(df):
    if df.empty:
        return {
            "num_transitions": 0,
            "rise_time_mean_s": np.nan,
            "overshoot_mean_pct": np.nan,
            "settling_time_mean_s": np.nan,
            "steady_state_error_mean": np.nan,
            "peak_following_error_mean": np.nan,
        }

    return {
        "num_transitions": int(len(df)),
        "rise_time_mean_s": float(df["rise_time_s"].mean(skipna=True)),
        "overshoot_mean_pct": float(df["overshoot_pct"].mean(skipna=True)),
        "settling_time_mean_s": float(df["settling_time_s"].mean(skipna=True)),
        "steady_state_error_mean": float(df["steady_state_error"].mean(skipna=True)),
        "peak_following_error_mean": float(df["peak_following_error"].mean(skipna=True)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Point-to-point controller evaluation (rise time, overshoot, settling) for lap data."
    )
    parser.add_argument(
        "--base-path",
        default=os.path.expanduser("~/sim_ws/src/neural_network_control/neural_network_control/"),
        help="Path containing waypoints and lap CSV files.",
    )
    parser.add_argument("--lap", type=int, default=2, help="Lap index to evaluate (default: 2).")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary CSV output path. Defaults to base-path/point_to_point_summary_lap<N>.csv",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional detailed transitions CSV output path. Defaults to base-path/point_to_point_details_lap<N>.csv",
    )
    parser.add_argument(
        "--settle-tol",
        type=float,
        default=0.05,
        help="Normalized settling tolerance (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--settle-window-ratio",
        type=float,
        default=0.75,
        help="Extra settling window length as a ratio of transition length (default: 0.75).",
    )
    parser.add_argument(
        "--settle-min-samples",
        type=int,
        default=30,
        help="Minimum extra samples used for settling window (default: 30).",
    )
    parser.add_argument(
        "--settle-abs-band",
        type=float,
        default=0.20,
        help="Optional absolute settling band in meters around final value (default: 0.20). Set <=0 to disable.",
    )
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    waypoints_path = os.path.join(base_path, "waypoints.csv")

    if not os.path.exists(waypoints_path):
        raise FileNotFoundError(f"Missing waypoint file: {waypoints_path}")

    waypoints_df = pd.read_csv(waypoints_path)
    if "x" not in waypoints_df.columns or "y" not in waypoints_df.columns:
        raise ValueError("waypoints.csv must have x and y columns")

    waypoints = close_loop(waypoints_df[["x", "y"]].to_numpy())

    controller_specs = [
        ("Pure Pursuit", "pure_pursuit_reference_lap.csv", "pp_reference_lap.csv", "x", "y"),
        ("PINN Drive", "pinn_reference_lap.csv", None, "x", "y"),
        ("NN AI Controller", "nn_ai_reference_lap.csv", None, "x", "y"),
    ]

    summary_rows = []
    detail_frames = []

    for controller_name, latest_name, legacy_name, x_col, y_col in controller_specs:
        csv_path = resolve_controller_csv(base_path, latest_name, args.lap, legacy_name)

        path_xy, lap_time = load_xy_and_time(csv_path, x_col=x_col, y_col=y_col)
        if path_xy is None and legacy_name is not None and csv_path is not None and csv_path.endswith(legacy_name):
            path_xy, lap_time = load_xy_and_time(csv_path, x_col="pp_x", y_col="pp_y")

        if path_xy is None:
            raise FileNotFoundError(f"Could not load lap data for {controller_name}")

        if lap_time is None or lap_time <= 0.0:
            lap_time = float(len(path_xy) - 1)

        time_axis = np.linspace(0.0, lap_time, len(path_xy))
        desired_xy = desired_from_arclength(waypoints, len(path_xy))

        # Point-to-point transitions are extracted from desired X and desired Y trajectories.
        x_segments = segment_transitions(desired_xy[:, 0], min_delta=1.0, min_samples=20)
        y_segments = segment_transitions(desired_xy[:, 1], min_delta=1.0, min_samples=20)

        settle_abs_band = args.settle_abs_band if args.settle_abs_band > 0.0 else None

        x_metrics = evaluate_step_response(
            desired_xy[:, 0],
            path_xy[:, 0],
            time_axis,
            x_segments,
            settle_tol=args.settle_tol,
            settle_window_ratio=args.settle_window_ratio,
            settle_min_samples=args.settle_min_samples,
            settle_abs_band=settle_abs_band,
        )
        y_metrics = evaluate_step_response(
            desired_xy[:, 1],
            path_xy[:, 1],
            time_axis,
            y_segments,
            settle_tol=args.settle_tol,
            settle_window_ratio=args.settle_window_ratio,
            settle_min_samples=args.settle_min_samples,
            settle_abs_band=settle_abs_band,
        )

        x_summary = summarize_transition_metrics(x_metrics)
        y_summary = summarize_transition_metrics(y_metrics)

        pos_err = np.linalg.norm(path_xy - desired_xy, axis=1)

        summary_rows.append(
            {
                "controller": controller_name,
                "lap": args.lap,
                "lap_time_s": float(lap_time),
                "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
                "position_mean_error_m": float(np.mean(pos_err)),
                "position_max_error_m": float(np.max(pos_err)),
                "x_num_transitions": x_summary["num_transitions"],
                "x_rise_time_mean_s": x_summary["rise_time_mean_s"],
                "x_overshoot_mean_pct": x_summary["overshoot_mean_pct"],
                "x_settling_time_mean_s": x_summary["settling_time_mean_s"],
                "y_num_transitions": y_summary["num_transitions"],
                "y_rise_time_mean_s": y_summary["rise_time_mean_s"],
                "y_overshoot_mean_pct": y_summary["overshoot_mean_pct"],
                "y_settling_time_mean_s": y_summary["settling_time_mean_s"],
            }
        )

        if not x_metrics.empty:
            x_metrics = x_metrics.copy()
            x_metrics.insert(0, "axis", "x")
            x_metrics.insert(0, "controller", controller_name)
            detail_frames.append(x_metrics)
        if not y_metrics.empty:
            y_metrics = y_metrics.copy()
            y_metrics.insert(0, "axis", "y")
            y_metrics.insert(0, "controller", controller_name)
            detail_frames.append(y_metrics)

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    summary_output = args.summary_output or os.path.join(base_path, f"point_to_point_summary_lap{args.lap}.csv")
    details_output = args.details_output or os.path.join(base_path, f"point_to_point_details_lap{args.lap}.csv")

    summary_df.to_csv(summary_output, index=False)
    details_df.to_csv(details_output, index=False)

    print(f"Saved summary: {summary_output}")
    print(f"Saved details: {details_output}")


if __name__ == "__main__":
    main()
