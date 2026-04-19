import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_xy_and_time(csv_path, x_col='x', y_col='y'):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if x_col in df.columns and y_col in df.columns:
        xy = df[[x_col, y_col]].dropna().values
        lap_time = None
        if 'lap_time_s' in df.columns:
            time_series = pd.to_numeric(df['lap_time_s'], errors='coerce').dropna()
            if not time_series.empty:
                lap_time = float(time_series.iloc[0])
        return xy, lap_time
    return None, None


def load_step_response(csv_path):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    required = {'actual_x', 'actual_y', 'desired_x', 'desired_y'}
    if not required.issubset(df.columns):
        return None

    actual = df[['actual_x', 'actual_y']].apply(pd.to_numeric, errors='coerce').dropna()
    if actual.empty:
        return None

    desired_x = pd.to_numeric(df['desired_x'], errors='coerce').dropna()
    desired_y = pd.to_numeric(df['desired_y'], errors='coerce').dropna()
    if desired_x.empty or desired_y.empty:
        return None

    out = {
        'xy': actual[['actual_x', 'actual_y']].values,
        'target': np.array([float(desired_x.iloc[0]), float(desired_y.iloc[0])], dtype=float),
    }

    if 'time_s' in df.columns:
        t = pd.to_numeric(df['time_s'], errors='coerce')
        t = t.loc[actual.index].dropna()
        if not t.empty:
            out['time_s'] = t.values

    return out


def compute_step_overshoot_m(actual_xy, target_xy):
    if actual_xy is None or len(actual_xy) < 2:
        return 0.0

    start = actual_xy[0]
    approach = target_xy - start
    if float(np.linalg.norm(approach)) < 1e-9:
        return 0.0

    # Use dominant raw axis (x or y) to avoid any normalization while
    # still measuring crossing beyond the target in meters.
    if abs(approach[0]) >= abs(approach[1]):
        sign = 1.0 if approach[0] >= 0.0 else -1.0
        signed_delta = sign * (actual_xy[:, 0] - target_xy[0])
    else:
        sign = 1.0 if approach[1] >= 0.0 else -1.0
        signed_delta = sign * (actual_xy[:, 1] - target_xy[1])

    return float(max(0.0, np.max(signed_delta)))


def resolve_controller_csv(base_path, latest_name, lap, legacy_name=None):
    if lap is not None:
        stem, ext = os.path.splitext(latest_name)
        lap_candidate = os.path.join(base_path, f'{stem}_{lap}{ext}')
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


def load_lap_times(metrics_path, lap):
    if not os.path.exists(metrics_path):
        return {}

    df = pd.read_csv(metrics_path)
    required = {'controller', 'lap', 'lap_time_s'}
    if not required.issubset(df.columns):
        return {}

    df = df.dropna(subset=['controller', 'lap', 'lap_time_s'])
    df = df[df['lap'].astype(int) == int(lap)]

    lap_times = {}

    pp = df[df['controller'] == 'Pure_Pursuit']
    if not pp.empty:
        lap_times['Pure Pursuit'] = float(pp.iloc[-1]['lap_time_s'])

    pinn = df[df['controller'] == 'PINN']
    if not pinn.empty:
        lap_times['PINN Drive'] = float(pinn.iloc[-1]['lap_time_s'])

    nn = df[df['controller'] == 'PINN_v2']
    if not nn.empty:
        lap_times['NN AI Controller'] = float(nn.iloc[-1]['lap_time_s'])

    return lap_times


def with_lap_time(label, lap_times):
    if label in lap_times:
        return f"{label} ({lap_times[label]:.3f}s)"
    return label


def plot_lap_mode(ax, desired_closed, pure_pursuit, pinn, nn_ai, lap_times, lap):
    ax.plot(desired_closed[:, 0], desired_closed[:, 1], color='green', linewidth=2.5, label='Desired Path')
    ax.plot(pure_pursuit[:, 0], pure_pursuit[:, 1], color='blue', linewidth=2.0, label=with_lap_time('Pure Pursuit', lap_times))
    ax.plot(pinn[:, 0], pinn[:, 1], color='red', linewidth=2.0, label=with_lap_time('PINN Drive', lap_times))
    ax.plot(nn_ai[:, 0], nn_ai[:, 1], color='orange', linewidth=2.0, label=with_lap_time('NN AI Controller', lap_times))

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best')
    ax.set_title(f'Controller Path Comparison (Lap {lap})')


def plot_step_mode(ax, pp_step, pinn_step, nn_step):
    target = None
    for data in (pp_step, pinn_step, nn_step):
        if data is not None:
            target = data['target']
            break

    if target is None:
        raise FileNotFoundError('No valid step-response trajectories were found.')

    pp_overshoot = compute_step_overshoot_m(pp_step['xy'], target)
    pinn_overshoot = compute_step_overshoot_m(pinn_step['xy'], target)
    nn_overshoot = compute_step_overshoot_m(nn_step['xy'], target)

    ax.plot(pp_step['xy'][:, 0], pp_step['xy'][:, 1], color='blue', linewidth=2.0,
            label=f'Pure Pursuit (overshoot {pp_overshoot:.3f} m)')
    ax.plot(pinn_step['xy'][:, 0], pinn_step['xy'][:, 1], color='red', linewidth=2.0,
            label=f'PINN Drive (overshoot {pinn_overshoot:.3f} m)')
    ax.plot(nn_step['xy'][:, 0], nn_step['xy'][:, 1], color='orange', linewidth=2.0,
            label=f'NN AI Controller (overshoot {nn_overshoot:.3f} m)')

    ax.scatter([target[0]], [target[1]], marker='*', s=220, color='green', edgecolors='black',
               linewidths=0.8, label='Target')
    ax.scatter([pp_step['xy'][0, 0], pinn_step['xy'][0, 0], nn_step['xy'][0, 0]],
               [pp_step['xy'][0, 1], pinn_step['xy'][0, 1], nn_step['xy'][0, 1]],
               marker='o', s=45, color='black', alpha=0.55, label='Start')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best')
    ax.set_title('Single-Target Step Response Trajectories')


def main():
    parser = argparse.ArgumentParser(
        description='Create one PNG overlaying desired path and actual paths from three controllers.'
    )
    parser.add_argument(
        '--base-path',
        default=os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/'),
        help='Directory containing waypoints.csv and controller trajectory CSV files.'
    )
    parser.add_argument(
        '--output',
        default='controller_path_comparison.png',
        help='Output PNG file name (saved under base-path unless absolute path is used).'
    )
    parser.add_argument(
        '--lap',
        type=int,
        default=2,
        help='Lap number to plot for each controller. Default: 2.'
    )
    parser.add_argument(
        '--mode',
        choices=['auto', 'lap', 'step'],
        default='auto',
        help='Plot mode: lap (closed-track lap CSVs), step (single-target step-response CSVs), or auto.'
    )
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    waypoints_path = os.path.join(base_path, 'waypoints.csv')
    metrics_path = os.path.join(base_path, 'performance_metrics.csv')
    pp_xy_path = resolve_controller_csv(
        base_path,
        latest_name='pure_pursuit_reference_lap.csv',
        lap=args.lap,
        legacy_name='pp_reference_lap.csv'
    )
    pinn_path = resolve_controller_csv(
        base_path,
        latest_name='pinn_reference_lap.csv',
        lap=args.lap
    )
    nn_path = resolve_controller_csv(
        base_path,
        latest_name='nn_ai_reference_lap.csv',
        lap=args.lap
    )

    pp_step_path = os.path.join(base_path, 'pure_pursuit_step_response.csv')
    pinn_step_path = os.path.join(base_path, 'pinn_drive_step_response.csv')
    nn_step_path = os.path.join(base_path, 'nn_ai_step_response.csv')

    if args.mode == 'auto':
        if all(os.path.exists(p) for p in (pp_step_path, pinn_step_path, nn_step_path)):
            mode = 'step'
        else:
            mode = 'lap'
    else:
        mode = args.mode

    if not os.path.exists(waypoints_path):
        raise FileNotFoundError(f'Missing desired path file: {waypoints_path}')

    waypoints_df = pd.read_csv(waypoints_path)
    if 'x' not in waypoints_df.columns or 'y' not in waypoints_df.columns:
        raise ValueError('waypoints.csv must contain x and y columns')
    desired = waypoints_df[['x', 'y']].dropna().values

    fig, ax = plt.subplots(figsize=(12, 6))

    if mode == 'step':
        pp_step = load_step_response(pp_step_path)
        pinn_step = load_step_response(pinn_step_path)
        nn_step = load_step_response(nn_step_path)

        missing = []
        if pp_step is None:
            missing.append('pure_pursuit_step_response.csv')
        if pinn_step is None:
            missing.append('pinn_drive_step_response.csv')
        if nn_step is None:
            missing.append('nn_ai_step_response.csv')
        if missing:
            msg = 'Missing/invalid step-response CSVs:\n' + '\n'.join(missing)
            msg += '\n\nRun each controller in single-target mode to generate step-response logs, then re-run this script.'
            raise FileNotFoundError(msg)

        plot_step_mode(ax, pp_step, pinn_step, nn_step)
    else:
        pure_pursuit, pp_time = None, None
        if pp_xy_path is not None:
            pure_pursuit, pp_time = load_xy_and_time(pp_xy_path)
            if pure_pursuit is None and pp_xy_path.endswith('pp_reference_lap.csv'):
                pure_pursuit, pp_time = load_xy_and_time(pp_xy_path, x_col='pp_x', y_col='pp_y')

        pinn, pinn_time = load_xy_and_time(pinn_path) if pinn_path is not None else (None, None)
        nn_ai, nn_time = load_xy_and_time(nn_path) if nn_path is not None else (None, None)
        lap_times = load_lap_times(metrics_path, args.lap)
        if pp_time is not None:
            lap_times['Pure Pursuit'] = pp_time
        if pinn_time is not None:
            lap_times['PINN Drive'] = pinn_time
        if nn_time is not None:
            lap_times['NN AI Controller'] = nn_time

        missing = []
        if pure_pursuit is None:
            missing.append(f'Pure Pursuit lap {args.lap} trajectory CSV')
        if pinn is None:
            missing.append(f'PINN lap {args.lap} trajectory CSV')
        if nn_ai is None:
            missing.append(f'NN AI lap {args.lap} trajectory CSV')

        if missing:
            msg = 'Missing trajectory CSVs for:\n' + '\n'.join(missing)
            msg += '\n\nRun each controller until lap 2 is completed, then re-run this script.'
            raise FileNotFoundError(msg)

        desired_closed = pd.concat([waypoints_df[['x', 'y']], waypoints_df[['x', 'y']].iloc[[0]]], ignore_index=True).values
        plot_lap_mode(ax, desired_closed, pure_pursuit, pinn, nn_ai, lap_times, args.lap)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_path, output_path)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
