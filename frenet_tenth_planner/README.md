# frenet_tenth_planner

A ready-to-use ROS 2 local planner package built from the logic of the FrenetTenth repository and adapted to your workflow:

- you already have a centerline
- track width is almost constant
- the package auto-generates left/right borders from the centerline
- it subscribes to odometry and publishes a Frenet local path

## What was reused from the original repo

This package keeps the original planning ideas and code structure:
- cubic spline centerline representation
- quintic lateral and quartic longitudinal trajectory generation
- Frenet candidate sampling and cost selection
- collision checking against synthetic border obstacles

It intentionally removes the CUDA-only and benchmark/demo parts.

## Package topics

Subscribes:
- `/ego_racecar/odom` by default

Publishes:
- `/frenet_path` (`nav_msgs/Path`)
- `/frenet_target_point` (`geometry_msgs/PointStamped`)
- `/frenet_debug_markers` (`visualization_msgs/MarkerArray`)

## Expected centerline file

CSV with 2 numeric columns:

```csv
x,y
0.0,0.0
1.0,0.0
2.0,0.2
```

Header row is optional.

## Build

Put the package in your ROS 2 workspace `src/` folder, then:

```bash
cd ~/sim_ws
colcon build --packages-select frenet_tenth_planner
source install/setup.bash
```

## Configure

Edit:

```bash
share/frenet_tenth_planner/config/planner_params.yaml
```

Set:
- `centerline_csv` to the absolute path of your centerline file
- `track_width` to your real track width in meters

## Run

```bash
ros2 launch frenet_tenth_planner frenet_planner.launch.py \
  params_file:=/absolute/path/to/planner_params.yaml
```

## RViz

Display these topics:
- `Path` -> `/frenet_path`
- `PointStamped` -> `/frenet_target_point`
- `MarkerArray` -> `/frenet_debug_markers`

## First integration notes

This package currently assumes:
- no dynamic obstacles yet
- odometry pose is already in the same `map` frame as your centerline
- lateral velocity and lateral acceleration start at zero

That is enough for a solid first F1TENTH trial.

## Files that matter most

- `src/frenet_planner_node.cpp` -> ROS 2 node wrapper
- `src/frenet_optimal_trajectory.cpp` -> Frenet planner core
- `src/cubic_spline_planner.cpp` -> spline and projection math
- `src/track_loader.cpp` -> loads centerline CSV and generates borders

## Known limitations

- I did not add obstacle subscriptions yet
- I did not add a controller node; this publishes the path and target point only
- This was prepared to be practical and self-contained, but I could not compile it here against your ROS 2 environment, so you may need tiny include/package-name adjustments depending on your distro
