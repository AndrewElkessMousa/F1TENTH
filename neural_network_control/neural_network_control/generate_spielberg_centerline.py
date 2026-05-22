from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree


def load_map_metadata(map_yaml_path: Path):
    with map_yaml_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin = meta["origin"]
    image_name = meta["image"]
    image_path = (map_yaml_path.parent / image_name).resolve()
    return resolution, origin, image_path


def resample_closed_polyline(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    if np.linalg.norm(points_xy[0] - points_xy[-1]) > 1e-6:
        points_xy = np.vstack([points_xy, points_xy[0]])

    seg_len = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.linspace(0.0, arc[-1], num_points)

    x_new = np.interp(samples, arc, points_xy[:, 0])
    y_new = np.interp(samples, arc, points_xy[:, 1])
    return np.column_stack((x_new, y_new))


def save_debug_overlay(map_gray: np.ndarray, center_px: np.ndarray, output_png: Path):
    overlay = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)

    pts_i = np.round(center_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)

    start = tuple(np.round(center_px[0]).astype(np.int32))
    cv2.circle(overlay, start, 5, (255, 0, 0), -1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png), overlay)


def generate_spielberg_centerline_from_map(map_yaml_path: Path, output_csv: Path, num_points: int = 500):
    resolution, origin, image_path = load_map_metadata(map_yaml_path)

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read map image: {image_path}")

    # Track walls are dark in this map; threshold isolates them.
    walls = (img < 100).astype(np.uint8) * 255
    walls = cv2.dilate(walls, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(walls, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 4:
        raise RuntimeError("Expected at least 4 wall contours. Check map thresholding.")

    # For Spielberg this robustly picks the inner edge of the outer wall and
    # the outer edge of the inner wall, then computes their midpoint.
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    outer_inner = contours[1][:, 0, :].astype(np.float64)
    inner_outer = contours[2][:, 0, :].astype(np.float64)

    tree = cKDTree(inner_outer)
    _, nn_idx = tree.query(outer_inner, k=1)
    center_px = 0.5 * (outer_inner + inner_outer[nn_idx])

    # Remove near-duplicate successive points from contour sampling.
    keep = [0]
    for i in range(1, len(center_px)):
        if np.linalg.norm(center_px[i] - center_px[keep[-1]]) > 1.0:
            keep.append(i)
    center_px = center_px[keep]

    center_px = resample_closed_polyline(center_px, num_points=num_points)

    h = img.shape[0]
    wx = center_px[:, 0] * resolution + origin[0]
    wy = (h - 1 - center_px[:, 1]) * resolution + origin[1]

    out_df = pd.DataFrame({"x": wx, "y": wy})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    debug_png = output_csv.with_name("center_line_sp_overlay.png")
    save_debug_overlay(img, center_px, debug_png)

    print(f"Map YAML: {map_yaml_path}")
    print(f"Map image: {image_path}")
    print(f"Saved centerline: {output_csv}")
    print(f"Saved overlay: {debug_png}")
    print(f"Waypoints: {len(out_df)}")


if __name__ == "__main__":
    map_yaml = Path("/home/andrew/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map.yaml")
    out_csv = Path("/home/andrew/sim_ws/src/neural_network_control/neural_network_control/center_line_sp.csv")
    generate_spielberg_centerline_from_map(map_yaml, out_csv)
