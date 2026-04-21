#include "frenet_tenth_planner/track_loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace frenet_tenth_planner
{

namespace
{

void append_if_closed(std::vector<double> &x, std::vector<double> &y)
{
  if (x.size() < 3) {
    return;
  }
  const double dx = x.front() - x.back();
  const double dy = y.front() - y.back();
  if (std::hypot(dx, dy) > 1e-6) {
    x.push_back(x.front());
    y.push_back(y.front());
  }
}

}  // namespace

TrackData load_track_from_csv(const std::string &csv_path, double track_width, bool closed_loop)
{
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open centerline CSV: " + csv_path);
  }

  std::vector<double> cx, cy;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::replace(line.begin(), line.end(), ';', ',');
    std::stringstream ss(line);
    std::string sx, sy;
    if (!std::getline(ss, sx, ',')) {
      continue;
    }
    if (!std::getline(ss, sy, ',')) {
      continue;
    }
    try {
      cx.push_back(std::stod(sx));
      cy.push_back(std::stod(sy));
    } catch (const std::exception &) {
      continue;  // skip header rows
    }
  }

  if (cx.size() < 3) {
    throw std::runtime_error("Centerline CSV must contain at least 3 numeric x,y rows.");
  }

  if (closed_loop) {
    append_if_closed(cx, cy);
  }

  const double half_width = track_width / 2.0;
  TrackData track;
  track.center_x = cx;
  track.center_y = cy;
  for (size_t i = 0; i < cx.size(); ++i) {
    size_t prev = (i == 0) ? cx.size() - 2 : i - 1;
    size_t next = (i + 1 >= cx.size()) ? 1 : i + 1;
    double dx = cx[next] - cx[prev];
    double dy = cy[next] - cy[prev];
    double yaw = std::atan2(dy, dx);
    double nx = -std::sin(yaw);
    double ny = std::cos(yaw);

    track.left_x.push_back(cx[i] + half_width * nx);
    track.left_y.push_back(cy[i] + half_width * ny);
    track.right_x.push_back(cx[i] - half_width * nx);
    track.right_y.push_back(cy[i] - half_width * ny);
  }
  return track;
}

void save_track_yaml_template(const std::string &yaml_path)
{
  std::ofstream out(yaml_path);
  out << "# Example parameter file snippet\n"
         "frenet_planner_node:\n"
         "  ros__parameters:\n"
         "    centerline_csv: /absolute/path/to/centerline.csv\n"
         "    track_width: 1.8\n";
}

}  // namespace frenet_tenth_planner
