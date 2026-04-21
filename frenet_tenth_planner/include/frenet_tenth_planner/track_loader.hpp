#ifndef FRENET_TENTH_PLANNER_TRACK_LOADER_HPP_
#define FRENET_TENTH_PLANNER_TRACK_LOADER_HPP_

#include <string>
#include <vector>

namespace frenet_tenth_planner
{

struct TrackData
{
  std::vector<double> center_x;
  std::vector<double> center_y;
  std::vector<double> left_x;
  std::vector<double> left_y;
  std::vector<double> right_x;
  std::vector<double> right_y;
};

TrackData load_track_from_csv(const std::string &csv_path, double track_width, bool closed_loop = true);
void save_track_yaml_template(const std::string &yaml_path);

}  // namespace frenet_tenth_planner

#endif
