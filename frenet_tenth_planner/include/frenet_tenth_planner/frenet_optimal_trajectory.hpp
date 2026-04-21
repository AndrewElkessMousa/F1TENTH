#ifndef FRENET_TENTH_PLANNER_FRENET_OPTIMAL_TRAJECTORY_HPP_
#define FRENET_TENTH_PLANNER_FRENET_OPTIMAL_TRAJECTORY_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "frenet_tenth_planner/common_structs.hpp"
#include "frenet_tenth_planner/cubic_spline_planner.hpp"

namespace frenet_tenth_planner
{

template<class T>
struct Params
{
  T max_speed;
  T max_accel;
  T max_curvature;
  T max_road_width;
  T d_road_w;
  T dt;
  T maxt;
  T mint;
  T target_speed;
  T d_t_s;
  T n_s_sample;
  T robot_radius;
  T max_road_width_left;
  T max_road_width_right;
  T safe_distance;
  T range_path_check;
  T next_s_borders;
  T kj;
  T kt;
  T kd;
  T klat;
  T klon;
  bool check_derivatives;
};

template<class T>
class FrenetPlanner
{
public:
  FrenetPlanner() = default;
  explicit FrenetPlanner(Params<T> params) : params_(params) {}
  FrenetPlanner(Params<T> params, Spline2D reference_path, Spline2D i_border, Spline2D o_border)
  : params_(params), reference_path_(reference_path), i_border_(i_border), o_border_(o_border) {}

  FrenetPath<double> frenet_optimal_planning(
    double s0, double s_d, double c_d, double c_d_d, double c_d_dd,
    std::vector<Obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last,
    int overtake_strategy);

private:
  Params<T> params_{};
  Spline2D reference_path_;
  Spline2D i_border_;
  Spline2D o_border_;

  std::vector<FrenetPath<T>> calc_frenet_paths(T s_d, T c_d, T c_d_d, T c_d_dd, T s0);
  std::vector<FrenetPath<T>> calc_global_paths(std::vector<FrenetPath<T>> fplist);
  bool check_collision_path(FrenetPath<T> fp, std::vector<Obstacle> obs);
  std::vector<FrenetPath<T>> check_path(std::vector<FrenetPath<T>> fplist, std::vector<Obstacle> obs);
  bool check_derivatives(T s_d, T s_dd, T c);
};

extern template class FrenetPlanner<double>;

}  // namespace frenet_tenth_planner

#endif
