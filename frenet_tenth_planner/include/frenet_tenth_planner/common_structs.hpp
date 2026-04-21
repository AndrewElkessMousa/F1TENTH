#ifndef FRENET_TENTH_PLANNER_COMMON_STRUCTS_HPP_
#define FRENET_TENTH_PLANNER_COMMON_STRUCTS_HPP_

#include <vector>

namespace frenet_tenth_planner
{

template<typename T>
struct FrenetPath
{
  std::vector<T> t;
  std::vector<T> d;
  std::vector<T> d_d;
  std::vector<T> d_dd;
  std::vector<T> d_ddd;

  std::vector<T> s;
  std::vector<T> s_d;
  std::vector<T> s_dd;
  std::vector<T> s_ddd;

  std::vector<T> x;
  std::vector<T> y;
  std::vector<T> yaw;
  std::vector<T> ds;
  std::vector<T> c;

  T cd{0};
  T cv{0};
  T cf{0};
  bool empty{true};
};

struct Obstacle
{
  double x{0.0};
  double y{0.0};
  double radius{0.3};
  double s{0.0};
  double d{0.0};
};

}  // namespace frenet_tenth_planner

#endif
