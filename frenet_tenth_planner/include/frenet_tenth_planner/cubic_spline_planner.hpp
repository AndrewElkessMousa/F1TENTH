#ifndef FRENET_TENTH_PLANNER_CUBIC_SPLINE_PLANNER_HPP_
#define FRENET_TENTH_PLANNER_CUBIC_SPLINE_PLANNER_HPP_

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

namespace frenet_tenth_planner
{

constexpr double NONE = -1e9;
using vecD = std::vector<double>;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Unaligned;
using Eigen::VectorXd;

class Spline
{
public:
  void init(vecD x_in, vecD y_in);
  double calc(double t);
  double calcd(double t);
  double calcdd(double t);

private:
  int search_index(double p);

  int nx{0};
  vecD a, b, c, d, w;
  vecD x, y;
};

class Spline2D
{
public:
  Spline2D() = default;
  Spline2D(vecD x_in, vecD y_in);

  vecD calc_s(vecD x, vecD y);
  void calc_position(double *x, double *y, double t);
  void calc_projection(double *s, double *d, double x, double y, double s_guess);
  double calc_curvature(double t);
  double calc_yaw(double t);
  double get_s_last();

private:
  vecD x_;
  vecD y_;
  vecD s_;
  vecD ds_;

public:
  Spline sx;
  Spline sy;
};

Spline2D calc_spline_course(vecD x, vecD y, vecD &rx, vecD &ry, vecD &ryaw, vecD &rk, double ds);

}  // namespace frenet_tenth_planner

#endif
