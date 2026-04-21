#include "frenet_tenth_planner/frenet_optimal_trajectory.hpp"
#include "frenet_tenth_planner/polynomials.hpp"

#include <cfloat>
#include <cmath>
#include <numeric>

namespace frenet_tenth_planner
{

template<class T>
static T dist(T x1, T y1, T x2, T y2)
{
  return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

template<class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::calc_frenet_paths(
  T s_d, T c_d, T c_d_d, T c_d_dd, T s0)
{
  std::vector<FrenetPath<T>> frenet_paths;

  for (T di = -params_.max_road_width_right; di <= params_.max_road_width_left; di += params_.d_road_w) {
    for (T Ti = params_.mint; Ti <= params_.maxt; Ti += params_.dt) {
      FrenetPath<T> fp;

      quintic<T> lat_qp(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti);

      for (T t = 0.0; t <= Ti + params_.dt; t += params_.dt) {
        fp.t.push_back(t);
        fp.d.push_back(lat_qp.calc_point(t));
        fp.d_d.push_back(lat_qp.calc_first_derivative(t));
        fp.d_dd.push_back(lat_qp.calc_second_derivative(t));
        fp.d_ddd.push_back(lat_qp.calc_third_derivative(t));
      }

      T Jp = std::inner_product(
        fp.d_ddd.begin(), fp.d_ddd.end(),
        fp.d_ddd.begin(), static_cast<T>(0));

      T minV = params_.target_speed - params_.d_t_s * params_.n_s_sample;
      T maxV = params_.target_speed + params_.d_t_s * params_.n_s_sample;

      for (T tv = minV; tv <= maxV + params_.d_t_s; tv += params_.d_t_s) {
        FrenetPath<T> tfp = fp;

        quartic<T> lon_qp(s0, s_d, 0.0, tv, 0.0, Ti);

        for (const auto &t : fp.t) {
          tfp.s.push_back(lon_qp.calc_point(t));
          tfp.s_d.push_back(lon_qp.calc_first_derivative(t));
          tfp.s_dd.push_back(lon_qp.calc_second_derivative(t));
          tfp.s_ddd.push_back(lon_qp.calc_third_derivative(t));
        }

        T Js = std::inner_product(
          tfp.s_ddd.begin(), tfp.s_ddd.end(),
          tfp.s_ddd.begin(), static_cast<T>(0));

        T ds = std::pow(params_.target_speed - tfp.s_d.back(), 2);

        tfp.cd = params_.kj * Jp + params_.kt * Ti + params_.kd * tfp.d.back() * tfp.d.back();
        tfp.cv = params_.kj * Js + params_.kt * Ti + params_.kd * ds;
        tfp.cf = params_.klat * tfp.cd + params_.klon * tfp.cv;

        frenet_paths.push_back(tfp);
      }
    }
  }

  return frenet_paths;
}

template<class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::calc_global_paths(std::vector<FrenetPath<T>> fplist)
{
  for (auto &fp : fplist) {
    for (size_t i = 0; i < fp.s.size(); ++i) {
      double ix_d, iy_d;
      reference_path_.calc_position(&ix_d, &iy_d, fp.s[i]);

      T ix = static_cast<T>(ix_d);
      T iy = static_cast<T>(iy_d);

      if (ix == NONE) {
        break;
      }

      T iyaw = static_cast<T>(reference_path_.calc_yaw(fp.s[i]));
      T di = fp.d[i];

      fp.x.push_back(ix - di * std::sin(iyaw));
      fp.y.push_back(iy + di * std::cos(iyaw));
    }

    if (fp.x.size() < 2) {
      continue;
    }

    for (size_t i = 0; i + 1 < fp.x.size(); ++i) {
      T dx = fp.x[i + 1] - fp.x[i];
      T dy = fp.y[i + 1] - fp.y[i];
      fp.yaw.push_back(std::atan2(dy, dx));
      fp.ds.push_back(std::sqrt(dx * dx + dy * dy));
    }

    if (!fp.yaw.empty()) {
      fp.yaw.push_back(fp.yaw.back());
    }
    if (!fp.ds.empty()) {
      fp.ds.push_back(fp.ds.back());
    }

    for (size_t i = 0; i + 1 < fp.yaw.size(); ++i) {
      fp.c.push_back((fp.yaw[i + 1] - fp.yaw[i]) / std::max(fp.ds[i], static_cast<T>(1e-6)));
    }

    if (!fp.c.empty()) {
      fp.c.push_back(fp.c.back());
    } else {
      fp.c.push_back(0.0);
    }
  }

  return fplist;
}

template<class T>
bool FrenetPlanner<T>::check_derivatives(T s_d, T s_dd, T c)
{
  return std::abs(s_d) > params_.max_speed ||
         std::abs(s_dd) > params_.max_accel ||
         std::abs(c) > params_.max_curvature;
}

template<class T>
bool FrenetPlanner<T>::check_collision_path(FrenetPath<T> fp, std::vector<Obstacle> obs)
{
  if (fp.x.empty()) {
    return true;
  }

  size_t limit = static_cast<size_t>(std::max<T>(1, fp.x.size() * params_.range_path_check));
  limit = std::min(limit, fp.x.size());

  for (const auto &ob : obs) {
    for (size_t j = 0; j < limit; ++j) {
      T di = dist(
        fp.x[j], fp.y[j],
        static_cast<T>(ob.x), static_cast<T>(ob.y)) -
        static_cast<T>(ob.radius);

      if (di < params_.safe_distance) {
        return true;
      }

      if (params_.check_derivatives && j < fp.c.size() &&
        check_derivatives(fp.s_d[j], fp.s_dd[j], fp.c[j]))
      {
        return true;
      }
    }
  }

  return false;
}

template<class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::check_path(
  std::vector<FrenetPath<T>> fplist, std::vector<Obstacle> obs)
{
  std::vector<FrenetPath<T>> out;
  for (auto &fp : fplist) {
    if (!check_collision_path(fp, obs)) {
      out.push_back(fp);
    }
  }
  return out;
}

template<class T>
static FrenetPath<double> to_double_path(const FrenetPath<T> &path)
{
  FrenetPath<double> fp;
  fp.empty = path.empty;
  fp.cd = path.cd;
  fp.cv = path.cv;
  fp.cf = path.cf;
  fp.t.assign(path.t.begin(), path.t.end());
  fp.d.assign(path.d.begin(), path.d.end());
  fp.d_d.assign(path.d_d.begin(), path.d_d.end());
  fp.d_dd.assign(path.d_dd.begin(), path.d_dd.end());
  fp.d_ddd.assign(path.d_ddd.begin(), path.d_ddd.end());
  fp.s.assign(path.s.begin(), path.s.end());
  fp.s_d.assign(path.s_d.begin(), path.s_d.end());
  fp.s_dd.assign(path.s_dd.begin(), path.s_dd.end());
  fp.s_ddd.assign(path.s_ddd.begin(), path.s_ddd.end());
  fp.x.assign(path.x.begin(), path.x.end());
  fp.y.assign(path.y.begin(), path.y.end());
  fp.yaw.assign(path.yaw.begin(), path.yaw.end());
  fp.ds.assign(path.ds.begin(), path.ds.end());
  fp.c.assign(path.c.begin(), path.c.end());
  return fp;
}

template<class T>
FrenetPath<double> FrenetPlanner<T>::frenet_optimal_planning(
  double s0, double s_d, double c_d, double c_d_d, double c_d_dd,
  std::vector<Obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last,
  int overtake_strategy)
{
  switch (overtake_strategy) {
    case 1:
      params_.max_road_width_left = params_.max_road_width;
      params_.max_road_width_right = 0.0;
      break;
    case 2:
      params_.max_road_width_left = 0.0;
      params_.max_road_width_right = params_.max_road_width;
      break;
    default:
      params_.max_road_width_left = params_.max_road_width;
      params_.max_road_width_right = params_.max_road_width;
      break;
  }

  auto fplist = calc_frenet_paths(
    static_cast<T>(s_d),
    static_cast<T>(c_d),
    static_cast<T>(c_d_d),
    static_cast<T>(c_d_dd),
    static_cast<T>(s0));

  fplist = calc_global_paths(fplist);

  if (fplist.empty()) {
    return FrenetPath<double>();
  }

  first = to_double_path(fplist.front());
  last = to_double_path(fplist.back());

  // Add border samples as small obstacles ahead of the car.
  // The original version used radius=1.0 and triangular s stepping,
  // which made almost all paths invalid on narrow tracks.
  const double border_step = 0.20;
  const double border_radius = 0.08;

  for (int i = 0; i < static_cast<int>(params_.next_s_borders); ++i) {
    double s_i = s0 + i * border_step;
    double s_o = s0 + i * border_step;

    const double s_i_last = i_border_.get_s_last();
    const double s_o_last = o_border_.get_s_last();

    while (s_i > s_i_last) {
      s_i -= s_i_last;
    }
    while (s_o > s_o_last) {
      s_o -= s_o_last;
    }

    double x_i, y_i, x_o, y_o;
    i_border_.calc_position(&x_i, &y_i, s_i);
    o_border_.calc_position(&x_o, &y_o, s_o);

    obstacles.push_back(Obstacle{x_i, y_i, border_radius, s_i, 0.0});
    obstacles.push_back(Obstacle{x_o, y_o, border_radius, s_o, 0.0});
  }

  fplist = check_path(fplist, obstacles);

  double min_cost = DBL_MAX;
  int best_index = -1;

  for (size_t i = 0; i < fplist.size(); ++i) {
    if (fplist[i].cf <= min_cost) {
      min_cost = fplist[i].cf;
      best_index = static_cast<int>(i);
    }
  }

  FrenetPath<double> best_path;
  if (best_index != -1) {
    best_path = to_double_path(fplist[best_index]);
    best_path.empty = false;
  } else {
    best_path.empty = true;
  }

  return best_path;
}

template class FrenetPlanner<double>;

}  // namespace frenet_tenth_planner