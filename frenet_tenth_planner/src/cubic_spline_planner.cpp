#include "frenet_tenth_planner/cubic_spline_planner.hpp"

#include <algorithm>

namespace frenet_tenth_planner
{

void Spline::init(vecD x_in, vecD y_in)
{
  VectorXd xs = Map<VectorXd, Unaligned>(x_in.data(), x_in.size());
  VectorXd ys = Map<VectorXd, Unaligned>(y_in.data(), y_in.size());
  nx = static_cast<int>(x_in.size());

  VectorXd ax(nx), bx(nx - 1), cx(nx), dx(nx - 1);
  ax.setZero();
  bx.setZero();
  cx.setZero();
  dx.setZero();

  VectorXd mu(nx - 1), h(nx - 1), alpha(nx - 1), l(nx), z(nx);
  mu.setZero();
  h.setZero();
  alpha.setZero();
  l.setZero();
  z.setZero();

  ax = ys;
  for (int i = 0; i < nx - 1; ++i) {
    h(i) = xs(i + 1) - xs(i);
  }
  for (int i = 1; i < nx - 1; ++i) {
    alpha(i) = 3.0 / h(i) * (ax(i + 1) - ax(i)) - 3.0 / h(i - 1) * (ax(i) - ax(i - 1));
  }

  l(0) = 1.0;
  for (int i = 1; i < nx - 1; ++i) {
    l(i) = 2.0 * (xs(i + 1) - xs(i - 1)) - h(i - 1) * mu(i - 1);
    mu(i) = h(i) / l(i);
    z(i) = (alpha(i) - h(i - 1) * z(i - 1)) / l(i);
  }
  l(nx - 1) = 1.0;
  cx(nx - 1) = 0.0;

  for (int i = nx - 2; i >= 0; --i) {
    cx(i) = z(i) - mu(i) * cx(i + 1);
    bx(i) = (ax(i + 1) - ax(i)) / h(i) - (h(i) * (cx(i + 1) + 2.0 * cx(i))) / 3.0;
    dx(i) = (cx(i + 1) - cx(i)) / (3.0 * h(i));
  }

  a = vecD(&ax[0], ax.data() + ax.size());
  b = vecD(&bx[0], bx.data() + bx.size());
  c = vecD(&cx[0], cx.data() + cx.size());
  d = vecD(&dx[0], dx.data() + dx.size());
  x = vecD(&xs[0], xs.data() + xs.size());
  y = vecD(&ys[0], ys.data() + ys.size());
}

int Spline::search_index(double p)
{
  return static_cast<int>(std::upper_bound(x.begin(), x.end(), p) - x.begin() - 1);
}

double Spline::calc(double t)
{
  if (t < x[0] || t > x[nx - 1]) {
    return NONE;
  }
  int i = search_index(t);
  double dx = t - x[i];
  return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}

double Spline::calcd(double t)
{
  if (t < x[0] || t > x[nx - 1]) {
    return NONE;
  }
  int i = search_index(t);
  double dx = t - x[i];
  return b[i] + 2 * c[i] * dx + 3 * d[i] * dx * dx;
}

double Spline::calcdd(double t)
{
  if (t < x[0] || t > x[nx - 1]) {
    return NONE;
  }
  int i = search_index(t);
  double dx = t - x[i];
  return 2 * c[i] + 6 * d[i] * dx;
}

Spline2D::Spline2D(vecD x_in, vecD y_in)
: x_(std::move(x_in)), y_(std::move(y_in))
{
  s_ = calc_s(x_, y_);
  sx.init(s_, x_);
  sy.init(s_, y_);
}

vecD Spline2D::calc_s(vecD x, vecD y)
{
  vecD dx, dy;
  for (size_t i = 1; i < x.size(); ++i) {
    dx.push_back(x[i] - x[i - 1]);
    dy.push_back(y[i] - y[i - 1]);
  }
  ds_.clear();
  for (size_t i = 0; i < dx.size(); ++i) {
    ds_.push_back(std::sqrt(dx[i] * dx[i] + dy[i] * dy[i]));
  }
  vecD t{0.0};
  for (double dsi : ds_) {
    t.push_back(t.back() + dsi);
  }
  return t;
}

void Spline2D::calc_position(double *x, double *y, double t)
{
  double s_last = get_s_last();
  if (t <= 0.0) {
    t = 0.0;
  }
  while (t >= s_last) {
    t -= s_last;
  }
  *x = sx.calc(t);
  *y = sy.calc(t);
}

void Spline2D::calc_projection(double *s, double *d, double x, double y, double s_guess)
{
  double pos_x, pos_y;
  calc_position(&pos_x, &pos_y, s_guess);
  double s_opt = s_guess;
  double s_previous = s_opt;
  *s = s_guess;

  for (int i = 0; i < 30; ++i) {
    calc_position(&pos_x, &pos_y, s_opt);
    double dx = sx.calcd(s_opt);
    double ddx = sx.calcdd(s_opt);
    double dy = sy.calcd(s_opt);
    double ddy = sy.calcdd(s_opt);
    double diff_x = pos_x - x;
    double diff_y = pos_y - y;
    double jac = 2.0 * diff_x * dx + 2.0 * diff_y * dy;
    double hessian = 2.0 * dx * dx + 2.0 * diff_x * ddx + 2.0 * dy * dy + 2.0 * diff_y * ddy;
    if (std::abs(hessian) < 1e-9) {
      break;
    }
    s_opt -= jac / hessian;
    s_opt = s_opt - this->get_s_last() * std::floor(s_opt / this->get_s_last());
    if (std::abs(s_previous - s_opt) <= 1e-5) {
      *s = s_opt;
      break;
    }
    s_previous = s_opt;
    *s = s_opt;
  }

  calc_position(&pos_x, &pos_y, *s);
  double diff_x = pos_x - x;
  double diff_y = pos_y - y;
  double yaw = calc_yaw(*s);
  *d = -diff_x * std::sin(yaw) + diff_y * std::cos(yaw);
}

double Spline2D::calc_curvature(double t)
{
  double s_last = get_s_last();
  if (t <= 0.0) {
    t = 0.0;
  }
  while (t >= s_last) {
    t -= s_last;
  }
  double dx = sx.calcd(t);
  double ddx = sx.calcdd(t);
  double dy = sy.calcd(t);
  double ddy = sy.calcdd(t);
  return (ddy * dx - ddx * dy) / (dx * dx + dy * dy);
}

double Spline2D::calc_yaw(double t)
{
  double s_last = get_s_last();
  if (t <= 0.0) {
    t = 0.0;
  }
  while (t >= s_last) {
    t -= s_last;
  }
  double dx = sx.calcd(t);
  double dy = sy.calcd(t);
  if (dx == 0.0) {
    return 1.57 * (dy > 0);
  }
  return std::atan2(dy, dx);
}

double Spline2D::get_s_last()
{
  return s_.back();
}

Spline2D calc_spline_course(vecD x, vecD y, vecD &rx, vecD &ry, vecD &ryaw, vecD &rk, double ds)
{
  Spline2D sp(x, y);
  double s_range = sp.get_s_last();
  for (double s = 0.0; s < s_range; s += ds) {
    double ix, iy;
    sp.calc_position(&ix, &iy, s);
    rx.push_back(ix);
    ry.push_back(iy);
    ryaw.push_back(sp.calc_yaw(s));
    rk.push_back(sp.calc_curvature(s));
  }
  return sp;
}

}  // namespace frenet_tenth_planner
