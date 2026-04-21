#ifndef FRENET_TENTH_PLANNER_POLYNOMIALS_HPP_
#define FRENET_TENTH_PLANNER_POLYNOMIALS_HPP_

#include <cmath>
#include <eigen3/Eigen/Dense>

namespace frenet_tenth_planner
{

using Eigen::MatrixXd;

template<class T>
class quintic
{
public:
  quintic(T xs_t, T vxs_t, T axs_t, T xe_t, T vxe_t, T axe_t, T Tm);
  T calc_point(T t);
  T calc_first_derivative(T t);
  T calc_second_derivative(T t);
  T calc_third_derivative(T t);

private:
  T xs{}, vxs{}, axs{}, xe{}, vxe{}, axe{};
  T a0{}, a1{}, a2{}, a3{}, a4{}, a5{};
};

template<class T>
class quartic
{
public:
  quartic(T xs_t, T vxs_t, T axs_t, T vxe_t, T axe_t, T Tm);
  T calc_point(T t);
  T calc_first_derivative(T t);
  T calc_second_derivative(T t);
  T calc_third_derivative(T t);

private:
  T xs{}, vxs{}, axs{}, vxe{}, axe{};
  T a0{}, a1{}, a2{}, a3{}, a4{};
};

extern template class quintic<double>;
extern template class quartic<double>;

}  // namespace frenet_tenth_planner

#endif
