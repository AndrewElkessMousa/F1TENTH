#include "frenet_tenth_planner/polynomials.hpp"

namespace frenet_tenth_planner
{

template<class C>
quintic<C>::quintic(C xs_t, C vxs_t, C axs_t, C xe_t, C vxe_t, C axe_t, C Tm)
: xs(xs_t), vxs(vxs_t), axs(axs_t), xe(xe_t), vxe(vxe_t), axe(axe_t)
{
  a0 = xs;
  a1 = vxs;
  a2 = axs / 2.0;

  MatrixXd A(3, 3), B(3, 1);
  A << std::pow(Tm, 3), std::pow(Tm, 4), std::pow(Tm, 5),
       3 * std::pow(Tm, 2), 4 * std::pow(Tm, 3), 5 * std::pow(Tm, 4),
       6 * Tm, 12 * Tm * Tm, 20 * std::pow(Tm, 3);
  B << xe - a0 - a1 * Tm - a2 * Tm * Tm,
       vxe - a1 - 2 * a2 * Tm,
       axe - 2 * a2;
  MatrixXd X = A.inverse() * B;
  a3 = static_cast<C>(X(0, 0));
  a4 = static_cast<C>(X(1, 0));
  a5 = static_cast<C>(X(2, 0));
}

template<class C>
C quintic<C>::calc_point(C t)
{
  return a0 + a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
}

template<class C>
C quintic<C>::calc_first_derivative(C t)
{
  return a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t * t * t + 5 * a5 * t * t * t * t;
}

template<class C>
C quintic<C>::calc_second_derivative(C t)
{
  return 2 * a2 + 6 * a3 * t + 12 * a4 * t * t + 20 * a5 * t * t * t;
}

template<class C>
C quintic<C>::calc_third_derivative(C t)
{
  return 6 * a3 + 24 * a4 * t + 60 * a5 * t * t;
}

template<class C>
quartic<C>::quartic(C xs_t, C vxs_t, C axs_t, C vxe_t, C axe_t, C Tm)
: xs(xs_t), vxs(vxs_t), axs(axs_t), vxe(vxe_t), axe(axe_t)
{
  a0 = xs;
  a1 = vxs;
  a2 = axs / 2.0;

  MatrixXd A(2, 2), B(2, 1);
  A << 3 * std::pow(Tm, 2), 4 * std::pow(Tm, 3),
       6 * Tm, 12 * Tm * Tm;
  B << vxe - a1 - 2 * a2 * Tm,
       axe - 2 * a2;
  MatrixXd X = A.inverse() * B;
  a3 = static_cast<C>(X(0, 0));
  a4 = static_cast<C>(X(1, 0));
}

template<class C>
C quartic<C>::calc_point(C t)
{
  return a0 + a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t;
}

template<class C>
C quartic<C>::calc_first_derivative(C t)
{
  return a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t * t * t;
}

template<class C>
C quartic<C>::calc_second_derivative(C t)
{
  return 2 * a2 + 6 * a3 * t + 12 * a4 * t * t;
}

template<class C>
C quartic<C>::calc_third_derivative(C t)
{
  return 6 * a3 + 24 * a4 * t;
}

template class quintic<double>;
template class quartic<double>;

}  // namespace frenet_tenth_planner
