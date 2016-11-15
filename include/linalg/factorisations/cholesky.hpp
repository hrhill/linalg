#ifndef LINALG_FACTORISATIONS_CHOLESKY_HPP_
#define LINALG_FACTORISATIONS_CHOLESKY_HPP_

#include "linalg/factorisations/tools.hpp"
#include "linalg/lapack.hpp"

namespace linalg
{
namespace factorisations
{

/// \brief User friendly Cholesky factorisation.
template <typename Matrix>
Matrix
cholesky(Matrix m)
{
    linalg::potrf(m);
    return tools::select_lower_triangular<Matrix>(m);
}
}
} // ns ook

#endif
