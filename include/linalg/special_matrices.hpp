#ifndef LINALG_SPECIAL_MATRICES_HPP_
#define LINALG_SPECIAL_MATRICES_HPP_

#include "linalg/special_matrices/identity_matrix.hpp"
#include "linalg/special_matrices/diagonal_matrix.hpp"
#include "linalg/special_matrices/constant_diagonal_matrix.hpp"

namespace linalg{

identity_matrix
eye(const int n)
{
    return identity_matrix(n);
}

} // ns linalg

#endif
