# ifndef LINALG_LINALG_HPP_
# define LINALG_LINALG_HPP_

#include <vector>
#include <stdexcept>

#include <boost/cast.hpp>

#include "linalg/blas.hpp"
#include "linalg/lapack.hpp"
#include "linalg/operations.hpp"
#include "linalg/traits.hpp"

#include "linalg/norms.hpp"
#include "linalg/factorisations/cholesky.hpp"
#include "linalg/factorisations/gmw81.hpp"
#include "linalg/factorisations/ldlt.hpp"

#include "linalg/special_matrices.hpp"
#include "linalg/determinants.hpp"

namespace linalg{

// select the lower triangular part of the matrix.
template <typename Matrix>
Matrix
select_lower_triangular(const Matrix& m){

    const int nrows = linalg::num_rows(m);
    const int ncols = linalg::num_cols(m);

    const int dim = std::min(nrows, ncols);
    const int row_offset = std::max(0, nrows - ncols);
    Matrix lower(dim, dim, 0.0);

    for (int i = 0; i < dim; ++i){
        for (int j = 0; j <= i; ++j){
            lower(i, j) = m(row_offset + i, j);
        }
    }
    return lower;
}

// select the upper triangular part of the matrix.
template <typename Matrix>
Matrix
select_upper_triangular(const Matrix& m){

    const int nrows = linalg::num_rows(m);
    const int ncols = linalg::num_cols(m);

    const int dim = std::min(nrows, ncols);
    const int col_offset = std::max(0, ncols - nrows);
    Matrix upper(dim, dim, 0.0);

    for (int i = 0; i < dim; ++i){
        for (int j = i; j < dim; ++j){
            upper(i, j) = m(i, col_offset + j);
        }
    }
    return upper;
}

template <typename Matrix>
Matrix make_symmetric(Matrix a)
{
    mksym(a);
    return a;
}

template <typename MatrixType, typename VectorType>
VectorType
cholesky_solve(MatrixType a, const VectorType& y)
{
    // set up Ax = y
    MatrixType y1(linalg::size(y), 1, 0);
    column(y1, 0) = y;

    // solve
    posv(a, y1);

    return linalg::column(y1, 0);
}

template <typename MatrixType, typename VectorType>
VectorType
lu_solve(MatrixType a, const VectorType& y)
{
    const int n = linalg::size(y);
    MatrixType y1(n, 1, 0);
    column(y1, 0) = y;
    std::vector<int> ipiv(n, 0);
    int info = getrf(a, ipiv);
    if (info != 0){
        throw std::runtime_error(
            "getrf failed in lu_solve " + std::to_string(info));
    }

    getrs(a, ipiv, y1);
    return linalg::column(y1, 0);

}

template <typename MatrixType>
MatrixType
cholesky_invert(MatrixType a)
{
    // factorize
    int info = potrf(a);
    if (info != 0){
        throw std::runtime_error(
            "potrf failed in cholesky_invert " + std::to_string(info));
    }

    potri(a);
    return make_symmetric(a);
}

template <typename MatrixType>
MatrixType
lu_invert(MatrixType a)
{
    const int n = linalg::num_rows(a);
    std::vector<int> ipiv(n, 0);
    int info = getrf(a, ipiv);
    if (info != 0){
        throw std::runtime_error(
            "getrf failed in lu_invert " + std::to_string(info));
    }

    info = getri(a, ipiv);
    if (info != 0){
        throw std::runtime_error(
            "getri failed in lu_invert " + std::to_string(info));
    }
    return a;
}


template <typename MatrixType>
double
log_cholesky_determinant(MatrixType a){

    // factorize
    int info = potrf(a);
    if (info == -1)
        throw std::runtime_error(
            "potrf failed in log_cholesky_determinant" + std::to_string(info));
    return log(potrfdet(a));
}

template <typename MatrixType>
double
cholesky_determinant(MatrixType A){

    return  exp(log_cholesky_determinant(A));
}

template <typename MatrixType>
double
lu_determinant(MatrixType a){

    // factorize
    const int n = linalg::num_rows(a);
    std::vector<int> ipiv(n, 0);

    int info = getrf(a, ipiv);
    if (info == -1)
        throw std::runtime_error(
            "getrf failed in lu_determinant" + std::to_string(info));

    return getrfdet(a, ipiv);
}


} // ns linalg

#endif
