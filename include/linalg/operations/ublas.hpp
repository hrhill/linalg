#ifndef LINALG_OPERATIONS_UBLAS_HPP_
#define LINALG_OPERATIONS_UBLAS_HPP_

#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace linalg{

template <typename Exp>
size_t
size(const boost::numeric::ublas::vector_expression<Exp>& x)
{
    return x().size();
}

template <typename MatrixExprT>
size_t
num_rows(const boost::numeric::ublas::matrix_expression<MatrixExprT>& me)
{
    return me().size1();
}

template <typename MatrixExprT>
size_t
num_cols(const boost::numeric::ublas::matrix_expression<MatrixExprT>& me)
{
    return me().size2();
}


template <typename Matrix>
auto
column(Matrix& a, size_t idx) -> decltype(boost::numeric::ublas::column(a, idx))
{
    return boost::numeric::ublas::column(a, idx);
}

template <typename Matrix>
auto
row(Matrix& a, size_t idx) -> decltype(boost::numeric::ublas::row(a, idx))
{
    return boost::numeric::ublas::row(a, idx);
}

template <typename MatrixType>
inline
MatrixType
trans(const MatrixType& M)
{
	return boost::numeric::ublas::trans(M);
}

template <typename T>
auto
outer_prod(const boost::numeric::ublas::vector_expression<T>& x,
           const boost::numeric::ublas::vector_expression<T>& y)
    -> decltype(boost::numeric::ublas::outer_prod(x(), y()))
{
    return boost::numeric::ublas::outer_prod(x(), y());
}

template <typename T, typename U>
void swap(T& t, U& u){
    std::swap(t, u);
}

/// \brief Inner product
template <typename T>
double inner_prod(const boost::numeric::ublas::vector_expression<T>& x,
                const boost::numeric::ublas::vector_expression<T>& y)
{
    assert(linalg::size(x()) == linalg::size(y()));
    return std::inner_product(
                x().begin(), x().end(), y().begin(), 0.0);
}

/// \brief \f$ l_1 \f$ norm.
template <typename T>
double norm_1(const boost::numeric::ublas::vector_expression<T>& x)
{
    double nx(0.0);
    for (const auto& xi : x())
        nx += fabs(xi);
    return nx;
}

/// \brief \f$ l_2  \f$ norm.
template <typename T>
double norm_2(const boost::numeric::ublas::vector_expression<T>& x)
{
    return sqrt(inner_prod(x(), x()));
}

/// \brief \f$ l_p  \f$ norm.
template <typename T>
double norm_p(const boost::numeric::ublas::vector_expression<T>& x, int p)
{
    double nx(0.0);
    for (const auto& xi : x())
        nx += std::pow(xi, p);
    return exp(log(nx)/p);
}

/// \brief \f$ l_{\infty} \f$ norm.
template <typename T>
double norm_infinity(const boost::numeric::ublas::vector_expression<T>& x)
{
    double nx(0.0);
    for (const auto& xi : x()){
        auto fxi = fabs(xi);
        if (fxi > nx)
            nx = fxi;
    }
    return nx;
}

} // ns linalg

#endif
