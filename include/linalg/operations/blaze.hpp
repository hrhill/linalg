#ifndef LINALG_OPERATIONS_BLAZE_HPP_
#define LINALG_OPERATIONS_BLAZE_HPP_

#include <blaze/Math.h>

namespace linalg{

template <
    template <typename, bool> class V,
    typename T,
    bool SO>
std::size_t
size(const V<T, SO>& v)
{
    return v.size();
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
size_t
num_rows(const Matrix<T, SO>& m)
{
    return m.rows();
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
size_t
num_cols(const Matrix<T, SO>& m)
{
    return m.columns();
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
auto
column(Matrix<T, SO>& a, size_t idx) -> decltype(blaze::column(a, idx))
{
    return blaze::column(a, idx);
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
auto
row(Matrix<T, SO>& a, size_t idx) -> decltype(blaze::row(a, idx))
{
    return blaze::row(a, idx);
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
Matrix<T, SO>
trans(const Matrix<T, SO>& M)
{
    return blaze::trans(M);
}
/*
template<
    template <typename, bool> class Matrix1,
    typename T1, bool SO1,
    template <typename, bool> class Matrix2,
    typename T2, bool SO2>
void
swap(blaze::DenseRow<Matrix1<T1, SO1>>& r1, blaze::DenseRow<Matrix2<T2, SO2>>& r2)
{
    const int n = blaze::size(r1);
    std::vector<T1> tmp(n);
    for (int i = 0; i < n; ++i)
        tmp[i] = r1[i];
    for (int i = 0; i < n; ++i)
        r1[i] = r2[i];
    for (int i = 0; i < n; ++i)
        r2[i] = tmp[i];
}*/

/// \brief Inner product
template <typename T, bool TF1, bool TF2>
auto inner_prod(const blaze::DynamicVector<T, TF1>& x, const blaze::DynamicVector<T, TF2>& y)
{
    assert(linalg::size(x) == linalg::size(y));
    return std::inner_product(
                x.begin(), x.end(), y.begin(), T(0.0));
}

template <typename T, bool TF1, bool TF2>
blaze::DynamicMatrix<double>
outer_prod(const blaze::DynamicVector<T, TF1>& x, const blaze::DynamicVector<T, TF2>& y)
{
    blaze::DynamicMatrix<double> m(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < y.size(); ++j){
            m(i, j) = x[i] * y[j];
        }
    }
    return m;
}


/// \brief \f$ l_1 \f$ norm.
template <typename T, bool TF>
T norm_1(const blaze::DynamicVector<T, TF>& x)
{
    T nx(0.0);
    for (const auto& xi : x)
        nx += fabs(xi);
    return nx;
}

/// \brief \f$ l_2  \f$ norm.
template <typename T, bool TF>
T norm_2(const blaze::DynamicVector<T, TF>& x)
{
    return sqrt(inner_prod(x, x));
}

/// \brief \f$ l_p  \f$ norm.
template <typename T, bool TF>
T norm_p(const blaze::DynamicVector<T, TF>& x, int p)
{
    assert(p > 0);
    T r(0.0);
    for (const auto& xi : x)
        r += std::pow(xi, p);
    return exp(log(r)/p);
}

/// \brief \f$ l_{\infty} \f$ norm.
template <typename T, bool TF>
T norm_infinity(const blaze::DynamicVector<T, TF>& x)
{
    T r(0.0);
    for (const auto& xi : x){
        const T fxi = fabs(xi);
        if (fxi > r)
            r = fxi;
    }
    return r;
}


} // ns linalg

#endif

