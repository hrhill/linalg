#ifndef LINALG_LAPACK_BLAZE_HPP_
#define LINALG_LAPACK_BLAZE_HPP_

#include <blaze/Math.h>

extern "C"{
    #include <cblas.h>
    #include <clapack.h>
}

namespace linalg{

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
int
potrf(Matrix<T, SO>& a)
{
    const int n(a.rows());
    const int lda(a.spacing());
    int info = 0;
    blaze::potrf('L', n, a.data(), lda, &info);
    return info;
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
int
potrs(Matrix<T, SO>& a, Matrix<T, SO>& b)
{
    const int n(a.rows());
    const int nrhs(b.columns());
    const int lda(a.spacing());
    const int ldb(b.spacing());
    int info = 0;
    blaze::potrs('L', n, nrhs, a.data(), lda, b.data(), ldb, &info);
    return info;
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
int
potri(Matrix<T, SO>& a)
{
    const int m(a.rows());
    const int lda(a.spacing());
    int info = 0;
    blaze::potri('L', m, a.data(), lda, &info);
    return info;
}

template <
    template <typename, bool> class MatrixA,
    template <typename, bool> class MatrixB,
    typename TA,
    bool SOA,
    typename TB,
    bool SOB>
int
posv(MatrixA<TA, SOA>& a, MatrixB<TB, SOB>& b)
{
    const int n(a.rows());
    const int nrhs(b.columns());
    const int lda(a.spacing());
    const int ldb(b.spacing());
    int info = 0;
    blaze::posv('L', n, nrhs, a.data(), lda, b.data(), ldb, &info);
    return info;
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getrf(Matrix<T, SO>& a, std::vector<int>& ipiv)
{
    typedef Matrix<T, SO> matrix_type;
    const int m(static_cast<int>(a.rows()));
    const int n(static_cast<int>(a.columns()));
    const int lda(static_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dgetrf(order, m, n, a.data(), lda, ipiv.data());
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getri(Matrix<T, SO>& a, std::vector<int>& ipiv)
{
    typedef Matrix<T, SO> matrix_type;
    const int m(static_cast<int>(a.rows()));
    const int lda(static_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dgetri(order, m, a.data(), lda, ipiv.data());
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getrs(Matrix<T, SO>& a, std::vector<int>& ipiv, Matrix<T, SO>& b)
{
    typedef Matrix<T, SO> matrix_type;
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    //auto trans = blaze::IsTransExpr<matrix_type>::value ? CblasTrans : CblasNoTrans;
    const int n(static_cast<int>(a.rows()));
    const int nrhs(static_cast<int>(b.columns()));
    const int lda(static_cast<int>(a.spacing()));
    const int ldb(static_cast<int>(b.spacing()));
    return clapack_dgetrs(order, CblasNoTrans, n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb);
}

} // ns linalg

#endif
