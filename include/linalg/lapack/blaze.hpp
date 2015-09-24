#ifndef LINALG_LAPACK_BLAZE_HPP_
#define LINALG_LAPACK_BLAZE_HPP_

#include <blaze/Math.h>
#include <boost/cast.hpp>

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
    typedef Matrix<T, SO> matrix_type;
    const int M(boost::numeric_cast<int>(a.rows()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dpotrf(order, CblasLower, M, a.data(), lda);
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
int
potrs(Matrix<T, SO>& a, Matrix<T, SO>& b)
{
    typedef Matrix<T, SO> matrix_type;
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    const int N(boost::numeric_cast<int>(a.rows()));
    const int NRHS(boost::numeric_cast<int>(b.columns()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    const int ldb(boost::numeric_cast<int>(b.spacing()));
    return clapack_dpotrs(order, CblasLower, N, NRHS, a.data(), lda, b.data(), ldb);
}

template <
    template <typename, bool> class Matrix,
    typename T,
    bool SO>
int
potri(Matrix<T, SO>& a)
{
    typedef Matrix<T, SO> matrix_type;
    const int M(boost::numeric_cast<int>(a.rows()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dpotri(order, CblasLower, M, a.data(), lda);
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
    typedef MatrixA<TA, SOA> matrix_type;
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    const int N(boost::numeric_cast<int>(a.rows()));
    const int NRHS(boost::numeric_cast<int>(b.columns()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    const int ldb(boost::numeric_cast<int>(b.spacing()));
    return clapack_dposv(order, CblasLower, N, NRHS, a.data(), lda, b.data(), ldb);
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getrf(Matrix<T, SO>& a, std::vector<int>& ipiv)
{
    typedef Matrix<T, SO> matrix_type;
    const int M(boost::numeric_cast<int>(a.rows()));
    const int N(boost::numeric_cast<int>(a.columns()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dgetrf(order, M, N, a.data(), lda, ipiv.data());
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getri(Matrix<T, SO>& a, std::vector<int>& ipiv)
{
    typedef Matrix<T, SO> matrix_type;
    const int M(boost::numeric_cast<int>(a.rows()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    auto order = blaze::IsRowMajorMatrix<matrix_type>::value ? CblasRowMajor : CblasColMajor;
    return clapack_dgetri(order, M, a.data(), lda, ipiv.data());
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
    const int N(boost::numeric_cast<int>(a.rows()));
    const int NRHS(boost::numeric_cast<int>(b.columns()));
    const int lda(boost::numeric_cast<int>(a.spacing()));
    const int ldb(boost::numeric_cast<int>(b.spacing()));
    return clapack_dgetrs(order, CblasNoTrans, N, NRHS, a.data(), lda, ipiv.data(), b.data(), ldb);
}

} // ns linalg

#endif
