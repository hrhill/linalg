#ifndef LINALG_LAPACK_BLAZE_HPP_
#define LINALG_LAPACK_BLAZE_HPP_

#include <blaze/Math.h>

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
    const int m(a.rows());
    const int n(a.columns());
    const int lda(a.spacing());
    int info = 0;
    blaze::getrf(m, n, a.data(), lda, ipiv.data(), &info);
    return info;
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getri(Matrix<T, SO>& a, std::vector<int>& ipiv)
{
    const int m(a.rows());
    const int lda(a.spacing());
    // Calculate workspace first
    std::vector<double> work{0};
    int info = 0;
    blaze::getri(m, a.data(), lda, ipiv.data(), work.data(), -1, &info);
    if (info == 0)
    {
        work.resize(work[0]);
    }
    blaze::getri(m, a.data(), lda, ipiv.data(), work.data(), work.size(), &info);
    return info;
}

template <
        template <typename, bool> class Matrix,
        typename T,
        bool SO>
int
getrs(Matrix<T, SO>& a, std::vector<int>& ipiv, Matrix<T, SO>& b)
{
    const int n(a.rows());
    const int nrhs(b.columns());
    const int lda(a.spacing());
    const int ldb(b.spacing());
    int info = 0;
    blaze::getrs('N', n, nrhs, a.data(), lda, ipiv.data(), b.data(), ldb, &info);
    return info;
}

} // ns linalg

#endif
