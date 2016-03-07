#ifndef LINALG_BLAS_BLAZE_HPP_
#define LINALG_BLAS_BLAZE_HPP_

#include <blaze/Math.h>

namespace linalg{

using blaze::gemv;

template <
    template <typename, bool> class MatrixType,
    typename T1,
    bool SO,
    template <typename, bool> class VectorType,
    typename T2,
    bool TF>
inline
void
gemv(const T1& a,
     const MatrixType<T1, SO>& m,
     const VectorType<T2, TF>& v,
     const T1& b,
     VectorType<T2, TF>& res)
{
    return blaze::gemv(res, m, v, a, b);
}

/// gemm
template <
    template <typename, bool> class MatrixType1,
    typename T,
    bool SO1,
    template <typename, bool> class MatrixType2,
    bool SO2,
    template <typename, bool> class MatrixType3,
    bool SO3>
inline
void
gemm(const T& a,
        const MatrixType1<T, SO1>& A,
        const MatrixType2<T, SO2>& B,
        const T& b,
        MatrixType3<T, SO3>& C)
{
    blaze::gemm(C, A, B, a, b);
}

} //ns linalg

#endif
