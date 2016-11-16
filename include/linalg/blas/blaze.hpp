#ifndef LINALG_BLAS_BLAZE_HPP_
#define LINALG_BLAS_BLAZE_HPP_

#include <blaze/Math.h>

namespace linalg
{

template <template <typename, bool> class MatrixType,
          typename T1,
          bool SO,
          template <typename, bool> class VectorType,
          typename T2,
          bool TF>
inline void
gemv(const T1& a,
     const MatrixType<T1, SO>& m,
     const VectorType<T2, TF>& v,
     const T1& b,
     VectorType<T2, TF>& res)
{
    if (b == 0.0)
    {
        res = a * m * v;
    }
    else
    {
        if (b != 1.0)
        {
            res *= b;
        }
        res += a * m * v;
    }
}

/// gemm
template <template <typename, bool> class MatrixType1,
          typename T,
          bool SO1,
          template <typename, bool> class MatrixType2,
          bool SO2,
          template <typename, bool> class MatrixType3,
          bool SO3>
inline void
gemm(const T& a,
     const MatrixType1<T, SO1>& A,
     const MatrixType2<T, SO2>& B,
     const T& b,
     MatrixType3<T, SO3>& C)
{
    if (b == 0.0)
    {
        C = a * A * B;
    }else{
        if (b != 1.0)
        {
            C *= b;
        }
        C += a * A * B;
    }   
}

} // ns linalg

#endif
