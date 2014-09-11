#include <iostream>
#include <random>
#include <ctime>
#include <exception>
#include <stdexcept>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/bindings/blas.hpp>

#include "linalg.hpp"
#include "time_it.hpp"

namespace ublas = boost::numeric::ublas;
namespace blas = boost::numeric::bindings::blas;

typedef ublas::matrix<double, ublas::column_major> ublas_matrix_t;
typedef ublas::compressed_matrix<double> ublas_sparse_matrix_t;

#ifdef HAVE_MTL
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/bindings/mtl/dense2D.hpp>
typedef mtl::dense2D<double,
    mtl::matrix::parameters<mtl::tag::col_major>> mtl_matrix_t;
typedef mtl::compressed2D<double,
    mtl::matrix::parameters<mtl::tag::col_major>> mtl_sparse_matrix_t;
#endif

#ifdef HAVE_BLAZE
#include <blaze/Math.h>
typedef blaze::DynamicVector<double> blaze_vector;
typedef blaze::DynamicMatrix<double, blaze::columnMajor> blaze_dense;
typedef blaze::CompressedMatrix<double, blaze::columnMajor> blaze_sparse;
#endif

void gemm(const int n){
    /// Generate two matrices of size n
    std::mt19937 rng(std::time(0));

    ublas_matrix_t A(n, n);
    ublas_matrix_t B(n, n);
    ublas_matrix_t C(n, n);
    auto normrnd = std::bind(std::normal_distribution<>(0, 1), rng);

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            A(i, j) = normrnd();
            B(i, j) = normrnd();
        }
    }

    std::cout << "ublas::prod        : ";
    time_it([&A, &B, &C](){noalias(C) = ublas::prod(A, B);});
    std::cout << "blas::gemm (ublas) : ";
    time_it([&A, &B, &C](){linalg::gemm(1.0, A, B, 0.0, C);});

#ifdef HAVE_MTL
    mtl_matrix_t Ap(n, n);
    mtl_matrix_t Bp(n, n);
    mtl_matrix_t Cp(n, n);

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            Ap(i, j) = normrnd();
            Bp(i, j) = normrnd();
        }
    }
    std::cout << "blas::gemm (mtl  ) : ";
    time_it([&Ap, &Bp, &Cp](){linalg::gemm(1.0, Ap, Bp, 0.0, Cp);});
#endif

#ifdef HAVE_BLAZE
    blaze_dense Ab(n, n);
    blaze_dense Bb(n, n);
    blaze_dense Cb(n, n);

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            Ab(i, j) = normrnd();
            Bb(i, j) = normrnd();
        }
    }
    std::cout << "blaze::*           : ";
    time_it([&Ab, &Bb, &Cb](){
        Cb = Ab * Bb;
        //blas::gemm(1.0, Ap, Bp, 0.0, Cp);
    });
#endif
}

void sparse_gemm(const int n){

    std::mt19937 rng(std::time(0));
    auto normrnd = std::bind(std::normal_distribution<>(0, 1), rng);

    ublas_sparse_matrix_t uAs(n, n, 4 * n);
    ublas_sparse_matrix_t uBs(n, n, 4 * n);

    ublas_matrix_t uCs(n, n);
    for (int i = 0; i < n; ++i){
        uAs(i, i) = normrnd();
        uAs(i, i + 1) = normrnd();
        uBs(i, i) = normrnd();
        uBs(i, i + 1) = normrnd();

        uAs(i, n - 1) = normrnd();
        uAs(i, n - 2) = normrnd();
        uBs(i, n - 1) = normrnd();
        uBs(i, n - 2) = normrnd();
    }

    std::cout << "ublas::axpy_prod() : ";
    time_it([&uAs, &uBs, &uCs](){
        noalias(uCs) = ublas::prod(uAs, uBs);
        //axpy_prod(uAs, uBs, uCs, true);
        //ublas_matrix_t Res(uCs);
    });
#ifdef HAVE_MTL
    // MTL sparse
    mtl_sparse_matrix_t As(n, n);
    mtl_sparse_matrix_t Bs(n, n);
    mtl_matrix_t Cs(n, n);

    As = 0.;
    Bs = 0.;
    Cs = 0.;
    {
        mtl::matrix::inserter<mtl_sparse_matrix_t> insA(As);
        mtl::matrix::inserter<mtl_sparse_matrix_t> insB(Bs);
        for (int i = 0; i < n; ++i){
            insA[i][i] << normrnd();
            insB[i][i] << normrnd();
            if (i + 1 < n){
                insA[i][i + 1] << normrnd();
                insB[i][i + 1] << normrnd();
            }
            insA[i][n - 1] << normrnd();
            insA[i][n - 2] << normrnd();
            insB[i][n - 1] << normrnd();
            insB[i][n - 2] << normrnd();
        }
    } // Ensure that inserter is destroyed

    std::cout << "mtl_sparse::*      : ";
    time_it([&As, &Bs, &Cs](){Cs = As * Bs;});
#endif

#ifdef HAVE_BLAZE

    blaze_sparse Ab(n, n);
    blaze_sparse Bb(n, n);
    blaze_dense Cb(n, n);

    for (int i = 0; i < n; ++i){
        Ab(i, i) = normrnd();
        Bb(i, i) = normrnd();
        if (i + 1 < n){
            Ab(i, i + 1) = normrnd();
            Bb(i, i + 1) = normrnd();
        }
        Ab(i, n - 1) = normrnd();
        Ab(i, n - 2) = normrnd();
        Bb(i, n - 1) = normrnd();
        Bb(i, n - 2) = normrnd();
    } // Ensure that inserter is destroyed

    std::cout << "blaze_sparse::*    : ";
    time_it([&Ab, &Bb, &Cb](){Cb = Ab * Bb;});
#endif
}

int main(int argc, char** argv)
{
    if(argc != 2)
        std::runtime_error("Inavlid number of arguments to gemm benchmark.");

    /// Get dimension
    const int n = atoi(argv[1]);

    gemm(n);
    sparse_gemm(n);
    return 0;
}
