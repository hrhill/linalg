#include <iostream>
#include <random>
#include <ctime>
#include <exception>
#include <stdexcept>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operations.hpp>

#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/bindings/blas.hpp>

#include "ftl/tools/time_it.h"
#include "linalg.hpp"

namespace ublas = boost::numeric::ublas;
namespace blas = boost::numeric::bindings::blas;

typedef ublas::vector<double> ublas_vector_t;
typedef ublas::matrix<double, boost::numeric::ublas::column_major> ublas_matrix_t;
typedef ublas::identity_matrix<double> ublas_identity_matrix_t;

int main(int argc, char** argv)
{
    if(argc != 2)
        std::runtime_error("Inavlid number of arguments to gemv benchmark.");

    int n = atoi(argv[1]);
    ublas_matrix_t T(n, n);
    ublas_vector_t v(n);
    ublas_vector_t a(n);
    ublas_identity_matrix_t I(n);

    const int n_trials = 20;
    std::cout << "Full gemv     : ";
    time_it([&T, &v, &a](){
            blas::gemv(1.0, T, v, 0.0, a);
        }, n_trials);

    std::cout << "ublas Identity gemv : ";
    time_it([&I, &v, &a](){
            a = ublas::prod(I, v);
        }, n_trials);

    std::cout << "blas Identity gemv : ";
    time_it([&I, &v, &a](){
            linalg::gemv(1.0, I, v, 0.0, a);
        }, n_trials);
}
