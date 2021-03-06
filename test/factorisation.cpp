/// \file factorisation_test.cpp
#define BOOST_TEST_DYN_LINK

#include <iostream>
#include <limits>
#include <random>
#include <string>

#include <boost/timer.hpp>

#include <boost/mpl/list.hpp>

#include <blaze/Math.h>

#define BOOST_TEST_MODULE factorisation_test

#include <boost/test/unit_test.hpp>

#include "linalg/blas.hpp"
#include "linalg/factorisations/cholesky.hpp"
#include "linalg/factorisations/gmw81.hpp"
#include "linalg/factorisations/ldlt.hpp"
#include "linalg/factorisations/tools.hpp"
#include "linalg/operations.hpp"
#include "linalg/special_matrices.hpp"

#include "test_utilities.hpp"

template <typename Matrix>
Matrix
convert_to_cholesky(Matrix LD)
{
    int n = linalg::num_rows(LD);
    Matrix D(n, n, 0);

    for (int i = 0; i < n; ++i)
    {
        D(i, i) = sqrt(LD(i, i));
        LD(i, i) = 1.0;
    }
    Matrix L(n, n, 0);
    linalg::gemm(1.0, LD, D, 0.0, L);
    return L;
}

template <typename Matrix>
int
max_magnitude_diagonal_check()
{
    const int n = 5;
    const int max_id = 3;
    const double max_value = 10;

    Matrix m = linalg::eye(n);
    m(max_id, max_id) = max_value;

    auto x = linalg::factorisations::tools::max_magnitude_diagonal(m);

    BOOST_CHECK_EQUAL(std::get<0>(x), max_id);
    BOOST_CHECK_EQUAL(std::get<1>(x), max_value);
    return 0;
}

template <typename Matrix>
int
identity_matrix_check()
{
    const int ndim = 5;

    Matrix A = linalg::identity_matrix(ndim);
    Matrix L = linalg::factorisations::gmw81(A);
    const int n = linalg::num_rows(L);
    for (int i = 0; i < n; ++i)
    {
        BOOST_CHECK_EQUAL(L(i, i), 1.0);
        for (int j = 0; j < i; ++j)
        {
            BOOST_CHECK_EQUAL(L(i, j), 0.0);
        }
    }
    return 0;
}

template <typename Matrix>
int
scaled_identity_matrix_check()
{
    const int ndim = 5;
    std::mt19937 rng(std::time(0));
    auto rnorm = std::bind(std::normal_distribution<>(0, 1), std::ref(rng));
    const double d = exp(rnorm());

    Matrix A = d * linalg::eye(ndim);
    Matrix L = linalg::factorisations::gmw81(A);
    const int n = linalg::num_rows(L);
    for (int i = 0; i < n; ++i)
    {
        BOOST_CHECK_EQUAL(L(i, i), d);
        for (int j = 0; j < i; ++j)
        {
            BOOST_CHECK_EQUAL(L(i, j), 0.0);
        }
    }
    return 0;
}

template <typename Matrix>
int
factorisation_equivalence_test()
{
    // Make all three factorisations agree on positive definite matrices.
    std::mt19937 rng(std::time(0));
    const int ndim = 5;
    Matrix A = linalg::generate_spd_matrix<Matrix>(rng, ndim);

    Matrix cholL = linalg::factorisations::cholesky(A);
    Matrix ldlLD = linalg::factorisations::ldlt(A);
    Matrix gmw81LD = linalg::factorisations::gmw81(A);

    std::cout << cholL << std::endl;

    Matrix ldlL = convert_to_cholesky(ldlLD);
    Matrix gmw81L = convert_to_cholesky(gmw81LD);

    std::cout << ldlL << std::endl;
    std::cout << gmw81L << std::endl;

    for (int i = 0; i < ndim; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            BOOST_CHECK_CLOSE(ldlLD(i, j), gmw81LD(i, j), 1e-01);
            BOOST_CHECK_CLOSE(ldlL(i, j), gmw81L(i, j), 1e-01);
            BOOST_CHECK_CLOSE(gmw81L(i, j), cholL(i, j), 1e-01);
        }
    }
    return 0;
}

template <typename Matrix>
int
all_tests()
{
    max_magnitude_diagonal_check<Matrix>();
    identity_matrix_check<Matrix>();
    scaled_identity_matrix_check<Matrix>();
    factorisation_equivalence_test<Matrix>();
    return 0;
}

BOOST_AUTO_TEST_CASE(blaze_factorisation_tests)
{
    std::cout << "Testing blaze\n";
    typedef blaze::DynamicMatrix<double, blaze::columnMajor> matrix_t;

    BOOST_CHECK_EQUAL((all_tests<matrix_t>()), 0);
}
