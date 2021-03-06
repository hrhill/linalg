#define BOOST_TEST_MODULE norms
#define BOOST_TEST_DYN_LINK

#include <ctime>
#include <iostream>
#include <limits>
#include <random>

#include <blaze/Math.h>

#include <boost/test/unit_test.hpp>

#include "linalg/operations.hpp"
#include "linalg/special_matrices.hpp"

#include "test_utilities.hpp"

using namespace std;

template <typename Vector, typename Matrix>
int
zero_tests()
{
    const int n = 5;
    Vector zero(n, 0.0);

    BOOST_CHECK_EQUAL(linalg::norm_1(zero), linalg::norm_2(zero));
    BOOST_CHECK_EQUAL(linalg::norm_2(zero), linalg::norm_p(zero, 2));
    BOOST_CHECK_EQUAL(linalg::norm_1(zero), linalg::norm_p(zero, 1));
    BOOST_CHECK_EQUAL(linalg::norm_p(zero, 5), linalg::norm_infinity(zero));

    return 0;
}

template <typename Vector, typename Matrix>
int
one_tests()
{
    const int n = 5;
    Vector one(n, 1.0);

    BOOST_CHECK_EQUAL(linalg::norm_1(one), n);
    BOOST_CHECK_EQUAL(linalg::norm_2(one), sqrt(n));
    BOOST_CHECK_EQUAL(linalg::norm_infinity(one), 1);

    return 0;
}

template <typename Vector, typename Matrix>
int
expr_tests()
{
    const double tol = 1e-04;
    const int n = 5;
    Vector one(n, 1.0);

    BOOST_CHECK_EQUAL(linalg::norm_1(one - one), 0);
    BOOST_CHECK_EQUAL(linalg::norm_2(one - one), 0);
    BOOST_CHECK_EQUAL(linalg::norm_infinity(one - one), 0);

    BOOST_CHECK_CLOSE(linalg::norm_1(one / n), 1.0, tol);
    BOOST_CHECK_CLOSE(linalg::norm_2(one / sqrt(n)), 1.0, tol);
    BOOST_CHECK_CLOSE(linalg::norm_infinity(1.0 * one), 1.0, tol);

    return 0;
}

BOOST_AUTO_TEST_CASE(blaze_norm_tests)
{
    std::cout << "Testing blaze\n";
    typedef blaze::DynamicVector<double> vector_t;
    typedef blaze::DynamicMatrix<double, blaze::columnMajor> matrix_t;

    BOOST_CHECK_EQUAL((zero_tests<vector_t, matrix_t>()), 0);
    BOOST_CHECK_EQUAL((one_tests<vector_t, matrix_t>()), 0);
}
