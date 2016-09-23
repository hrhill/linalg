#define BOOST_TEST_MODULE operations
#include <iostream>
#include <random>
#include <limits>
#include <ctime>

#include <blaze/Math.h>

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "linalg.hpp"
#include "linalg/operations.hpp"

using namespace std;
using namespace linalg;

template <typename Vector, typename Matrix>
int size_test()
{
    const int n = 5;
    Vector v(n, 0.0);

    BOOST_CHECK_EQUAL(size(v), n);
    return 0;
}

BOOST_AUTO_TEST_CASE(blaze_norm_tests)
{
    std::cout << "Testing blaze\n";
    typedef blaze::DynamicVector<double> vector_t;
    typedef blaze::DynamicMatrix<double, blaze::columnMajor> matrix_t;

    BOOST_CHECK_EQUAL((size_test<vector_t, matrix_t>()), 0);
}

BOOST_AUTO_TEST_CASE(blaze_outer_prod)
{
    // Default vectors
    blaze::DynamicVector<double> x(3);
    blaze::DynamicVector<double, blaze::rowVector> y(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    y[0] = 10;
    y[1] = 20;
    y[2] = 30;

    blaze::DynamicMatrix<double> m = linalg::outer_prod(x, y);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            BOOST_CHECK_EQUAL(m(i, j), (i + 1) * (10 * (j + 1)));
        }
    }

    blaze::DynamicMatrix<double> p = linalg::outer_prod(x, x);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            BOOST_CHECK_EQUAL(p(i, j), (i + 1) * (j + 1));
        }
    }
}
