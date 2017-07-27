#define BOOST_TEST_MODULE Special types
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "linalg/special_matrices.hpp"

#include <blaze/Math.h>

using namespace linalg;

BOOST_AUTO_TEST_CASE(identity_checker)
{
    int n = 5;
    const identity_matrix a(n);

    BOOST_CHECK_EQUAL(linalg::num_rows(a), n);
    BOOST_CHECK_EQUAL(linalg::num_cols(a), n);

    blaze::DynamicMatrix<double, blaze::columnMajor> blaze_a = a;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const double aij = a(i, j);
            BOOST_CHECK_EQUAL(aij, blaze_a(i, j));
            if (i == j)
            {
                BOOST_CHECK_EQUAL(blaze_a(i, i), 1.0);
            }
            else
            {
                BOOST_CHECK_EQUAL(blaze_a(i, j), 0.0);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(constant_diagonal_checker)
{
    BOOST_CHECK_THROW(constant_diagonal_matrix(1, 2, 3), std::logic_error);

    int n = 5;
    const constant_diagonal_matrix a(n, 3.3);

    BOOST_CHECK_EQUAL(linalg::num_rows(a), n);
    BOOST_CHECK_EQUAL(linalg::num_cols(a), n);

    blaze::DynamicMatrix<double, blaze::columnMajor> blaze_a = a;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const double aij = a(i, j);
            BOOST_CHECK_EQUAL(aij, blaze_a(i, j));
            if (i == j)
            {
                BOOST_CHECK_EQUAL(blaze_a(i, i), 3.3);
            }
            else
            {
                BOOST_CHECK_EQUAL(blaze_a(i, j), 0.0);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(diagonal_checker)
{
    BOOST_CHECK_THROW(diagonal_matrix(1, 2, 3), std::logic_error);
    int n = 5;
    const diagonal_matrix a(std::vector<double>{1, 2, 3, 4, 5});

    BOOST_CHECK_EQUAL(linalg::num_rows(a), n);
    BOOST_CHECK_EQUAL(linalg::num_cols(a), n);

    blaze::DynamicMatrix<double, blaze::columnMajor> blaze_a = a;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const double aij = a(i, j);
            BOOST_CHECK_EQUAL(aij, blaze_a(i, j));
            if (i == j)
            {
                BOOST_CHECK_EQUAL(blaze_a(i, i), i + 1);
            }
            else
            {
                BOOST_CHECK_EQUAL(blaze_a(i, j), 0.0);
            }
        }
    }

    diagonal_matrix b(n, n);
    for (int i = 0; i < n; ++i)
    {
        b(i, i) = i;
        for (int j = i + 1; j < n; ++j)
        {
            BOOST_CHECK_THROW(b(i, j), std::logic_error);
            BOOST_CHECK_THROW(b(j, i), std::logic_error);
        }
    }
}
