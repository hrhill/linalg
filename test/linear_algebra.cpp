#define BOOST_TEST_MODULE linalg
#define BOOST_TEST_DYN_LINK

#include <ctime>
#include <iostream>
#include <limits>
#include <random>

#include <boost/test/unit_test.hpp>

#include "linalg.hpp"
#include "linalg/operations.hpp"
#include "linalg/special_matrices.hpp"

#include "test_utilities.hpp"

using namespace std;
using namespace linalg;

const double threshold = sqrt(std::numeric_limits<double>::epsilon());

template <typename Vector, typename Matrix>
void
cholesky_tests()
{
    const int n = 5;

    mt19937 rng(std::time(0));
    auto M = generate_spd_matrix<Matrix>(rng, n);

    /// Test solver
    auto x = generate_vector<Vector>(rng, n);
    Vector y(n);
    gemv(1.0, M, x, 0.0, y);

    Vector xsol = cholesky_solve(M, y);

    BOOST_CHECK(norm_infinity(static_cast<const Vector&>(x - xsol)) <= 1e-04);

    /// Test determinant and inversions
    Matrix invM = cholesky_invert(M);
    BOOST_CHECK_CLOSE(
        cholesky_determinant(M), 1.0 / cholesky_determinant(invM), threshold);
    BOOST_CHECK_CLOSE(
        log_cholesky_determinant(M), log(cholesky_determinant(M)), threshold);

    Matrix id1(n, n, 0);
    gemm(1.0, M, invM, 0.0, id1);
    Matrix id2(n, n, 0);
    gemm(1.0, invM, M, 0.0, id2);

    for (int i = 0; i < n; ++i)
    {

        BOOST_CHECK_CLOSE(id1(i, i), 1.0, threshold);
        BOOST_CHECK_CLOSE(id2(i, i), 1.0, threshold);

        for (int j = 0; j < i; ++j)
        {
            BOOST_CHECK(fabs(id1(i, j)) <= threshold);
            BOOST_CHECK(fabs(id1(j, i)) <= threshold);

            BOOST_CHECK(fabs(id2(i, j)) <= threshold);
            BOOST_CHECK(fabs(id2(j, i)) <= threshold);
        }
    }
    // Check the determinants are 1
    BOOST_CHECK_CLOSE(cholesky_determinant(id1), 1.0, threshold);
    BOOST_CHECK_CLOSE(cholesky_determinant(id2), 1.0, threshold);
}

template <typename Vector, typename Matrix>
void
lu_tests()
{
    const int n = 3;

    mt19937 rng(std::time(0));
    auto M = generate_spd_matrix<Matrix>(rng, n);
    Matrix Mp = M;
    std::vector<int> ipiv(n);
    linalg::getrf(Mp, ipiv);
    std::cout << Mp << std::endl;
    for (int i = 0; i < n; ++i)
        std::cout << ipiv[i] << ",";
    std::cout << "\n";

    auto invM = lu_invert(M);

    Matrix idl(n, n, 0.0);
    Matrix idr(n, n, 0.0);

    linalg::gemm(1.0, M, invM, 0.0, idl);
    linalg::gemm(1.0, invM, M, 0.0, idr);

    BOOST_CHECK_CLOSE(lu_determinant(M), 1.0 / lu_determinant(invM), threshold);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            BOOST_CHECK(fabs(idl(i, j) - idr(i, j)) <= threshold);
            BOOST_CHECK(fabs(idl(i, j) - static_cast<double>(i == j)) <=
                        threshold);
        }
    }

    // test solver
    auto x = generate_vector<Vector>(rng, n);
    Vector y(n);
    gemv(1.0, M, x, 0.0, y);

    Vector xsol = lu_solve(M, y);

    BOOST_CHECK(norm_infinity(static_cast<const Vector&>(x - xsol)) <= 1e-04);

    // Check the determinants are 1
    BOOST_CHECK_CLOSE(lu_determinant(idl), 1.0, threshold);
    BOOST_CHECK_CLOSE(lu_determinant(idr), 1.0, threshold);
}

BOOST_AUTO_TEST_CASE(blaze_lapack_tests)
{
    BOOST_TEST_MESSAGE("Testing blaze");
    typedef blaze::DynamicVector<double> vector_t;
    typedef blaze::DynamicMatrix<double, blaze::columnMajor> matrix_t;

    cholesky_tests<vector_t, matrix_t>();
    lu_tests<vector_t, matrix_t>();
}
