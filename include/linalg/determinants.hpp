#ifndef LINALG_DETERMINANTS_HPP_
#define LINALG_DETERMINANTS_HPP_

#include <vector>

namespace linalg{

/// \brief Compute the determinant, assuming that getrf has been called on m.
template <typename Matrix>
double getrfdet(const Matrix& a, const std::vector<int>& ipiv)
{
    double det = 1.0;
    for (size_t i = 0; i < linalg::num_rows(a); ++i)
    {
        if (ipiv[i] == i){
            det *= a(i, i);
        }else{
            det *= -a(i, i);
        }
    }
    return det;
}

/// \brief Compute the determinant, assuming that potrf has been called on m.
template <typename Matrix>
double potrflogdet(const Matrix& a)
{
    double logd = 0;
    for (size_t i = 0; i < linalg::num_rows(a); ++i)
    {
        logd += log(a(i, i));
    }
    // return the square since |A| = |L|^2
    return 2.0 * logd;
}

/// \brief Compute the determinant, assuming that potrf has been called on m.
template <typename Matrix>
double potrfdet(const Matrix& a)
{
    return exp(potrflogdet(a));
}

}

#endif
