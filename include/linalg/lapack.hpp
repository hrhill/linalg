#ifndef LINALG_LAPACK_HPP_
#define LINALG_LAPACK_HPP_

#include "lapack/blaze.hpp"
#include "operations/blaze.hpp"

namespace linalg{

template <typename Matrix>
void mksym(Matrix& a)
{
    for (size_t i = 0; i < linalg::num_rows(a); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            a(j, i) = a(i, j);
        }
    }
}

}

#endif
