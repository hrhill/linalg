#ifndef LINALG_SPECIAL_MATRICES_IDENTITY_MATRIX_HPP_
#define LINALG_SPECIAL_MATRICES_IDENTITY_MATRIX_HPP_

#include <cstddef>

#include "constant_diagonal_matrix.hpp"

namespace linalg
{
/// \brief Identity matrix
struct identity_matrix
{
    typedef double value_type;
    typedef size_t size_type;

    identity_matrix() = default;

    identity_matrix(const size_type &n)
    :
        size_(n)
    {}

    template <typename T>
    operator T() const
    {
        T m(size1(), size2(), 0.0);
        for (size_type i = 0; i < size_; ++i)
            m(i, i) = 1.0;
        return m;
    }

    value_type operator()(const size_type &i, const size_type &j) const
    {
        assert(i < size_ && j < size_);
        return static_cast<value_type>(i == j);
    }

    inline size_type size1() const
    {
        return size_;
    }

    inline size_type size2() const
    {
        return size_;
    }

private:
    size_type size_;
};

inline size_t num_rows(const identity_matrix& m)
{
    return m.size1();
}

inline size_t num_cols(const identity_matrix& m)
{
    return m.size2();
}

constant_diagonal_matrix operator*(const double a, const identity_matrix& i)
{
    return constant_diagonal_matrix(i.size1(), a);
}

}

#endif
