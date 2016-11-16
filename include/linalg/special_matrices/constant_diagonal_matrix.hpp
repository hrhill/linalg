#ifndef LINALG_SPECIAL_MATRICES_CONSTANT_DIAGONAL_MATRIX_HPP_
#define LINALG_SPECIAL_MATRICES_CONSTANT_DIAGONAL_MATRIX_HPP_

#include <cstddef>

namespace linalg
{

/// \brief Constant diagonal matrix
struct constant_diagonal_matrix
{
    typedef double value_type;
    typedef std::size_t size_type;

    constant_diagonal_matrix(const size_type& n, const value_type& v)
        : size_(n), d_(v)
    {
    }

    constant_diagonal_matrix(const size_type& n,
                             const size_type& m,
                             const value_type& v)
        : size_(n), d_(v)
    {
        if (m != n)
            throw std::logic_error("Invalid indices passes to "
                                   "constant_diagonal_matrix_t constructor.");
    }

    template <typename T>
    operator T() const
    {
        T m(size_, size_, 0.0);
        for (size_type i = 0; i < size_; ++i)
            m(i, i) = d_;
        return m;
    }

    value_type
    operator()(const size_type& i, const size_type& j) const
    {
        return (i == j) ? d_ : value_type(0.0);
    }

    inline size_type
    size1() const
    {
        return size_;
    }

    inline size_type
    size2() const
    {
        return size_;
    }

private:
    const size_type size_;
    const value_type d_;
};

inline size_t
num_rows(const constant_diagonal_matrix& m)
{
    return m.size1();
}

inline size_t
num_cols(const constant_diagonal_matrix& m)
{
    return m.size2();
}

} // ns linalg

#endif
