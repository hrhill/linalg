#ifndef LINALG_SPECIAL_MATRICES_DIAGONAL_MATRIX_HPP_
#define LINALG_SPECIAL_MATRICES_DIAGONAL_MATRIX_HPP_

#include <cstddef>
#include <vector>

namespace linalg
{

/// \brief Diagonal matrix.
struct diagonal_matrix
{
    typedef double value_type;
    typedef size_t size_type;

    template <typename T>
    explicit diagonal_matrix(const T& d)
    :
        d_(d.size())
    {
        std::copy(d.begin(), d.end(), d_.begin());
    }

    diagonal_matrix(const size_type &n)
    :
        diagonal_matrix(n, n, 0.0)
    {}

    diagonal_matrix(const size_type &n, const size_type& m)
    :
        diagonal_matrix(n, m, 0.0)
    {}

    diagonal_matrix(const size_type &n,
                    const size_type &m,
                    const value_type &v)
    :
        d_(n)
    {
        if (m != n)
            throw std::logic_error(
                "Invalid indices passes to diagonal matrix constructor.");
        std::fill(d_.begin(), d_.end(), v);
    }

    template <typename T>
    operator T() const
    {
        T m(size1(), size2(), 0.0);
        for (size_type i = 0; i < d_.size(); ++i)
            m(i, i) = d_[i];
        return m;
    }

    value_type operator()(const size_type &i, const size_type &j) const
    {
        assert(i < d_.size() && j < d_.size());
        return (i == j) ? d_[i] : value_type(0.0);
    }

    value_type &operator()(const size_type &i, const size_type &j)
    {
        if (i != j)
            throw std::logic_error("Can't assign to non-diagonal elements");
        return d_[i];
    }

    inline size_type size1() const
    {
        return d_.size();
    }

    inline size_type size2() const
    {
        return d_.size();
    }

private:
    std::vector<double> d_;
};

inline size_t num_rows(const diagonal_matrix& m)
{
    return m.size1();
}

inline size_t num_cols(const diagonal_matrix& m)
{
    return m.size2();
}

}

#endif
