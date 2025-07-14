#include <execution> // Optional, for parallel policies
#include <iostream>
#include <numeric> // for std::reduce
#include <vector>

namespace deploy_aux
{

#if (PARALLELEXEC)
    /**
     * @brief Computes the product of elements in a vector in parallel.
     *
     * @tparam T The type of elements in the vector.
     * @param v The input vector.
     * @return T The product of the elements.
     */
    template <typename T>
    T AccumProduct(const std::vector<T> &v)
    {

        return std::reduce(
            std::execution::par,
            v.begin(), v.end(),
            static_cast<T>(1),
            std::multiplies<T>());
    }
#else

    /**
     * @brief Computes the product of elements in a vector serially.
     *
     * @tparam T The type of elements in the vector.
     * @param v The input vector.
     * @return T The product of the elements.
     */
    template <typename T>
    T AccumProduct(const std::vector<T> &v)
    {
        // serial:
        return std::reduce(
            v.begin(), v.end(),
            static_cast<T>(1),
            std::multiplies<T>());
    }
#endif

};
