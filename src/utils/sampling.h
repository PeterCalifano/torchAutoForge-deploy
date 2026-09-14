/** @file sampling.h
 * @brief Sampling policies for ordered index ranges, independent of data type.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace ptafdeploy::utils
{
    /** @brief Select evenly spaced indices without replacement, including endpoints when possible.
     * @param element_count Number of ordered elements; must be positive.
     * @param sample_count Positive maximum number of indices to select, limited to element_count.
     * @return Increasing indices floor(i*(element_count-1)/(sample_count-1)); a singleton selects
     * zero.
     * @throws std::invalid_argument For zero element_count or zero sample_count.
     */
    [[nodiscard]] std::vector<size_t> SelectEvenlySpacedIndices(size_t element_count,
                                                            size_t sample_count);
} // namespace ptafdeploy::utils
