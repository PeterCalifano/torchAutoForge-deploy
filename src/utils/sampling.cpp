/** @file sampling.cpp
 * @brief Deterministic sampling of ordered index ranges.
 */

#include "sampling.h"
#include <algorithm>
#include <stdexcept>

namespace ptafdeploy::utils
{
    std::vector<size_t> SelectEvenlySpacedIndices(size_t element_count, size_t sample_count)
    {
        if (!element_count || !sample_count)
            throw std::invalid_argument("Selection requires inputs and positive count");

        sample_count = std::min(element_count, sample_count);
        std::vector<size_t> selected_indices;
        selected_indices.reserve(sample_count);
        if (sample_count == 1)
            return {0};

        // Quotient/remainder stepping avoids multiplying potentially large sequence indices.
        const auto interval_count = sample_count - 1,
                   base_step = (element_count - 1) / interval_count;
        const auto step_remainder = (element_count - 1) % interval_count;
        size_t selected_index = 0, accumulated_remainder = 0;

        for (size_t sample_index = 0; sample_index < sample_count; ++sample_index)
        {
            selected_indices.push_back(selected_index);
            if (sample_index + 1 == sample_count)
                break;
            selected_index += base_step;
            if (accumulated_remainder >= interval_count - step_remainder)
            {
                ++selected_index;
                accumulated_remainder -= interval_count - step_remainder;
            }
            else
                accumulated_remainder += step_remainder;
        }

        return selected_indices;
    }
} // namespace ptafdeploy::utils
