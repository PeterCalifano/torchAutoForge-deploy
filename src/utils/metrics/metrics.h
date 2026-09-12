/** @file metrics.h
 * @brief Finite-sample statistics and source-matched 2D point errors, independent of models.
 */

#pragma once

#include <cstddef>
#include <inference/task_value_types.h>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ptafdeploy::utils::metrics
{
    /** @brief Descriptive statistics in the input samples' units; p95 uses linear interpolation. */
    struct SStatistics
    {
        size_t count{};
        double mean{}, median{}, p95{}, minimum{}, maximum{};
    };

    /** @brief Summarize borrowed samples without modifying their order.
     * @param samples Nonempty finite values.
     * @return Statistics in the same units.
     * @throws std::invalid_argument For empty/non-finite samples.
     * @throws std::overflow_error If a result cannot be represented as a finite double.
     */
    [[nodiscard]] SStatistics ComputeSampleStatistics(std::span<const double> samples);

    /** @brief Matched signed prediction-minus-reference errors and missing source identities. */
    struct SPointErrors
    {
        size_t matched{};
        std::optional<SStatistics> euclidean;
        double bias_x{}, bias_y{}, rmse{};
        std::vector<std::string> unmatched_predictions, unmatched_references;
    };

    /** @brief Match points by exact source identity and compute prediction-minus-reference errors.
     * @param predictions Source keys paired with predicted coordinates.
     * @param references Source keys paired with reference coordinates in matching units.
     * @return Errors with absent Euclidean statistics when no identities match.
     * @throws std::invalid_argument For empty/duplicate identities or non-finite coordinates.
     * @throws std::overflow_error For unrepresentable differences or metrics.
     */
    [[nodiscard]] SPointErrors ComputeMatchedPointErrors(
        std::span<const std::pair<std::string, ptafdeploy::inference::SPoint2D>> predictions,
        std::span<const std::pair<std::string, ptafdeploy::inference::SPoint2D>> references);
} // namespace ptafdeploy::utils::metrics
