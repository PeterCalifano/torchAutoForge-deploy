/** @file metrics.cpp
 * @brief Stable finite statistics and deterministic point correspondence.
 */

#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace ptafdeploy::utils::metrics
{
    namespace
    {
        double ConvertToFiniteDouble(long double value)
        {
            const double result = static_cast<double>(value);
            if (!std::isfinite(result))
                throw std::overflow_error("Metric exceeds finite double range");
            return result;
        }

        void ValidatePointSourceAndCoordinates(const std::string& source,
                                               const ptafdeploy::inference::SPoint2D& point)
        {
            if (source.empty() || !std::isfinite(point.x) || !std::isfinite(point.y))
                throw std::invalid_argument("Invalid point for source: " + source);
        }
    } // namespace

    SStatistics ComputeSampleStatistics(std::span<const double> samples)
    {
        if (samples.empty())
            throw std::invalid_argument("Statistics require at least one sample");

        // Scale each contribution before summing to limit overflow in the mean.
        std::vector<double> sorted_samples(samples.begin(), samples.end());
        long double scaled_sample_sum = 0;
        for (double value : sorted_samples)
        {
            if (!std::isfinite(value))
                throw std::invalid_argument("Statistics require finite samples");
            scaled_sample_sum += static_cast<long double>(value) / samples.size();
        }

        // Interpolate quantiles on a copy so the caller retains the original ordering.
        std::sort(sorted_samples.begin(), sorted_samples.end());
        const auto interpolate_quantile = [&](long double quantile_fraction) {
            const long double fractional_index = quantile_fraction * (sorted_samples.size() - 1);
            const auto lower_index = static_cast<size_t>(fractional_index);
            const auto upper_index = std::min(lower_index + 1, sorted_samples.size() - 1);
            const long double interpolation_weight = fractional_index - lower_index;
            return ConvertToFiniteDouble((1 - interpolation_weight) * sorted_samples[lower_index] +
                                         interpolation_weight * sorted_samples[upper_index]);
        };

        return {samples.size(),
                ConvertToFiniteDouble(scaled_sample_sum),
                interpolate_quantile(0.5L),
                interpolate_quantile(0.95L),
                sorted_samples.front(),
                sorted_samples.back()};
    }

    SPointErrors ComputeMatchedPointErrors(
        std::span<const std::pair<std::string, ptafdeploy::inference::SPoint2D>> predictions,
        std::span<const std::pair<std::string, ptafdeploy::inference::SPoint2D>> references)
    {
        // Build reference lookups and reject ambiguous source identities.
        std::map<std::string, const ptafdeploy::inference::SPoint2D*> reference_points_by_source;
        for (const auto& [source, point] : references)
        {
            ValidatePointSourceAndCoordinates(source, point);
            if (!reference_points_by_source.emplace(source, &point).second)
                throw std::invalid_argument("Duplicate reference source: " + source);
        }

        // Match predictions by source, recording missing references separately.
        std::set<std::string> prediction_sources;
        SPointErrors result;
        std::vector<double> euclidean_errors;
        long double residual_sum_x = 0, residual_sum_y = 0;
        double scaled_error_norm = 0;

        for (const auto& [source, point] : predictions)
        {
            ValidatePointSourceAndCoordinates(source, point);
            if (!prediction_sources.insert(source).second)
                throw std::invalid_argument("Duplicate prediction source: " + source);

            const auto reference_position = reference_points_by_source.find(source);
            if (reference_position == reference_points_by_source.end())
            {
                result.unmatched_predictions.push_back(source);
                continue;
            }

            // Form signed residuals only for matching pairs.
            const double residual_x = ConvertToFiniteDouble(static_cast<long double>(point.x) -
                                                            reference_position->second->x);
            const double residual_y = ConvertToFiniteDouble(static_cast<long double>(point.y) -
                                                            reference_position->second->y);
            const double euclidean_error = ConvertToFiniteDouble(std::hypot(
                static_cast<long double>(residual_x), static_cast<long double>(residual_y)));
            euclidean_errors.push_back(euclidean_error);
            residual_sum_x += residual_x;
            residual_sum_y += residual_y;
        }

        // Report absent predictions in deterministic source order.
        for (const auto& [source, point] : reference_points_by_source)
            if (!prediction_sources.contains(source))
                result.unmatched_references.push_back(source);
        std::sort(result.unmatched_predictions.begin(), result.unmatched_predictions.end());

        // Leave statistics absent when the inputs have no source identities in common.
        result.matched = euclidean_errors.size();
        if (!euclidean_errors.empty())
        {
            result.euclidean = ComputeSampleStatistics(euclidean_errors);
            result.bias_x = ConvertToFiniteDouble(residual_sum_x / euclidean_errors.size());
            result.bias_y = ConvertToFiniteDouble(residual_sum_y / euclidean_errors.size());

            // Accumulate scaled norms to avoid squaring large finite distances.
            const double normalization_factor =
                std::sqrt(static_cast<double>(euclidean_errors.size()));
            for (double euclidean_error : euclidean_errors)
                scaled_error_norm =
                    std::hypot(scaled_error_norm, euclidean_error / normalization_factor);
            result.rmse = ConvertToFiniteDouble(scaled_error_norm);
        }

        return result;
    }
} // namespace ptafdeploy::utils::metrics
