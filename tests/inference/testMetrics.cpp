/** @file testMetrics.cpp
 * @brief Known-value and failure checks for model-independent metrics.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utils/metrics/metrics.h>

namespace metrics = ptafdeploy::utils::metrics;

TEST_CASE("Metrics compute defined percentiles without changing input order", "[utils][metrics]")
{
    const std::vector<double> values{4, 1, 3, 2};

    // Check the defined interpolation against a small sample with a known ordering.
    const auto sample_statistics = metrics::ComputeSampleStatistics(values);
    REQUIRE(sample_statistics.count == 4);
    REQUIRE(sample_statistics.mean == 2.5);
    REQUIRE(sample_statistics.median == 2.5);
    REQUIRE(sample_statistics.p95 == Catch::Approx(3.85));
    REQUIRE(sample_statistics.minimum == 1);
    REQUIRE(sample_statistics.maximum == 4);

    // The borrowed input remains unchanged, and singleton samples are valid.
    REQUIRE(values.front() == 4);
    REQUIRE(metrics::ComputeSampleStatistics(std::vector<double>{7}).p95 == 7);

    // Reject invalid samples while retaining large finite values.
    REQUIRE_THROWS_AS(metrics::ComputeSampleStatistics({}), std::invalid_argument);
    REQUIRE_THROWS_AS(metrics::ComputeSampleStatistics(
                          std::vector<double>{std::numeric_limits<double>::infinity()}),
                      std::invalid_argument);
    REQUIRE(
        std::isfinite(metrics::ComputeSampleStatistics(std::vector<double>{1e308, 1e308}).mean));
}

TEST_CASE("Point metrics match source identities and report missing records", "[utils][metrics]")
{
    const std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>> predicted_points{
        {"b", {-1, 0}}, {"a", {3, 4}}, {"missing", {5, 6}}};
    const std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>> reference_points{
        {"a", {0, 0}}, {"b", {0, 0}}, {"unused", {7, 8}}};

    // Match by source rather than vector position and report both kinds of missing records.
    const auto result = metrics::ComputeMatchedPointErrors(predicted_points, reference_points);
    REQUIRE(result.matched == 2);
    REQUIRE(result.bias_x == 1);
    REQUIRE(result.bias_y == 2);
    REQUIRE(result.rmse == Catch::Approx(std::sqrt(13.0)));
    REQUIRE(result.euclidean->mean == 3);
    REQUIRE(result.euclidean->p95 == Catch::Approx(4.8));
    REQUIRE(result.unmatched_predictions == std::vector<std::string>{"missing"});
    REQUIRE(result.unmatched_references == std::vector<std::string>{"unused"});

    // Empty matches are valid, but duplicate identities would make residuals ambiguous.
    REQUIRE_FALSE(metrics::ComputeMatchedPointErrors({}, reference_points).euclidean.has_value());
    REQUIRE_THROWS_AS(metrics::ComputeMatchedPointErrors(
                          std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>>{
                              {"a", {0, 0}}, {"a", {1, 1}}},
                          reference_points),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(metrics::ComputeMatchedPointErrors(
                          {},
                          std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>>{
                              {"a", {0, 0}}, {"a", {1, 1}}}),
                      std::invalid_argument);
}
