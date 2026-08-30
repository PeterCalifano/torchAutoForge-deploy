/**
 * @file inference_tensor_parsing.h
 * @brief Reusable parsing and resolution for inference tensor specifications.
 *
 * This installed layer composes generic textual parsing with model metadata.
 * It owns tensor-specific rules such as positive concrete dimensions,
 * single-input positional shorthand, and deterministic model-input ordering.
 */

#pragma once

#include <inference/inference_common.h>
#include <utils/value_parsing.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Parse a concrete tensor shape from comma-separated dimensions.
     *
     * Surrounding ASCII whitespace is accepted for each dimension. Zero and
     * negative values, including the `-1` convention used for dynamic model
     * metadata, are invalid because runtime input tensors require a concrete
     * shape.
     *
     * @param value Comma-separated positive dimensions.
     * @param context User-facing description included in validation errors.
     * @return Concrete dimensions in source order.
     * @throws std::invalid_argument If any dimension is missing, malformed,
     * overflowing, or not strictly positive.
     */
    [[nodiscard]] std::vector<int64_t> ParseTensorShape(std::string_view value,
                                                        std::string_view context);

    /**
     * @brief Resolve `[name=]value` specifications against model input metadata.
     *
     * Unqualified values are accepted only for a single-input model. Results
     * follow model-input order regardless of specification order and contain
     * only inputs that were specified. The input spans are borrowed only for
     * the call; every returned name and value owns its storage.
     *
     * @param specifications Optional-name values to resolve.
     * @param model_inputs Authoritative model input metadata. Names must be
     * non-empty and unique.
     * @param context User-facing description included in validation errors.
     * @return Resolved owning values in model-input order.
     * @throws std::invalid_argument If model names are invalid, a specification
     * is malformed, unqualified input is ambiguous, a name is unknown, or one
     * input is specified more than once.
     */
    [[nodiscard]] std::vector<ptafdeploy::parsing::SNamedValue> ResolveNamedTensorValues(
        std::span<const std::string> specifications, std::span<const STensorInfo> model_inputs,
        std::string_view context);
} // namespace ptafdeploy::inference
