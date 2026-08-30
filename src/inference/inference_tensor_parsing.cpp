/**
 * @file inference_tensor_parsing.cpp
 * @brief Implementation of inference tensor specification parsing.
 *
 * Generic token conversion remains in `ptafdeploy::parsing`; this file applies
 * model-input identity, ordering, and concrete-shape policy.
 */

#include <inference/inference_tensor_parsing.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace
{
    /** @brief Build a tensor-policy error only after validation has failed. */
    [[nodiscard]] std::string MakeError(const std::string_view context,
                                        const std::string_view detail)
    {
        std::string message;
        message.reserve(context.size() + detail.size() + 2U);
        message.append(context);
        message.append(": ");
        message.append(detail);
        return message;
    }
} // namespace

namespace ptafdeploy::inference
{
    std::vector<int64_t> ParseTensorShape(const std::string_view value,
                                          const std::string_view context)
    {
        std::vector<int64_t> shape = ptafdeploy::parsing::ParseIntegerList(value, ',', context);

        // Generic integer parsing permits signed values, whereas runtime tensor
        // descriptors require every dimension to be concrete and positive.
        if (std::any_of(shape.begin(), shape.end(),
                        [](const int64_t dimension) { return dimension <= 0; }))
        {
            throw std::invalid_argument(MakeError(context, "dimensions must be positive"));
        }
        return shape;
    }

    std::vector<ptafdeploy::parsing::SNamedValue> ResolveNamedTensorValues(
        const std::span<const std::string> specifications,
        const std::span<const STensorInfo> model_inputs, const std::string_view context)
    {
        // Validate metadata while building an allocation-conscious lookup. The
        // string views remain valid because model_inputs is borrowed for this call.
        std::unordered_map<std::string_view, size_t> input_indices;
        input_indices.reserve(model_inputs.size());
        for (size_t index = 0U; index < model_inputs.size(); ++index)
        {
            const std::string& input_name = model_inputs[index].name;
            if (input_name.empty())
            {
                throw std::invalid_argument(
                    MakeError(context, "model input names must not be empty"));
            }
            if (!input_indices.emplace(input_name, index).second)
            {
                throw std::invalid_argument(
                    MakeError(context,
                              "model input name '" + input_name + "' appears more than once"));
            }
        }

        // Resolve every specification to a model position. Optional storage
        // makes duplicate detection independent of an empty textual payload.
        std::vector<std::optional<std::string>> values(model_inputs.size());
        for (const std::string& specification : specifications)
        {
            ptafdeploy::parsing::SNamedValue parsed =
                ptafdeploy::parsing::ParseNamedValue(specification, context);

            size_t input_index = 0U;
            if (parsed.name.empty())
            {
                if (model_inputs.size() != 1U)
                {
                    throw std::invalid_argument(
                        MakeError(context, "requires name=value for multi-input models"));
                }
            }
            else
            {
                const auto matched_input = input_indices.find(parsed.name);
                if (matched_input == input_indices.end())
                {
                    throw std::invalid_argument(
                        MakeError(context, "names unknown model input '" + parsed.name + "'"));
                }
                input_index = matched_input->second;
            }

            if (values[input_index].has_value())
            {
                throw std::invalid_argument(
                    MakeError(context, "specifies model input '" +
                                           model_inputs[input_index].name + "' more than once"));
            }
            values[input_index] = std::move(parsed.value);
        }

        // Materialize only specified values, iterating model metadata to make
        // the result order deterministic regardless of argument order.
        std::vector<ptafdeploy::parsing::SNamedValue> resolved;
        resolved.reserve(specifications.size());
        for (size_t index = 0U; index < model_inputs.size(); ++index)
        {
            if (values[index].has_value())
            {
                resolved.push_back(
                    {model_inputs[index].name, std::move(values[index]).value()});
            }
        }
        return resolved;
    }
} // namespace ptafdeploy::inference
