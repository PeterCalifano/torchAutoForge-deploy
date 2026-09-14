/**
 * @file task_adapters.h
 * @brief Generic preprocessing and schema-driven tensor row decoding.
 */

#pragma once

#include <inference/inference_common.h>
#include <inference/task_value_types.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ptafdeploy::inference
{
    /** @brief Attribute-axis and column mapping for generic 2D feature rows. */
    struct SFeatureRowSchema
    {
        /** @brief Tensor axis containing row attributes; negative values count from the end. */
        int64_t attribute_axis{-1};
        /** @brief Zero-based x-coordinate index on the attribute axis. */
        size_t x_index{0U};
        /** @brief Zero-based y-coordinate index on the attribute axis. */
        size_t y_index{1U};
        /** @brief Optional score index, with `-1` selecting the default score of one. */
        int64_t score_index{-1};
    };

    /**
     * @brief Attribute-axis and column mapping for dense detection rows.
     *
     * Box-coordinate fields map to `[cx, cy, width, height]` for
     * `center_xywh` and `[min_x, min_y, max_x, max_y]` for `corners_xyxy`.
     */
    struct SDetectionRowSchema
    {
        /** @brief Tensor axis containing row attributes; negative values count from the end. */
        int64_t attribute_axis{-1};
        /** @brief Interpretation of the four configured box coordinates. */
        EBoundingBoxEncoding box_encoding{EBoundingBoxEncoding::center_xywh};
        /** @brief First box-coordinate index on the attribute axis. */
        size_t box_coordinate_0_index{0U};
        /** @brief Second box-coordinate index on the attribute axis. */
        size_t box_coordinate_1_index{1U};
        /** @brief Third box-coordinate index on the attribute axis. */
        size_t box_coordinate_2_index{2U};
        /** @brief Fourth box-coordinate index on the attribute axis. */
        size_t box_coordinate_3_index{3U};
        /** @brief Optional objectness index, with `-1` selecting an implicit value of one. */
        int64_t objectness_index{-1};
        /** @brief First index in the contiguous class-score range. */
        size_t first_class_score_index{4U};
        /** @brief Number of contiguous class scores; must be positive for decoding. */
        size_t class_score_count{0U};
    };

    /**
     * @brief Convert an owned HWC float image buffer to an NCHW tensor.
     * @param tensor_name Name assigned to the returned tensor.
     * @param hwc_values Flat HWC source values.
     * @param height Image height.
     * @param width Image width.
     * @param channels Channel count.
     * @param scale Multiplicative value scale.
     * @param swap_rb Swap channels zero and two when true.
     * @return Owned float tensor with shape `[1,C,H,W]`.
     * @throws std::invalid_argument If dimensions, channel policy, or cardinality is invalid.
     * @throws std::overflow_error If the image cardinality overflows `size_t`.
     */
    [[nodiscard]] SFloatTensor MakeNchwFloatTensorFromHwcFloat(const std::string& tensor_name,
                                                               const std::vector<float>& hwc_values,
                                                               size_t height, size_t width,
                                                               size_t channels, float scale,
                                                               bool swap_rb);

    /**
     * @brief Convert HWC samples from a non-owning accessor into NCHW storage.
     * @tparam TValueAt Callable accepting a flat HWC index and returning a float-like value.
     * @param tensor_name Name assigned to the returned tensor.
     * @param height Image height.
     * @param width Image width.
     * @param channels Channel count.
     * @param scale Multiplicative value scale.
     * @param swap_rb Swap channels zero and two when true.
     * @param value_at Source accessor invoked once per output value.
     * @return Owned float tensor with shape `[1,C,H,W]`.
     * @throws std::invalid_argument If dimensions or channel policy is invalid.
     * @throws std::overflow_error If the image cardinality overflows `size_t`.
     */
    template <typename TValueAt>
        requires std::invocable<TValueAt, size_t> &&
                 std::convertible_to<std::invoke_result_t<TValueAt, size_t>, float>
    [[nodiscard]] SFloatTensor
    MakeNchwFloatTensorFromHwcAccessor(const std::string& tensor_name, const size_t height,
                                       const size_t width, const size_t channels, const float scale,
                                       const bool swap_rb, TValueAt&& value_at)
    {
        if (height == 0U || width == 0U || channels == 0U)
        {
            throw std::invalid_argument("Image dimensions must be positive.");
        }
        if (swap_rb && channels < 3U)
        {
            throw std::invalid_argument("RGB/BGR channel swap requires at least three channels.");
        }

        const size_t pixel_count =
            CheckedMultiply(height, width, "Image pixel count overflows size_t.");
        const size_t value_count =
            CheckedMultiply(pixel_count, channels, "Image value count overflows size_t.");

        std::vector<float> nchw_values(value_count);
        for (size_t channel = 0U; channel < channels; ++channel)
        {
            size_t source_channel = channel;
            if (swap_rb && channel == 0U)
            {
                source_channel = 2U;
            }
            else if (swap_rb && channel == 2U)
            {
                source_channel = 0U;
            }

            for (size_t y = 0U; y < height; ++y)
            {
                for (size_t x = 0U; x < width; ++x)
                {
                    const size_t hwc_index = ((y * width + x) * channels) + source_channel;
                    const size_t nchw_index = ((channel * height + y) * width) + x;
                    nchw_values[nchw_index] =
                        static_cast<float>(std::invoke(value_at, hwc_index)) * scale;
                }
            }
        }

        return SFloatTensor{tensor_name,
                            {1, static_cast<int64_t>(channels), static_cast<int64_t>(height),
                             static_cast<int64_t>(width)},
                            std::move(nchw_values)};
    }

    /**
     * @brief Decode unit-agnostic feature points from a dense float tensor.
     * @param output Concrete tensor whose configured axis contains row attributes.
     * @param schema Attribute-axis and coordinate/score column mapping.
     * @return Feature rows in tensor traversal order; absent scores default to one.
     * @throws std::exception If shape, cardinality, schema, or values are invalid.
     */
    [[nodiscard]] std::vector<SFeature2D> DecodeFeatureRows(const SFloatTensor& output,
                                                            const SFeatureRowSchema& schema);

    /**
     * @brief Decode scored geometric detections from a dense float tensor.
     * @param output Concrete tensor whose configured axis contains row attributes.
     * @param schema Attribute, box, objectness, and class-score mapping.
     * @param score_threshold Inclusive minimum final score; must be finite and non-negative.
     * @param max_detections Maximum results after stable score sorting; zero keeps all.
     * @return Detections sorted by descending score.
     * @throws std::exception If shape, cardinality, schema, values, or boxes are invalid.
     */
    [[nodiscard]] std::vector<SDetection2D> DecodeDetectionRows(const SFloatTensor& output,
                                                                const SDetectionRowSchema& schema,
                                                                float score_threshold,
                                                                size_t max_detections);
} // namespace ptafdeploy::inference
