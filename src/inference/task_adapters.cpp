/**
 * @file task_adapters.cpp
 * @brief Generic image conversion and schema-driven tensor row decoding.
 */

#include "task_adapters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ptafdeploy::inference
{
    namespace
    {
        struct SRowLayout
        {
            size_t attribute_count{0U};
            size_t inner_stride{0U};
            size_t row_count{0U};
        };

        [[nodiscard]] size_t NormalizeAttributeAxis(const int64_t requested_axis, const size_t rank)
        {
            if (rank == 0U || rank > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            {
                throw std::invalid_argument(
                    "Task-adapter tensors must have at least one representable dimension.");
            }

            const int64_t signed_rank = static_cast<int64_t>(rank);
            const int64_t normalized_axis =
                requested_axis < 0 ? requested_axis + signed_rank : requested_axis;
            if (normalized_axis < 0 || normalized_axis >= signed_rank)
            {
                throw std::invalid_argument("Tensor attribute axis is out of range.");
            }
            return static_cast<size_t>(normalized_axis);
        }

        [[nodiscard]] SRowLayout ResolveRowLayout(const SFloatTensor& output,
                                                  const int64_t requested_axis)
        {
            const size_t attribute_axis =
                NormalizeAttributeAxis(requested_axis, output.shape.size());
            const size_t expected_values = ComputeElementCount(output.shape);
            if (output.values.size() != expected_values)
            {
                throw std::invalid_argument(
                    "Task-adapter tensor value count does not match its shape.");
            }

            const size_t attribute_count = static_cast<size_t>(output.shape[attribute_axis]);
            if (attribute_count == 0U)
            {
                throw std::invalid_argument("Tensor attribute axis must not be empty.");
            }

            size_t inner_stride = 1U;
            for (size_t dimension = attribute_axis + 1U; dimension < output.shape.size();
                 ++dimension)
            {
                inner_stride =
                    CheckedMultiply(inner_stride, static_cast<size_t>(output.shape[dimension]),
                                    "Task-adapter row stride overflows size_t.");
            }

            return {attribute_count, inner_stride, expected_values / attribute_count};
        }

        [[nodiscard]] size_t GetRowValueIndex(const size_t row, const size_t attribute,
                                              const SRowLayout& layout)
        {
            const size_t outer_index = row / layout.inner_stride;
            const size_t inner_index = row % layout.inner_stride;
            return ((outer_index * layout.attribute_count + attribute) * layout.inner_stride) +
                   inner_index;
        }

        void ValidateSchemaIndex(const size_t index, const size_t attribute_count)
        {
            if (index >= attribute_count)
            {
                throw std::invalid_argument(
                    "Task-adapter schema index is outside the attribute axis.");
            }
        }

        void ValidateFiniteFeature(const SFeature2D& feature)
        {
            if (!std::isfinite(feature.position.x) || !std::isfinite(feature.position.y) ||
                !std::isfinite(feature.score))
            {
                throw std::invalid_argument("Decoded feature values must be finite.");
            }
        }

        [[nodiscard]] bool IsInClassRange(const size_t index, const SDetectionRowSchema& schema)
        {
            return index >= schema.first_class_score_index &&
                   index < schema.first_class_score_index + schema.class_score_count;
        }

        void ValidateDetectionSchema(const SDetectionRowSchema& schema,
                                     const size_t attribute_count)
        {
            if (schema.class_score_count == 0U)
            {
                throw std::invalid_argument(
                    "Detection schema must contain at least one class score.");
            }
            if (schema.first_class_score_index > attribute_count ||
                schema.class_score_count > attribute_count - schema.first_class_score_index)
            {
                throw std::invalid_argument(
                    "Detection class score range is outside the attribute axis.");
            }

            const std::array<size_t, 4U> box_indices{
                schema.box_coordinate_0_index, schema.box_coordinate_1_index,
                schema.box_coordinate_2_index, schema.box_coordinate_3_index};
            for (size_t index = 0U; index < box_indices.size(); ++index)
            {
                ValidateSchemaIndex(box_indices[index], attribute_count);
                if (IsInClassRange(box_indices[index], schema))
                {
                    throw std::invalid_argument(
                        "Detection schema fields must not overlap the class score range.");
                }
                for (size_t previous = 0U; previous < index; ++previous)
                {
                    if (box_indices[index] == box_indices[previous])
                    {
                        throw std::invalid_argument(
                            "Detection box schema fields must not overlap.");
                    }
                }
            }

            if (schema.objectness_index < -1)
            {
                throw std::invalid_argument(
                    "Detection objectness index must be -1 or non-negative.");
            }
            if (schema.objectness_index >= 0)
            {
                const size_t objectness_index = static_cast<size_t>(schema.objectness_index);
                ValidateSchemaIndex(objectness_index, attribute_count);
                if (IsInClassRange(objectness_index, schema) ||
                    std::find(box_indices.begin(), box_indices.end(), objectness_index) !=
                        box_indices.end())
                {
                    throw std::invalid_argument(
                        "Detection objectness field must not overlap another field.");
                }
            }
        }

        [[nodiscard]] SBoundingBox2D DecodeBox(const SFloatTensor& output, const size_t row,
                                               const SRowLayout& layout,
                                               const SDetectionRowSchema& schema)
        {
            const float value_0 =
                output.values[GetRowValueIndex(row, schema.box_coordinate_0_index, layout)];
            const float value_1 =
                output.values[GetRowValueIndex(row, schema.box_coordinate_1_index, layout)];
            const float value_2 =
                output.values[GetRowValueIndex(row, schema.box_coordinate_2_index, layout)];
            const float value_3 =
                output.values[GetRowValueIndex(row, schema.box_coordinate_3_index, layout)];

            if (schema.box_encoding == EBoundingBoxEncoding::center_xywh)
            {
                return MakeBoundingBox2DFromCenterSize(value_0, value_1, value_2, value_3);
            }
            if (schema.box_encoding == EBoundingBoxEncoding::corners_xyxy)
            {
                return MakeBoundingBox2DFromXyxy(value_0, value_1, value_2, value_3);
            }
            throw std::invalid_argument("Unsupported bounding-box encoding.");
        }
    } // namespace

    SFloatTensor MakeNchwFloatTensorFromHwcFloat(const std::string& tensor_name,
                                                 const std::vector<float>& hwc_values,
                                                 const size_t height, const size_t width,
                                                 const size_t channels, const float scale,
                                                 const bool swap_rb)
    {
        const size_t pixel_count =
            CheckedMultiply(height, width, "Image pixel count overflows size_t.");
        const size_t value_count =
            CheckedMultiply(pixel_count, channels, "Image value count overflows size_t.");
        if (hwc_values.size() != value_count)
        {
            throw std::invalid_argument(
                "HWC image value count does not match height*width*channels.");
        }

        return MakeNchwFloatTensorFromHwcAccessor(tensor_name, height, width, channels, scale,
                                                  swap_rb, [&hwc_values](const size_t index)
                                                  { return hwc_values[index]; });
    }

    std::vector<SFeature2D> DecodeFeatureRows(const SFloatTensor& output,
                                              const SFeatureRowSchema& schema)
    {
        const SRowLayout layout = ResolveRowLayout(output, schema.attribute_axis);
        ValidateSchemaIndex(schema.x_index, layout.attribute_count);
        ValidateSchemaIndex(schema.y_index, layout.attribute_count);
        if (schema.score_index < -1)
        {
            throw std::invalid_argument("Feature score index must be -1 or non-negative.");
        }
        if (schema.score_index >= 0)
        {
            ValidateSchemaIndex(static_cast<size_t>(schema.score_index), layout.attribute_count);
        }

        std::vector<SFeature2D> features;
        features.reserve(layout.row_count);
        for (size_t row = 0U; row < layout.row_count; ++row)
        {
            SFeature2D feature;
            feature.position.x = output.values[GetRowValueIndex(row, schema.x_index, layout)];
            feature.position.y = output.values[GetRowValueIndex(row, schema.y_index, layout)];
            if (schema.score_index >= 0)
            {
                feature.score = output.values[GetRowValueIndex(
                    row, static_cast<size_t>(schema.score_index), layout)];
            }
            ValidateFiniteFeature(feature);
            features.push_back(feature);
        }
        return features;
    }

    std::vector<SDetection2D> DecodeDetectionRows(const SFloatTensor& output,
                                                  const SDetectionRowSchema& schema,
                                                  const float score_threshold,
                                                  const size_t max_detections)
    {
        if (!std::isfinite(score_threshold) || score_threshold < 0.0F)
        {
            throw std::invalid_argument(
                "Detection score threshold must be finite and non-negative.");
        }

        const SRowLayout layout = ResolveRowLayout(output, schema.attribute_axis);
        ValidateDetectionSchema(schema, layout.attribute_count);

        std::vector<SDetection2D> detections;
        for (size_t row = 0U; row < layout.row_count; ++row)
        {
            float best_class_score = -std::numeric_limits<float>::infinity();
            int64_t best_class_id = -1;
            for (size_t class_offset = 0U; class_offset < schema.class_score_count; ++class_offset)
            {
                const float class_score = output.values[GetRowValueIndex(
                    row, schema.first_class_score_index + class_offset, layout)];
                if (!std::isfinite(class_score))
                {
                    throw std::invalid_argument("Decoded detection class scores must be finite.");
                }
                if (class_score > best_class_score)
                {
                    best_class_score = class_score;
                    best_class_id = static_cast<int64_t>(class_offset);
                }
            }

            float objectness = 1.0F;
            if (schema.objectness_index >= 0)
            {
                objectness = output.values[GetRowValueIndex(
                    row, static_cast<size_t>(schema.objectness_index), layout)];
                if (!std::isfinite(objectness))
                {
                    throw std::invalid_argument(
                        "Decoded detection objectness values must be finite.");
                }
            }

            const float final_score = objectness * best_class_score;
            if (!std::isfinite(final_score))
            {
                throw std::invalid_argument("Decoded detection scores must be finite.");
            }

            // Validate every decoded row even when its score excludes it from results.
            const SBoundingBox2D bounds = DecodeBox(output, row, layout, schema);
            if (final_score >= score_threshold)
            {
                detections.push_back({bounds, {best_class_id, final_score}});
            }
        }

        std::stable_sort(detections.begin(), detections.end(),
                         [](const SDetection2D& lhs, const SDetection2D& rhs)
                         { return lhs.classification.score > rhs.classification.score; });
        if (max_detections > 0U && detections.size() > max_detections)
        {
            detections.resize(max_detections);
        }
        return detections;
    }
} // namespace ptafdeploy::inference
