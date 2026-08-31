/**
 * @file plain_centroiding_support.h
 * @brief Example-side image and output policy for plain centroiding models.
 *
 * These helpers deliberately remain outside the installed inference library.
 * They compose generic tensor adapters with the selected integration's
 * grayscale, resize, normalization, and coordinate-mapping contract.
 */

#pragma once

#include <inference/task_adapters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ptafdeploy::examples::plain_centroiding
{
    namespace infer = ptafdeploy::inference;

    /** @brief One normalized centroid mapped into model and source-image pixels. */
    struct SCentroidResult
    {
        /** @brief Model output in normalized `[x / width, y / height]` coordinates. */
        infer::SPoint2D normalized{};

        /** @brief Centroid mapped into the resized model-input image. */
        infer::SPoint2D model_input_pixels{};

        /** @brief Centroid mapped into the original decoded image. */
        infer::SPoint2D original_image_pixels{};
    };

    /**
     * @brief Resize and scale one grayscale image into a model input tensor.
     *
     * The model contract must describe float32 `[N,1,H,W]`, with a dynamic or
     * unit batch and concrete spatial dimensions. OpenCV bilinear resize is
     * followed by `1 / 255` scaling through the generic HWC-to-NCHW adapter.
     *
     * @param grayscale_image Non-empty, unsigned 8-bit, single-channel image.
     * @param input_info Authoritative model input metadata.
     * @return Owned float32 tensor with concrete shape `[1,1,H,W]`.
     * @throws std::invalid_argument If the image or model contract is unsupported.
     * @throws std::overflow_error If image cardinality exceeds host limits.
     * @throws cv::Exception If OpenCV cannot resize or allocate the image.
     */
    [[nodiscard]] inline infer::SFloatTensor PrepareImageTensor(
        const cv::Mat& grayscale_image, const infer::STensorInfo& input_info)
    {
        if (grayscale_image.empty() || grayscale_image.type() != CV_8UC1)
        {
            throw std::invalid_argument(
                "Plain centroiding requires a non-empty unsigned 8-bit grayscale image.");
        }

        if (input_info.name.empty() || input_info.dtype != "float32" ||
            input_info.shape.size() != 4U)
        {
            throw std::invalid_argument(
                "Plain centroiding expects one named float32 input with shape [N,1,H,W].");
        }

        const bool supported_batch = input_info.shape[0] == -1 || input_info.shape[0] == 1;
        const bool supported_spatial_shape =
            input_info.shape[1] == 1 && input_info.shape[2] > 0 && input_info.shape[3] > 0 &&
            input_info.shape[2] <= std::numeric_limits<int>::max() &&
            input_info.shape[3] <= std::numeric_limits<int>::max();
        if (!supported_batch || !supported_spatial_shape)
        {
            throw std::invalid_argument(
                "Plain centroiding expects one named float32 input with shape [N,1,H,W].");
        }

        const int height = static_cast<int>(input_info.shape[2]);
        const int width = static_cast<int>(input_info.shape[3]);

        // Avoid a resize allocation when the decoded image already has the
        // model extent, while retaining contiguous storage for flat access.
        cv::Mat resized_image;
        if (grayscale_image.size() == cv::Size{width, height})
        {
            resized_image = grayscale_image;
        }
        else
        {
            cv::resize(grayscale_image, resized_image, cv::Size{width, height}, 0.0, 0.0,
                       cv::INTER_LINEAR);
        }
        if (!resized_image.isContinuous())
        {
            resized_image = resized_image.clone();
        }

        const auto* pixels = resized_image.ptr<uint8_t>(0);
        return infer::MakeNchwFloatTensorFromHwcAccessor(
            input_info.name,
            static_cast<size_t>(height),
            static_cast<size_t>(width),
            1U,
            1.0F / 255.0F,
            false,
            [pixels](const size_t index) { return pixels[index]; });
    }

    /**
     * @brief Validate and map one normalized `[x,y]` prediction.
     * @param output Float32 model output with concrete shape `[1,2]`.
     * @param model_input_size Resized image extent used for inference.
     * @param original_image_size Decoded source-image extent.
     * @return Normalized, model-input-pixel, and original-image-pixel coordinates.
     * @throws std::invalid_argument If shape, values, or image extents are invalid.
     */
    [[nodiscard]] inline SCentroidResult
    DecodeCentroid(const infer::SFloatTensor& output, const cv::Size model_input_size,
                   const cv::Size original_image_size)
    {
        if (output.shape != std::vector<int64_t>{1, 2})
        {
            throw std::invalid_argument(
                "Plain centroiding expects one output with concrete shape [1,2].");
        }
        if (model_input_size.width <= 0 || model_input_size.height <= 0 ||
            original_image_size.width <= 0 || original_image_size.height <= 0)
        {
            throw std::invalid_argument("Centroid coordinate extents must be positive.");
        }

        infer::SFeatureRowSchema schema;
        schema.attribute_axis = 1;
        const std::vector<infer::SFeature2D> features = infer::DecodeFeatureRows(output, schema);
        if (features.size() != 1U)
        {
            throw std::invalid_argument("Plain centroiding must produce exactly one feature.");
        }

        const infer::SPoint2D normalized = features.front().position;
        if (normalized.x < 0.0F || normalized.x > 1.0F || normalized.y < 0.0F ||
            normalized.y > 1.0F)
        {
            throw std::invalid_argument(
                "Plain centroiding normalized coordinates must remain within [0,1].");
        }

        return {
            normalized,
            {normalized.x * static_cast<float>(model_input_size.width),
             normalized.y * static_cast<float>(model_input_size.height)},
            {normalized.x * static_cast<float>(original_image_size.width),
             normalized.y * static_cast<float>(original_image_size.height)},
        };
    }
} // namespace ptafdeploy::examples::plain_centroiding
