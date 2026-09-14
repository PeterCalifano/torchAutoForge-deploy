/**
 * @file centroiding_adapter.h
 * @brief Example-side image and output policy for image-only centroiding models.
 *
 * These helpers deliberately remain outside the installed inference library.
 * They compose generic tensor adapters with the selected integration's
 * grayscale, resize, normalization, and coordinate-mapping contract.
 */

#pragma once

#include <filesystem>
#include <inference/model_facade.h>
#include <utils/inference_output/inference_output.h>

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

namespace ptafdeploy::examples::centroiding_models
{
    namespace infer = ptafdeploy::inference;

    /** @brief Double-precision coordinates avoid overflow when mapping finite float outputs. */
    struct SCoordinate
    {
        /** @brief Horizontal coordinate in the enclosing field's units. */
        double x{};
        /** @brief Vertical coordinate in the enclosing field's units. */
        double y{};
    };

    /** @brief One normalized centroid mapped into model and source-image pixels. */
    struct SCentroidResult
    {
        /** @brief Model output in normalized `[x / width, y / height]` coordinates. */
        SCoordinate normalized{};

        /** @brief Centroid mapped into the resized model-input image. */
        SCoordinate model_input_pixels{};

        /** @brief Centroid mapped into the original decoded image. */
        SCoordinate original_image_pixels{};
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
    [[nodiscard]] infer::SFloatTensor PrepareImageTensor(const cv::Mat& grayscale_image,
                                                         const infer::STensorInfo& input_info);

    /**
     * @brief Validate and map one normalized `[x,y]` prediction.
     * @param output Float32 model output with concrete shape `[1,2]`.
     * @param model_input_size Resized image extent used for inference.
     * @param original_image_size Decoded source-image extent.
     * @return Normalized, model-input-pixel, and original-image-pixel coordinates.
     * @throws std::invalid_argument If shape, values, or image extents are invalid.
     */
    [[nodiscard]] SCentroidResult DecodeCentroid(const infer::SFloatTensor& output,
                                                 const cv::Size model_input_size,
                                                 const cv::Size original_image_size);

    /**
     * @brief Return whether a coordinate belongs to the half-open image bounds.
     * @param result Decoded centroid.
     * @param size Original positive image extent.
     * @return True for 0 <= x < width and 0 <= y < height.
     */
    [[nodiscard]] bool Inside(const SCentroidResult& result, cv::Size size);
    /**
     * @brief Build one frame using the centroiding schema.
     * @param index Zero-based sequence index.
     * @param source Relative source filename.
     * @param size Original image extent.
     * @param output Validated finite [1,2] output tensor.
     * @param result Coordinates decoded from output.
     * @param duration Non-negative inference duration in milliseconds.
     * @param overlay Relative overlay path, or empty for JSON null.
     * @return One typed frame record with centroiding-specific fields.
     */
    [[nodiscard]] ptafdeploy::utils::inference_output::SFrameRecord FrameRecord(
        size_t index, const std::filesystem::path& source, cv::Size size,
        const infer::SFloatTensor& output, const SCentroidResult& result, double duration,
        const std::string& overlay);
    /**
     * @brief Build run metadata; model remains null until loading succeeds.
     * @param input Supplied source location.
     * @param count Number of selected frames.
     * @param model Borrowed loaded facade, or null before loading.
     * @param requested_model Supplied manifest or artifact path.
     * @return Typed run metadata without status or frame records.
     * @throws std::exception If metadata or source inspection fails.
     */
    [[nodiscard]] ptafdeploy::utils::inference_output::SRunMetadata RunMetadata(
        const std::filesystem::path& input, size_t count,
        const infer::CModelFacade* model = nullptr,
        const std::filesystem::path& requested_model = {});
    /**
     * @brief Draw clipped crosshair strokes, preserving original pixels elsewhere.
     * @param source Current image path; source bytes remain unchanged.
     * @param destination Reserved PNG output path.
     * @param result Decoded original-image coordinates.
     * @param expected_size Inference image extent, or zero to omit the extent check.
     * @throws std::runtime_error For unsupported samples, dimensions, or IO failures.
     * @throws cv::Exception If image processing fails.
     */
    void SaveOverlay(const std::filesystem::path& source, const std::filesystem::path& destination,
                     const SCentroidResult& result, cv::Size expected_size = {});
} // namespace ptafdeploy::examples::centroiding_models
