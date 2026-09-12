/** @file centroiding_adapter.cpp
 * @brief Image-only model policy and centroiding-specific output interpretation.
 */
#include "centroiding_adapter.h"
#include <cmath>
#include <inference/task_adapters.h>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utils/images/images.h>
namespace ptafdeploy::examples::centroiding_models
{
    namespace images = ptafdeploy::utils::images;
    namespace out = ptafdeploy::utils::inference_output;
    using J = out::SJsonValue;
    namespace fs = std::filesystem;
    infer::SFloatTensor PrepareImageTensor(const cv::Mat& grayscale_image,
                                           const infer::STensorInfo& input_info)
    {
        if (grayscale_image.empty() || grayscale_image.type() != CV_8UC1)
        {
            throw std::invalid_argument(
                "Centroiding requires a non-empty unsigned 8-bit grayscale image.");
        }

        if (input_info.name.empty() || input_info.dtype != "float32" ||
            input_info.shape.size() != 4U)
        {
            throw std::invalid_argument(
                "Centroiding expects one named float32 input with shape [N,1,H,W].");
        }

        const bool supported_batch = input_info.shape[0] == -1 || input_info.shape[0] == 1;
        const bool supported_spatial_shape =
            input_info.shape[1] == 1 && input_info.shape[2] > 0 && input_info.shape[3] > 0 &&
            input_info.shape[2] <= std::numeric_limits<int>::max() &&
            input_info.shape[3] <= std::numeric_limits<int>::max();
        if (!supported_batch || !supported_spatial_shape)
        {
            throw std::invalid_argument(
                "Centroiding expects one named float32 input with shape [N,1,H,W].");
        }

        const int height = static_cast<int>(input_info.shape[2]);
        const int width = static_cast<int>(input_info.shape[3]);

        // Avoid a resize allocation when the decoded image already has the
        // model extent, while retaining contiguous storage for flat access.
        cv::Mat resized_image =
            images::Resize(grayscale_image, cv::Size{width, height}, cv::INTER_LINEAR);
        if (!resized_image.isContinuous())
        {
            resized_image = resized_image.clone();
        }

        const auto* pixels = resized_image.ptr<uint8_t>(0);
        return infer::MakeNchwFloatTensorFromHwcAccessor(
            input_info.name, static_cast<size_t>(height), static_cast<size_t>(width), 1U,
            1.0F / 255.0F, false, [pixels](const size_t index) { return pixels[index]; });
    }

    SCentroidResult DecodeCentroid(const infer::SFloatTensor& output,
                                   const cv::Size model_input_size,
                                   const cv::Size original_image_size)
    {
        if (output.shape != std::vector<int64_t>{1, 2})
        {
            throw std::invalid_argument(
                "Centroiding expects one output with concrete shape [1,2].");
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
            throw std::invalid_argument("Centroiding must produce exactly one feature.");
        }

        const infer::SPoint2D normalized = features.front().position;
        if (!std::isfinite(normalized.x) || !std::isfinite(normalized.y))
            throw std::invalid_argument("Centroid coordinates must be finite.");

        return {
            {normalized.x, normalized.y},
            {normalized.x * static_cast<double>(model_input_size.width),
             normalized.y * static_cast<double>(model_input_size.height)},
            {normalized.x * static_cast<double>(original_image_size.width),
             normalized.y * static_cast<double>(original_image_size.height)},
        };
    }
    bool Inside(const SCentroidResult& p, cv::Size size)
    {
        return p.original_image_pixels.x >= 0 && p.original_image_pixels.y >= 0 &&
               p.original_image_pixels.x < size.width && p.original_image_pixels.y < size.height;
    }
    out::SFrameRecord FrameRecord(size_t index, const fs::path& source, cv::Size size,
                                  const infer::SFloatTensor& output, const SCentroidResult& result,
                                  double duration, const std::string& overlay)
    {
        const auto point = [](const SCoordinate& p) -> J {
            return J::Object{{"x", p.x}, {"y", p.y}};
        };
        auto raw = out::TensorValue(output);
        // Preserve the existing centroiding schema; the generic tensor API also includes dtype.
        std::get<J::Object>(raw.value).erase("dtype");
        return {index,
                source.generic_string(),
                duration,
                {{"image_size", J::Object{{"width", size.width}, {"height", size.height}}},
                 {"raw_output", std::move(raw)},
                 {"centroid", J::Object{{"normalized", point(result.normalized)},
                                        {"model_pixels", point(result.model_input_pixels)},
                                        {"image_pixels", point(result.original_image_pixels)},
                                        {"inside_image", Inside(result, size)}}},
                 {"overlay", overlay.empty() ? J{} : J{overlay}}}};
    }
    out::SRunMetadata RunMetadata(const fs::path& input, size_t count,
                                  const infer::CModelFacade* model, const fs::path& requested_model)
    {
        out::SRunMetadata metadata;
        metadata.fields = {
            {"requested_model_path", requested_model.string()},
            {"input", J::Object{{"path", input.string()},
                                {"kind", fs::is_directory(input) ? "directory" : "image"},
                                {"ordering", "natural_filename"},
                                {"selected_frame_count", count}}},
            {"model", J{}},
            {"preprocessing", J::Object{{"library", "OpenCV " CV_VERSION},
                                        {"grayscale", "IMREAD_GRAYSCALE uint8"},
                                        {"resize", "bilinear"},
                                        {"scale", 1.0 / 255.0}}}};
        if (model)
        {
            const auto c = model->GetContract();
            const auto tensors = [](const std::vector<infer::STensorInfo>& infos) -> J {
                J::Array result;
                for (const auto& info : infos)
                {
                    J::Array shape;
                    for (auto extent : info.shape)
                        shape.emplace_back(extent);
                    result.emplace_back(J::Object{
                        {"name", info.name}, {"dtype", info.dtype}, {"shape", std::move(shape)}});
                }
                return result;
            };
            J::Array priority;
            for (auto target : c.runtime.execution_target_priority)
                priority.emplace_back(static_cast<int>(target));
            metadata.fields["model"] = J::Object{
                {"artifact_path", c.artifact_path},
                {"config_path", c.config_path},
                {"role", c.role},
                {"backend_detail", c.backend_detail},
                {"preprocessing", c.preprocessing},
                {"postprocessing", c.postprocessing},
                {"inputs", tensors(c.inputs)},
                {"outputs", tensors(c.outputs)},
                {"runtime", J::Object{{"device_id", c.runtime.device_id},
                                      {"intra_op_num_threads", c.runtime.intra_op_num_threads},
                                      {"inter_op_num_threads", c.runtime.inter_op_num_threads},
                                      {"allow_fallback", c.runtime.allow_fallback},
                                      {"backend", static_cast<int>(c.runtime.backend)},
                                      {"artifact", static_cast<int>(c.runtime.artifact)},
                                      {"execution_target_priority", std::move(priority)},
                                      {"enable_profiling", c.runtime.enable_profiling},
                                      {"log_id", c.runtime.log_id},
                                      {"tensorrt_optimization_profile_index",
                                       c.runtime.tensorrt_optimization_profile_index}}}};
        }
        return metadata;
    }
    void SaveOverlay(const fs::path& source, const fs::path& destination,
                     const SCentroidResult& result, cv::Size expected_size)
    {
        auto overlay = images::DecodeForOverlay(source);
        if (expected_size.width > 0 && expected_size.height > 0 && overlay.size() != expected_size)
            throw std::runtime_error("Overlay and inference image extents differ: " +
                                     source.string());
        if (overlay.depth() != CV_8U && overlay.depth() != CV_16U)
            throw std::runtime_error("PNG overlays require uint8 or uint16 images");
        if (overlay.channels() == 1)
            overlay = images::Convert(overlay, cv::COLOR_GRAY2BGR);
        const cv::Point2d point{result.original_image_pixels.x, result.original_image_pixels.y};
        const double white = overlay.depth() == CV_16U ? 65535.0 : 255.0;
        images::DrawCrosshair(overlay, point, cv::Scalar(0, 0, 0, white), 19, 3);
        images::DrawCrosshair(overlay, point, cv::Scalar(white, white, white, white), 17, 1);
        images::SavePng(destination, overlay);
    }
} // namespace ptafdeploy::examples::centroiding_models
