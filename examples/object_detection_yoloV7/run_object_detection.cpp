/**
 * @file run_object_detection.cpp
 * @brief Run a YOLOv7 image through CModelFacade and generic task adapters.
 */

#include <inference/model_facade.h>
#include <inference/task_adapters.h>
#include <utils/logging/CLogger.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tclap/CmdLine.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("object_detection_yoloV7", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    struct SArguments
    {
        fs::path manifest_path{};
        fs::path image_path{};
        std::optional<infer::EExecutionTarget> execution_target{};
        int device_id{0};
        float score_threshold{0.25F};
        size_t max_detections{20U};
    };

    [[nodiscard]] std::optional<infer::EExecutionTarget> ParseTarget(const std::string& value)
    {
        if (value == "manifest")
        {
            return std::nullopt;
        }
        if (value == "cpu")
        {
            return infer::EExecutionTarget::cpu;
        }
        if (value == "cuda")
        {
            return infer::EExecutionTarget::cuda;
        }

        throw std::invalid_argument("--target expects manifest, cpu, or cuda.");
    }

    [[nodiscard]] SArguments ParseArguments(const int argc, char** argv)
    {
        TCLAP::CmdLine command(
            "Run one image through CModelFacade, shared HWC-to-NCHW preprocessing, and a "
            "schema-driven pre-NMS detection decoder.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        TCLAP::UnlabeledValueArg<std::string> manifest_path(
            "model", "Path to the object-detection model manifest", true, "",
            "model.ptafmodel", command);
        TCLAP::UnlabeledValueArg<std::string> image_path(
            "image", "Path to the input image", true, "", "image", command);
        TCLAP::ValueArg<std::string> target(
            "", "target", "Execution target override or manifest-configured runtime", false,
            "manifest", "manifest|cpu|cuda", command);
        TCLAP::ValueArg<int> device(
            "", "device", "Non-negative runtime device index", false, 0,
            "non-negative integer", command);
        TCLAP::ValueArg<float> score_threshold(
            "", "score-threshold", "Minimum finite non-negative detection score", false, 0.25F,
            "non-negative number", command);
        TCLAP::ValueArg<long long> max_detections(
            "", "max-detections", "Maximum number of score-sorted detections to report", false,
            20LL, "non-negative integer", command);

        command.parse(argc, argv);

        if (device.getValue() < 0)
        {
            throw std::invalid_argument("--device expects a non-negative integer.");
        }
        if (!std::isfinite(score_threshold.getValue()) || score_threshold.getValue() < 0.0F)
        {
            throw std::invalid_argument("--score-threshold expects a finite non-negative number.");
        }
        if (max_detections.getValue() < 0 ||
            static_cast<unsigned long long>(max_detections.getValue()) >
                std::numeric_limits<size_t>::max())
        {
            throw std::invalid_argument("--max-detections expects a non-negative integer.");
        }

        SArguments arguments;
        arguments.manifest_path = manifest_path.getValue();
        arguments.image_path = image_path.getValue();
        arguments.execution_target = ParseTarget(target.getValue());
        arguments.device_id = device.getValue();
        arguments.score_threshold = score_threshold.getValue();
        arguments.max_detections = static_cast<size_t>(max_detections.getValue());

        return arguments;
    }

    void LoadModel(const SArguments& arguments, infer::CModelFacade& model)
    {
        if (!arguments.execution_target.has_value())
        {
            model.LoadModelConfig(arguments.manifest_path.string());
            return;
        }

        infer::SRuntimeConfig runtime;
        runtime.SetDeviceId(arguments.device_id);
        runtime.ClearExecutionTargetPriority();
        runtime.AddExecutionTarget(*arguments.execution_target);
        runtime.SetAllowFallback(false);
        model.LoadModelConfigWithRuntimeConfig(arguments.manifest_path.string(), runtime);
    }

    [[nodiscard]] infer::SFloatTensor PrepareImage(const cv::Mat& image,
                                                   const infer::STensorInfo& input_info)
    {
        if (input_info.dtype != "float32" || input_info.shape.size() != 4U ||
            input_info.shape[0] != 1 || input_info.shape[1] != 3 || input_info.shape[2] <= 0 ||
            input_info.shape[3] <= 0 || input_info.shape[2] > std::numeric_limits<int>::max() ||
            input_info.shape[3] > std::numeric_limits<int>::max())
        {
            throw std::runtime_error(
                "YOLO demo expects one float32 input with concrete shape [1,3,H,W].");
        }

        const auto height = static_cast<size_t>(input_info.shape[2]);
        const auto width = static_cast<size_t>(input_info.shape[3]);
        cv::Mat resized_image;
        cv::resize(image, resized_image,
                   cv::Size{static_cast<int>(width), static_cast<int>(height)}, 0.0, 0.0,
                   cv::INTER_LINEAR);
        if (!resized_image.isContinuous())
        {
            resized_image = resized_image.clone();
        }

        const auto* pixels = resized_image.ptr<uint8_t>();
        return infer::MakeNchwFloatTensorFromHwcAccessor(
            input_info.name, height, width, 3U, 1.0F / 255.0F, true,
            [pixels](const size_t index) { return pixels[index]; });
    }

    [[nodiscard]] std::string FormatShape(const std::vector<int64_t>& shape)
    {
        std::ostringstream stream;
        stream << '[';
        for (size_t index = 0; index < shape.size(); ++index)
        {
            if (index != 0U)
            {
                stream << ',';
            }
            stream << shape[index];
        }
        stream << ']';
        return stream.str();
    }

    void PrintDetections(const infer::SFloatTensor& output,
                         const std::vector<infer::SDetection2D>& detections)
    {
        std::cout << "output_name=" << output.name << '\n'
                  << "output_shape=" << FormatShape(output.shape) << '\n'
                  << "detections=" << detections.size() << '\n';
        for (size_t index = 0; index < detections.size(); ++index)
        {
            const infer::SDetection2D& detection = detections[index];
            std::cout << "detection[" << index
                      << "]=" << "class=" << detection.classification.class_id
                      << ",score=" << detection.classification.score
                      << ",cx=" << detection.bounds.center.x << ",cy=" << detection.bounds.center.y
                      << ",w=" << detection.bounds.size.width
                      << ",h=" << detection.bounds.size.height << '\n';
        }
    }
} // namespace

/**
 * @brief Run one image through the facade-backed YOLO integration.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on success, one for invalid input or inference failure.
 */
int main(const int argc, char** argv)
{
    try
    {
        const SArguments arguments = ParseArguments(argc, argv);
        GetLogger().info("Loading YOLO model configuration: ", arguments.manifest_path.string());
        GetLogger().debug("Input image=", arguments.image_path.string(),
                          ";score_threshold=", arguments.score_threshold,
                          ";max_detections=", arguments.max_detections);
        infer::CModelFacade model;
        LoadModel(arguments, model);
        if (model.GetRole() != "object_detection")
        {
            throw std::runtime_error("YOLO demo requires role=object_detection in the manifest.");
        }
        if (model.GetNumInputs() != 1U || model.GetNumOutputs() != 1U)
        {
            throw std::runtime_error(
                "YOLO demo currently requires one model input and one output.");
        }

        const cv::Mat image = cv::imread(arguments.image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::invalid_argument("Could not read input image: " +
                                        arguments.image_path.string());
        }

        const infer::SFloatTensor input = PrepareImage(image, model.GetInputInfo(0));
        const infer::SFloatTensor output = model.InferSingleFloatTensor(input);
        if (output.shape.empty() || output.shape.back() < 6)
        {
            throw std::runtime_error(
                "YOLO demo expects an attribute axis containing xywh, objectness, and classes.");
        }

        infer::SDetectionRowSchema detection_schema;
        detection_schema.box_encoding = infer::EBoundingBoxEncoding::center_xywh;
        detection_schema.box_coordinate_0_index = 0U;
        detection_schema.box_coordinate_1_index = 1U;
        detection_schema.box_coordinate_2_index = 2U;
        detection_schema.box_coordinate_3_index = 3U;
        detection_schema.objectness_index = 4;
        detection_schema.first_class_score_index = 5U;
        detection_schema.class_score_count = static_cast<size_t>(output.shape.back() - 5);
        const std::vector<infer::SDetection2D> detections = infer::DecodeDetectionRows(
            output, detection_schema, arguments.score_threshold, arguments.max_detections);

        std::cout << "role=" << model.GetRole() << '\n'
                  << "backend=" << model.GetBackendDetail() << '\n'
                  << "input_name=" << input.name << '\n'
                  << "input_shape=" << FormatShape(input.shape) << '\n';
        PrintDetections(output, detections);
        return 0;
    }
    catch (const TCLAP::ExitException& exit_request)
    {
        return exit_request.getExitStatus();
    }
    catch (const TCLAP::ArgException& error)
    {
        GetLogger().error("Invalid command line: ", error.error(), " for ", error.argId());
        return 1;
    }
    catch (const std::exception& error)
    {
        GetLogger().error(error.what());
        return 1;
    }
}
