/**
 * @file run_centroiding.cpp
 * @brief Run independent image frames through a centroiding model with optional JSON output.
 *
 * The executable owns command-line collection, OpenCV image decoding, and
 * stable textual output. Model execution remains behind `CModelFacade`, while
 * example-side helpers own the selected model's grayscale and coordinate
 * semantics.
 */

#include "centroiding_io.h"
#include <chrono>
#include <memory>

#include <inference/inference_config_parsing.h>
#include <inference/model_facade.h>
#include <utils/logging/CLogger.h>

#include <opencv2/imgcodecs.hpp>
#include <tclap/CmdLine.h>

#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace demo = ptafdeploy::examples::centroiding_models;
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    /** @brief Return the process-wide logger configured from the environment once. */
    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("run_centroiding", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /** @brief Validated command-line state for one image or sequence invocation. */
    struct SArguments
    {
        /** @brief Centroiding `.ptafmodel` manifest or raw `.onnx` artifact. */
        fs::path model_path{};

        /** @brief One image or directory, decoded frame by frame into uint8 grayscale. */
        fs::path image_path{};
        /** @brief Optional new or empty output directory. */
        fs::path output_path{};
        /** @brief Whether annotated images accompany the JSON report. */
        bool overlays{false};

        /** @brief Backend-neutral runtime used for raw models or explicit overrides. */
        infer::SRuntimeConfig runtime{};

        /** @brief Whether explicit CLI runtime values replace manifest policy. */
        bool runtime_overridden{false};
    };

    /**
     * @brief Declare, collect, and validate one centroiding command line.
     * @param argc Number of command-line arguments.
     * @param argv Command-line argument values owned by the process.
     * @return Validated invocation state independent of TCLAP types.
     * @throws TCLAP::ExitException When help or version output requests an early exit.
     * @throws TCLAP::ArgException If TCLAP rejects an option or value.
     * @throws std::invalid_argument If a runtime value violates the public contract.
     */
    [[nodiscard]] SArguments ParseArguments(const int argc, char** argv)
    {
        TCLAP::CmdLine command(
            "Run an image or directory through an image-only centroiding model. The model may be a "
            "role manifest or a raw ONNX artifact; image decoding, grayscale conversion, "
            "resize, and unit scaling are handled by this integration.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        // TCLAP remains private to argument collection. Runtime target parsing
        // uses the same backend-neutral API as manifests and other programs.
        TCLAP::UnlabeledValueArg<std::string> model_path(
            "model", "Path to a centroiding manifest or raw ONNX artifact", true, "",
            "model.ptafmodel|model.onnx", command);
        TCLAP::UnlabeledValueArg<std::string> image_path(
            "input_path", "Path to one image or a directory of images", true, "", "image.png",
            command);
        TCLAP::ValueArg<std::string> targets("", "targets",
                                             "Comma-separated execution-target priority override",
                                             false, "", "cpu,cuda,tensorrt", command);
        TCLAP::ValueArg<int> device("", "device", "Non-negative runtime device index override",
                                    false, 0, "non-negative integer", command);
        TCLAP::ValueArg<int> intra_op_threads(
            "", "intra-op-threads", "Runtime intra-operation threads; zero uses its default", false,
            1, "non-negative integer", command);
        TCLAP::ValueArg<int> inter_op_threads(
            "", "inter-op-threads", "Runtime inter-operation threads; zero uses its default", false,
            1, "non-negative integer", command);
        TCLAP::SwitchArg no_fallback("", "no-fallback",
                                     "Reject fallback below the requested execution targets",
                                     command, false);

        TCLAP::ValueArg<std::string> output_path("", "output", "New or empty results directory",
                                                 false, "", "PATH", command);
        TCLAP::SwitchArg overlays("", "overlays", "Save original-resolution crosshairs", command,
                                  false);
        command.parse(argc, argv);
        if (overlays.getValue() && output_path.getValue().empty())
            throw std::invalid_argument("--overlays requires --output");

        SArguments arguments;
        arguments.model_path = model_path.getValue();
        arguments.image_path = image_path.getValue();
        arguments.output_path = output_path.getValue();
        arguments.overlays = overlays.getValue();
        arguments.runtime.SetDeviceId(device.getValue());
        arguments.runtime.SetThreadCounts(intra_op_threads.getValue(), inter_op_threads.getValue());
        arguments.runtime.SetAllowFallback(!no_fallback.getValue());
        arguments.runtime.SetLogId("run_centroiding");
        if (targets.isSet())
        {
            arguments.runtime.execution_target_priority =
                infer::ParseExecutionTargetPriority(targets.getValue());
        }

        arguments.runtime_overridden = targets.isSet() || device.isSet() ||
                                       intra_op_threads.isSet() || inter_op_threads.isSet() ||
                                       no_fallback.getValue();
        return arguments;
    }

    /**
     * @brief Load a manifest or raw ONNX through the role-level facade.
     * @param arguments Validated model path and runtime policy.
     * @param model Facade populated only if model loading succeeds.
     * @throws std::exception If the artifact, manifest, or runtime is invalid.
     */
    void LoadModel(const SArguments& arguments, infer::CModelFacade& model)
    {
        const std::string extension = arguments.model_path.extension().string();
        if (extension == ".ptafmodel")
        {
            if (arguments.runtime_overridden)
            {
                model.LoadModelConfigWithRuntimeConfig(arguments.model_path.string(),
                                                       arguments.runtime);
            }
            else
            {
                model.LoadModelConfig(arguments.model_path.string());
            }
            return;
        }
        if (extension == ".onnx")
        {
            model.LoadModelWithRoleAndRuntimeConfig(
                arguments.model_path.string(), infer::EModelRole::centroiding, arguments.runtime);
            return;
        }

        throw std::invalid_argument("Centroiding expects a .ptafmodel manifest or .onnx artifact.");
    }

    /**
     * @brief Format tensor dimensions as a stable bracketed list.
     * @param shape Dimensions in logical tensor order.
     * @return Text such as `[1,1,1536,2048]`.
     */
    [[nodiscard]] std::string FormatShape(const std::vector<int64_t>& shape)
    {
        std::ostringstream stream;
        stream << '[';
        for (size_t index = 0U; index < shape.size(); ++index)
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

    /**
     * @brief Print stable model, image, and centroid result fields to stdout.
     * @param model Loaded centroiding facade.
     * @param input Prepared model input tensor.
     * @param output Raw normalized centroid output.
     * @param image_size Original decoded image extent.
     * @param result Validated centroid in all supported coordinate spaces.
     */
    void PrintResult(const infer::CModelFacade& model, const infer::SFloatTensor& input,
                     const infer::SFloatTensor& output, const cv::Size image_size,
                     const demo::SCentroidResult& result)
    {
        std::cout << std::setprecision(9) << "role=" << model.GetRole() << '\n'
                  << "backend=" << model.GetBackendDetail() << '\n'
                  << "input_name=" << input.name << '\n'
                  << "input_shape=" << FormatShape(input.shape) << '\n'
                  << "image_size=[" << image_size.width << ',' << image_size.height << "]\n"
                  << "output_name=" << output.name << '\n'
                  << "output_shape=" << FormatShape(output.shape) << '\n'
                  << "centroid.normalized_x=" << result.normalized.x << '\n'
                  << "centroid.normalized_y=" << result.normalized.y << '\n'
                  << "centroid.model_input_x_px=" << result.model_input_pixels.x << '\n'
                  << "centroid.model_input_y_px=" << result.model_input_pixels.y << '\n'
                  << "centroid.original_x_px=" << result.original_image_pixels.x << '\n'
                  << "centroid.original_y_px=" << result.original_image_pixels.y << '\n';
    }
} // namespace

/**
 * @brief Run independent frames through a facade-backed centroiding model.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on success, one for invalid user input, or two for runtime failures.
 */
int main(const int argc, char** argv)
{
    try
    {
        const SArguments arguments = ParseArguments(argc, argv);
        const auto frames = demo::SelectFrames(arguments.image_path);
        std::unique_ptr<demo::CReport> report;
        if (!arguments.output_path.empty())
        {
            demo::PrepareOutput(arguments.output_path, arguments.image_path);
            report = std::make_unique<demo::CReport>(
                arguments.output_path, demo::MetadataJson(arguments.image_path, frames.size(),
                                                          nullptr, arguments.model_path));
        }
        size_t index = frames.size();
        std::string stage = "model_load";
        try
        {
            infer::CModelFacade model;
            LoadModel(arguments, model);
            if (model.GetRole() != "centroiding" || model.GetNumInputs() != 1U ||
                model.GetNumOutputs() != 1U)
                throw std::invalid_argument("Centroiding requires one image input and one output");
            if (report)
                report->metadata = demo::MetadataJson(arguments.image_path, frames.size(), &model,
                                                      arguments.model_path);
            if (arguments.overlays)
                fs::create_directory(arguments.output_path / "overlays");
            for (index = 0; index < frames.size(); ++index)
            {
                stage = "decode";
                const auto& source = frames[index];
                const cv::Mat image = cv::imread(source.string(), cv::IMREAD_GRAYSCALE);
                if (image.empty())
                    throw std::runtime_error("Cannot decode image: " + source.string());
                stage = "preprocessing";
                const auto input = demo::PrepareImageTensor(image, model.GetInputInfo(0));
                stage = "inference";
                const auto start = std::chrono::steady_clock::now();
                const auto output = model.InferSingleFloatTensor(input);
                const double duration = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - start)
                                            .count();
                stage = "decoding_output";
                const auto result = demo::DecodeCentroid(
                    output,
                    cv::Size{static_cast<int>(input.shape[3]), static_cast<int>(input.shape[2])},
                    image.size());
                std::cout << "frame.index=" << index
                          << "\nframe.source=" << source.filename().string()
                          << "\ninference_ms=" << duration << '\n';
                PrintResult(model, input, output, image.size(), result);
                std::string overlay;
                if (arguments.overlays)
                {
                    stage = "overlay";
                    std::ostringstream name;
                    name << "overlays/" << std::setfill('0') << std::setw(6) << index << '_'
                         << source.stem().string() << ".png";
                    overlay = name.str();
                    demo::SaveOverlay(source, arguments.output_path / overlay, result,
                                      image.size());
                }
                stage = "report";
                if (report)
                    report->Append(demo::FrameJson(index, source.filename(), image.size(), output,
                                                   result, duration, overlay));
            }
            if (report)
                report->Publish(true);
        }
        catch (const std::exception& error)
        {
            if (report)
            {
                rapidjson::StringBuffer buffer;
                demo::JsonWriter w(buffer);
                w.StartObject();
                demo::String(w, "stage", stage);
                const bool frame_failed = stage != "model_load" && index < frames.size();
                w.Key("frame_index");
                if (frame_failed)
                    w.Uint64(index);
                else
                    w.Null();
                w.Key("source");
                if (frame_failed)
                    w.String(frames[index].filename().string().c_str());
                else
                    w.Null();
                demo::String(w, "message", error.what());
                w.EndObject();
                try
                {
                    report->Publish(false, buffer.GetString());
                }
                catch (const std::exception& publication)
                {
                    GetLogger().error("Report publication failed: ", publication.what(),
                                      "; retained report/spool under ",
                                      arguments.output_path.string());
                }
            }
            throw;
        }
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
    catch (const std::invalid_argument& error)
    {
        GetLogger().error("Invalid input: ", error.what());
        return 1;
    }
    catch (const std::exception& error)
    {
        GetLogger().error("Centroiding failed: ", error.what());
        return 2;
    }
}
