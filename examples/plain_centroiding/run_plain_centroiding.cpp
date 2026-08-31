/**
 * @file run_plain_centroiding.cpp
 * @brief Run one decoded image through a plain centroiding ONNX model.
 *
 * The executable owns command-line collection, OpenCV image decoding, and
 * stable textual output. Model execution remains behind `CModelFacade`, while
 * example-side helpers own the selected model's grayscale and coordinate
 * semantics.
 */

#include "plain_centroiding_support.h"

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
    namespace demo = ptafdeploy::examples::plain_centroiding;
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    /** @brief Return the process-wide logger configured from the environment once. */
    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("run_plain_centroiding", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /** @brief Validated command-line state for one image inference. */
    struct SArguments
    {
        /** @brief Plain centroiding `.ptafmodel` manifest or raw `.onnx` artifact. */
        fs::path model_path{};

        /** @brief Image decoded by OpenCV into unsigned 8-bit grayscale. */
        fs::path image_path{};

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
            "Run one image through a plain image-only centroiding model. The model may be a "
            "role manifest or a raw ONNX artifact; image decoding, grayscale conversion, "
            "resize, and unit scaling are handled by this integration.",
            ' ',
            PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        // TCLAP remains private to argument collection. Runtime target parsing
        // uses the same backend-neutral API as manifests and other programs.
        TCLAP::UnlabeledValueArg<std::string> model_path(
            "model",
            "Path to a plain centroiding manifest or raw ONNX artifact",
            true,
            "",
            "model.ptafmodel|model.onnx",
            command);
        TCLAP::UnlabeledValueArg<std::string> image_path(
            "image",
            "Path to an image readable by OpenCV",
            true,
            "",
            "image.png",
            command);
        TCLAP::ValueArg<std::string> targets(
            "",
            "targets",
            "Comma-separated execution-target priority override",
            false,
            "",
            "cpu,cuda,tensorrt",
            command);
        TCLAP::ValueArg<int> device(
            "",
            "device",
            "Non-negative runtime device index override",
            false,
            0,
            "non-negative integer",
            command);
        TCLAP::ValueArg<int> intra_op_threads(
            "",
            "intra-op-threads",
            "Runtime intra-operation threads; zero uses its default",
            false,
            1,
            "non-negative integer",
            command);
        TCLAP::ValueArg<int> inter_op_threads(
            "",
            "inter-op-threads",
            "Runtime inter-operation threads; zero uses its default",
            false,
            1,
            "non-negative integer",
            command);
        TCLAP::SwitchArg no_fallback(
            "",
            "no-fallback",
            "Reject fallback below the requested execution targets",
            command,
            false);

        command.parse(argc, argv);

        SArguments arguments;
        arguments.model_path = model_path.getValue();
        arguments.image_path = image_path.getValue();
        arguments.runtime.SetDeviceId(device.getValue());
        arguments.runtime.SetThreadCounts(
            intra_op_threads.getValue(),
            inter_op_threads.getValue());
        arguments.runtime.SetAllowFallback(!no_fallback.getValue());
        arguments.runtime.SetLogId("run_plain_centroiding");
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
                model.LoadModelConfigWithRuntimeConfig(
                    arguments.model_path.string(),
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
                arguments.model_path.string(),
                infer::EModelRole::centroiding,
                arguments.runtime);
            return;
        }

        throw std::invalid_argument(
            "Plain centroiding expects a .ptafmodel manifest or .onnx artifact.");
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
    void PrintResult(const infer::CModelFacade& model,
                     const infer::SFloatTensor& input,
                     const infer::SFloatTensor& output,
                     const cv::Size image_size,
                     const demo::SCentroidResult& result)
    {
        std::cout << std::setprecision(9)
                  << "role=" << model.GetRole() << '\n'
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
 * @brief Run one image through a facade-backed plain centroiding model.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on success, one for invalid user input, or two for runtime failures.
 */
int main(const int argc, char** argv)
{
    try
    {
        const SArguments arguments = ParseArguments(argc, argv);
        GetLogger().info("Loading plain centroiding model: ", arguments.model_path.string());

        infer::CModelFacade model;
        LoadModel(arguments, model);
        if (model.GetRole() != "centroiding")
        {
            throw std::invalid_argument(
                "Plain centroiding requires role=centroiding in the manifest.");
        }
        if (model.GetNumInputs() != 1U || model.GetNumOutputs() != 1U)
        {
            throw std::invalid_argument(
                "Plain centroiding requires exactly one model input and one output.");
        }

        const cv::Mat image = cv::imread(arguments.image_path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            throw std::invalid_argument("Could not decode input image: " +
                                        arguments.image_path.string());
        }
        GetLogger().debug("Decoded grayscale image width=", image.cols, ", height=", image.rows);

        const infer::SFloatTensor input = demo::PrepareImageTensor(image, model.GetInputInfo(0));
        GetLogger().info("Running one centroiding inference.");
        const infer::SFloatTensor output = model.InferSingleFloatTensor(input);

        const cv::Size model_input_size{
            static_cast<int>(input.shape[3]),
            static_cast<int>(input.shape[2]),
        };
        const demo::SCentroidResult result = demo::DecodeCentroid(
            output,
            model_input_size,
            image.size());

        PrintResult(model, input, output, image.size(), result);
        GetLogger().info("Centroiding inference completed.");
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
