/**
 * @file run_ort_inference.cpp
 * @brief Run one ONNX inference from prepared float32 tensors.
 *
 * The executable owns CLI collection, raw float32 file IO, and stable textual
 * output. Model loading and inference remain behind `CInferenceManager`, while
 * reusable value and tensor-specification grammar comes from the installed
 * parsing APIs.
 */

#include <auxiliary/common_ops.h>
#include <inference/inference_config_parsing.h>
#include <inference/inference_manager.h>
#include <inference/inference_tensor_parsing.h>
#include <utils/logging/CLogger.h>
#include <utils/value_parsing.h>

#include <tclap/CmdLine.h>
#include <tclap/MultiArg.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;
    namespace parsing = ptafdeploy::parsing;

    using TNamedValues = std::vector<parsing::SNamedValue>;

    /** @brief Return the process-wide logger configured from the environment once. */
    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("run_ort_inference", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /** @brief Validated command-line state for one metadata or inference invocation. */
    struct SArguments
    {
        /** @brief ONNX artifact requested by the user. */
        fs::path model_path{};

        /** @brief Backend-neutral execution-provider and threading policy. */
        infer::SRuntimeConfig runtime{};

        /** @brief Repeated `[name=]path.f32` input specifications. */
        std::vector<std::string> input_specs{};

        /** @brief Repeated `[name=]d0,d1,...` concrete-shape specifications. */
        std::vector<std::string> shape_specs{};

        /** @brief Repeated `[name=]value` deterministic-fill specifications. */
        std::vector<std::string> fill_specs{};

        /** @brief Optional directory receiving raw float32 output files. */
        std::optional<fs::path> output_dir{};

        /** @brief Maximum values printed in each output preview. */
        size_t max_output_values{16U};

        /** @brief Whether to stop after printing model metadata. */
        bool metadata_only{false};
    };

    /**
     * @brief Declare, collect, and cross-validate one command line.
     * @param argc Number of command-line arguments.
     * @param argv Command-line argument values owned by the process.
     * @return Validated invocation state independent of TCLAP types.
     * @throws TCLAP::ExitException When help or version output requests an early exit.
     * @throws TCLAP::ArgException If TCLAP rejects an option or value.
     * @throws std::invalid_argument If option combinations violate the program contract.
     */
    [[nodiscard]] SArguments ParseArguments(const int argc, char** argv)
    {
        TCLAP::CmdLine command("Run one ONNX inference from prepared float32 "
                               "tensors. Static shapes come from model "
                               "metadata; dynamic inputs require --shape. Use "
                               "--metadata-only to inspect without "
                               "executing.",
                               ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        // TCLAP remains a private collection and help layer; reusable value
        // syntax is parsed only after model metadata is available.
        TCLAP::UnlabeledValueArg<std::string> model_path(
            "model",
            "Path to the ONNX model artifact",
            true,
            "",
            "model.onnx",
            command);
        TCLAP::MultiArg<std::string> inputs(
            "",
            "input",
            "Raw native-endian float32 input as [tensor_name=]path.f32",
            false,
            "[name=]path.f32",
            command);
        TCLAP::MultiArg<std::string> shapes(
            "",
            "shape",
            "Concrete input shape as [tensor_name=]d0,d1,...",
            false,
            "[name=]d0,d1,...",
            command);
        TCLAP::MultiArg<std::string> fills(
            "",
            "fill",
            "Fill an input with one finite float as [tensor_name=]value",
            false,
            "[name=]value",
            command);
        TCLAP::ValueArg<std::string> targets(
            "",
            "targets",
            "Comma-separated ORT execution-target priority",
            false,
            "",
            "cpu,cuda,tensorrt",
            command);
        TCLAP::ValueArg<int> device(
            "",
            "device",
            "Non-negative runtime device index",
            false,
            0,
            "non-negative integer",
            command);
        TCLAP::ValueArg<int> intra_op_threads(
            "",
            "intra-op-threads",
            "ORT intra-operation threads; zero uses its default",
            false,
            1,
            "non-negative integer",
            command);
        TCLAP::ValueArg<int> inter_op_threads(
            "",
            "inter-op-threads",
            "ORT inter-operation threads; zero uses its default",
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
        TCLAP::SwitchArg metadata_only(
            "",
            "metadata-only",
            "Print the loaded tensor contract without executing",
            command,
            false);
        TCLAP::ValueArg<std::string> output_dir(
            "",
            "output-dir",
            "Write each output to INDEX_NAME.f32 in this directory",
            false,
            "",
            "directory",
            command);
        TCLAP::ValueArg<long long> max_output_values(
            "",
            "max-output-values",
            "Maximum preview values printed for each output",
            false,
            16LL,
            "non-negative integer",
            command);

        command.parse(argc, argv);

        const long long requested_preview_count = max_output_values.getValue();
        if (requested_preview_count < 0 ||
            static_cast<unsigned long long>(requested_preview_count) >
                std::numeric_limits<size_t>::max())
        {
            throw std::invalid_argument("--max-output-values expects a non-negative integer.");
        }

        SArguments arguments;
        arguments.model_path = model_path.getValue();
        arguments.input_specs = inputs.getValue();
        arguments.shape_specs = shapes.getValue();
        arguments.fill_specs = fills.getValue();
        arguments.max_output_values = static_cast<size_t>(requested_preview_count);
        arguments.metadata_only = metadata_only.getValue();
        if (output_dir.isSet())
        {
            arguments.output_dir = fs::path(output_dir.getValue());
        }
        if (arguments.metadata_only &&
            (!arguments.input_specs.empty() || !arguments.shape_specs.empty() ||
             !arguments.fill_specs.empty() || arguments.output_dir.has_value() ||
             max_output_values.isSet()))
        {
            throw std::invalid_argument(
                "--metadata-only cannot be combined with tensor or output options.");
        }

        // Translate collected values into the backend-neutral runtime contract.
        arguments.runtime.SetDeviceId(device.getValue());
        arguments.runtime.SetThreadCounts(
            intra_op_threads.getValue(),
            inter_op_threads.getValue());
        arguments.runtime.SetAllowFallback(!no_fallback.getValue());
        arguments.runtime.SetLogId("run_ort_inference");
        if (targets.isSet())
        {
            arguments.runtime.execution_target_priority =
                infer::ParseExecutionTargetPriority(targets.getValue());
        }
        return arguments;
    }

    /**
     * @brief Format dimensions as a stable comma-separated bracketed list.
     * @param shape Dimensions to format in logical tensor order.
     * @return Text such as `[1,3,640,640]`.
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
     * @brief Find one resolved optional-name value without copying it.
     * @param values Values already resolved to model input names.
     * @param name Exact model input name to find.
     * @return Pointer into `values`, or `nullptr` when the input was not specified.
     * @note The pointer remains valid only while `values` is alive and unmodified.
     */
    [[nodiscard]] const std::string* FindNamedValue(const TNamedValues& values,
                                                    const std::string_view name)
    {
        const auto match =
            std::find_if(values.begin(), values.end(), [name](const parsing::SNamedValue& value) {
                return value.name == name;
            });
        return match == values.end() ? nullptr : &match->value;
    }

    /**
     * @brief Print stable line-oriented model metadata to standard output.
     * @param manager Loaded inference facade providing backend and tensor metadata.
     */
    void PrintMetadata(const infer::CInferenceManager& manager)
    {
        const std::vector<infer::STensorInfo> inputs = manager.GetInputInfos();
        const std::vector<infer::STensorInfo> outputs = manager.GetOutputInfos();

        std::cout << "backend=" << manager.GetBackendDetail() << '\n'
                  << "inputs=" << inputs.size() << '\n';
        for (size_t index = 0U; index < inputs.size(); ++index)
        {
            std::cout << "model.input[" << index << "].name=" << inputs[index].name << '\n'
                      << "model.input[" << index << "].dtype=" << inputs[index].dtype << '\n'
                      << "model.input[" << index << "].shape=" << FormatShape(inputs[index].shape)
                      << '\n';
        }

        std::cout << "outputs=" << outputs.size() << '\n';
        for (size_t index = 0U; index < outputs.size(); ++index)
        {
            std::cout << "model.output[" << index << "].name=" << outputs[index].name << '\n'
                      << "model.output[" << index << "].dtype=" << outputs[index].dtype << '\n'
                      << "model.output[" << index << "].shape=" << FormatShape(outputs[index].shape)
                      << '\n';
        }
    }

    /**
     * @brief Resolve and validate one runtime input shape against model metadata.
     * @param input_info Declared model input name, rank, and static/dynamic dimensions.
     * @param shape_specs User overrides already resolved to model input names.
     * @return A positive concrete shape compatible with every declared static dimension.
     * @throws std::invalid_argument If an override changes rank or a static dimension, or if a
     * dynamic dimension remains unresolved.
     */
    [[nodiscard]] std::vector<int64_t> ResolveShape(
        const infer::STensorInfo& input_info,
        const TNamedValues& shape_specs)
    {
        std::vector<int64_t> shape = input_info.shape;
        if (const std::string* shape_spec = FindNamedValue(shape_specs, input_info.name))
        {
            shape = infer::ParseTensorShape(*shape_spec,
                                            "--shape for input '" + input_info.name + "'");
            if (shape.size() != input_info.shape.size())
            {
                throw std::invalid_argument("--shape for input '" + input_info.name +
                                            "' has rank " + std::to_string(shape.size()) +
                                            ", but the model reports rank " +
                                            std::to_string(input_info.shape.size()) + ".");
            }
            // Explicit shapes may resolve dynamic dimensions but cannot alter
            // the static dimensions declared by the loaded model.
            for (size_t index = 0U; index < shape.size(); ++index)
            {
                if (input_info.shape[index] > 0 && shape[index] != input_info.shape[index])
                {
                    throw std::invalid_argument("--shape for input '" + input_info.name +
                                                "' changes static dimension " +
                                                std::to_string(index) + ".");
                }
            }
        }

        if (std::any_of(shape.begin(), shape.end(),
                        [](const int64_t dimension) { return dimension <= 0; }))
        {
            throw std::invalid_argument("Input '" + input_info.name +
                                        "' has dynamic dimensions; provide --shape " +
                                        input_info.name + "=d0,d1,... .");
        }
        return shape;
    }

    /**
     * @brief Read one dense native-endian float32 tensor from a headerless file.
     * @param path Regular file containing exactly `element_count` float values.
     * @param element_count Expected number of tensor elements.
     * @param input_name Model input name used in diagnostics.
     * @return Owned tensor values in file order.
     * @throws std::invalid_argument If the path or exact file size violates the input contract.
     * @throws std::filesystem::filesystem_error If path metadata cannot be inspected.
     * @throws std::overflow_error If the byte count cannot be represented by the IO interface.
     * @throws std::runtime_error If the complete file cannot be opened or read.
     */
    [[nodiscard]] std::vector<float> ReadFloatTensor(
        const fs::path& path,
        const size_t element_count,
        const std::string& input_name)
    {
        if (!fs::is_regular_file(path))
        {
            throw std::invalid_argument("Raw input for '" + input_name +
                                        "' is not a regular file: " + path.string());
        }
        if (element_count > std::numeric_limits<size_t>::max() / sizeof(float))
        {
            throw std::overflow_error("Raw input byte count exceeds size_t capacity.");
        }

        // Validate cardinality before allocating or opening the file so malformed
        // inputs cannot produce partial tensor state.
        const size_t expected_bytes = element_count * sizeof(float);
        const uintmax_t actual_bytes = fs::file_size(path);
        if (actual_bytes != expected_bytes)
        {
            throw std::invalid_argument("Raw input for '" + input_name + "' contains " +
                                        std::to_string(actual_bytes) + " bytes; expected " +
                                        std::to_string(expected_bytes) + ".");
        }
        if (expected_bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::overflow_error("Raw input is too large for one stream read.");
        }

        std::vector<float> values(element_count);
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Could not open raw input: " + path.string());
        }
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(expected_bytes));
        if (!input)
        {
            throw std::runtime_error("Could not read complete raw input: " + path.string());
        }
        return values;
    }

    /**
     * @brief Build all wrapper-safe float tensors required by the loaded model.
     * @param manager Loaded inference facade providing authoritative input metadata.
     * @param arguments Validated CLI state containing sources, fills, and shape overrides.
     * @return Inputs in model-declared order with owned concrete shapes and values.
     * @throws std::invalid_argument If names, dtypes, shapes, or source cardinality are invalid.
     * @throws std::filesystem::filesystem_error If raw-input metadata cannot be inspected.
     * @throws std::overflow_error If a tensor size exceeds supported host limits.
     * @throws std::runtime_error If a raw input cannot be read completely.
     */
    [[nodiscard]] std::vector<infer::SFloatTensor> PrepareInputs(
        const infer::CInferenceManager& manager, const SArguments& arguments)
    {
        // Resolve every repeated CLI grammar once against the authoritative
        // model contract before materializing any tensor storage.
        const std::vector<infer::STensorInfo> input_infos = manager.GetInputInfos();
        const TNamedValues input_specs =
            infer::ResolveNamedTensorValues(arguments.input_specs, input_infos, "--input");
        const TNamedValues shape_specs =
            infer::ResolveNamedTensorValues(arguments.shape_specs, input_infos, "--shape");
        const TNamedValues fill_specs =
            infer::ResolveNamedTensorValues(arguments.fill_specs, input_infos, "--fill");

        std::vector<infer::SFloatTensor> inputs;
        inputs.reserve(input_infos.size());
        for (const infer::STensorInfo& input_info : input_infos)
        {
            if (input_info.dtype != "float32")
            {
                throw std::invalid_argument("Input '" + input_info.name + "' uses " +
                                            input_info.dtype +
                                            "; run_ort_inference accepts float32 only.");
            }

            // Each model input must select exactly one data source; shape
            // overrides remain independent of whether data comes from file or fill.
            const std::string* input_spec = FindNamedValue(input_specs, input_info.name);
            const std::string* fill_spec = FindNamedValue(fill_specs, input_info.name);
            const bool has_file = input_spec != nullptr;
            const bool has_fill = fill_spec != nullptr;
            if (has_file == has_fill)
            {
                throw std::invalid_argument("Input '" + input_info.name +
                                            "' requires exactly one --input or --fill source.");
            }

            std::vector<int64_t> shape = ResolveShape(input_info, shape_specs);
            const size_t element_count = infer::ComputeElementCount(shape);
            std::vector<float> values;
            if (has_file)
            {
                values = ReadFloatTensor(*input_spec, element_count, input_info.name);
            }
            else
            {
                const float fill_value = parsing::ParseFiniteFloat(
                    *fill_spec,
                    "--fill for input '" + input_info.name + "'");
                values.assign(element_count, fill_value);
            }

            GetLogger().debug("Prepared input ", input_info.name, " shape=", FormatShape(shape),
                              " elements=", element_count);
            inputs.emplace_back(input_info.name, std::move(shape), std::move(values));
        }
        return inputs;
    }

    /**
     * @brief Convert a model output name into a portable filename component.
     * @param name Model-provided output name copied for in-place sanitization.
     * @return A non-empty name containing only alphanumeric, hyphen, or underscore characters.
     */
    [[nodiscard]] std::string SanitizeFilename(std::string name)
    {
        for (char& character : name)
        {
            const auto byte = static_cast<unsigned char>(character);
            if (!std::isalnum(byte) && character != '-' && character != '_')
            {
                character = '_';
            }
        }
        return name.empty() ? "output" : name;
    }

    /**
     * @brief Persist one output as a headerless native-endian float32 file.
     * @param output Named output tensor whose values are written in logical order.
     * @param index Stable output index used to avoid filename collisions.
     * @param output_dir Existing destination directory.
     * @return Complete path of the written output.
     * @throws std::overflow_error If the output size exceeds the stream interface.
     * @throws std::runtime_error If the output cannot be opened or written completely.
     */
    [[nodiscard]] fs::path WriteOutput(
        const infer::SFloatTensor& output,
        const size_t index,
        const fs::path& output_dir)
    {
        if (output.values.size() >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))
        {
            throw std::overflow_error("Output is too large for one stream write.");
        }

        const fs::path output_path =
            output_dir / (std::to_string(index) + "_" + SanitizeFilename(output.name) + ".f32");
        std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error("Could not open output file: " + output_path.string());
        }

        const size_t byte_count = output.values.size() * sizeof(float);
        file.write(reinterpret_cast<const char*>(output.values.data()),
                   static_cast<std::streamsize>(byte_count));
        if (!file)
        {
            throw std::runtime_error("Could not write complete output file: " +
                                     output_path.string());
        }
        return output_path;
    }

    /**
     * @brief Print stable output summaries and optionally persist raw tensors.
     * @param outputs Inference outputs in model order.
     * @param max_output_values Maximum values included in each standard-output preview.
     * @param output_dir Optional directory created before writing raw output files.
     * @throws std::invalid_argument If an existing output path is not a directory.
     * @throws std::filesystem::filesystem_error If directory creation fails.
     * @throws std::overflow_error If an output is too large for stream IO.
     * @throws std::runtime_error If an output file cannot be written completely.
     */
    void PrintOutputs(
        const std::vector<infer::SFloatTensor>& outputs,
        const size_t max_output_values,
        const std::optional<fs::path>& output_dir)
    {
        if (output_dir.has_value())
        {
            if (fs::exists(*output_dir) && !fs::is_directory(*output_dir))
            {
                throw std::invalid_argument("--output-dir is not a directory: " +
                                            output_dir->string());
            }
            fs::create_directories(*output_dir);
        }

        std::cout << std::setprecision(9);
        for (size_t index = 0U; index < outputs.size(); ++index)
        {
            const infer::SFloatTensor& output = outputs[index];
            std::cout << "output[" << index << "].name=" << output.name << '\n'
                      << "output[" << index << "].shape=" << FormatShape(output.shape) << '\n'
                      << "output[" << index << "].elements=" << output.values.size() << '\n'
                      << "output[" << index << "].values=";

            const size_t preview_count = std::min(max_output_values, output.values.size());
            for (size_t value_index = 0U; value_index < preview_count; ++value_index)
            {
                if (value_index != 0U)
                {
                    std::cout << ',';
                }
                std::cout << output.values[value_index];
            }
            if (preview_count < output.values.size())
            {
                std::cout << ",...";
            }
            std::cout << '\n';

            if (output_dir.has_value())
            {
                std::cout << "output[" << index
                          << "].file=" << WriteOutput(output, index, *output_dir).string() << '\n';
            }
        }
    }
} // namespace

/**
 * @brief Inspect or execute one ONNX model through the generic inference
 * facade.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on success, one for invalid user input, or two for runtime
 * failures.
 */
int main(const int argc, char** argv)
{
    try
    {
        const SArguments arguments = ParseArguments(argc, argv);
        GetLogger().info("Loading ONNX artifact: ", arguments.model_path.string());

        infer::CInferenceManager inference_manager;
        inference_manager.LoadModelWithRuntimeConfig(arguments.model_path.string(),
                                                     arguments.runtime);
        PrintMetadata(inference_manager);
        if (arguments.metadata_only)
        {
            GetLogger().info("Metadata inspection completed without inference.");
            return 0;
        }

        const std::vector<infer::SFloatTensor> inputs =
            PrepareInputs(inference_manager, arguments);
        GetLogger().info("Running one inference with ", inputs.size(), " input tensor(s).");
        const std::vector<infer::SFloatTensor> outputs =
            inference_manager.InferFloatTensors(inputs);
        PrintOutputs(outputs, arguments.max_output_values, arguments.output_dir);
        GetLogger().info("Inference completed with ", outputs.size(), " output tensor(s).");
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
        GetLogger().error("Inference failed: ", error.what());
        return 2;
    }
}
