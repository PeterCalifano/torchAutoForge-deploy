/**
 * @file run_ort_inference.cpp
 * @brief Run one ONNX inference from prepared float32 tensors.
 */

#include <auxiliary/common_ops.h>
#include <inference/inference_config_parsing.h>
#include <inference/inference_manager.h>
#include <utils/logging/CLogger.h>

#include <tclap/CmdLine.h>
#include <tclap/MultiArg.h>

#include <algorithm>
#include <cctype>
#include <cmath>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    using TNamedSpecs = std::unordered_map<std::string, std::string>;

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("run_ort_inference", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /** @brief Parsed command-line values for one invocation. */
    struct SArguments
    {
        fs::path model_path{};
        infer::SRuntimeConfig runtime{};
        std::vector<std::string> input_specs{};
        std::vector<std::string> shape_specs{};
        std::vector<std::string> fill_specs{};
        std::optional<fs::path> output_dir{};
        size_t max_output_values{16U};
        bool metadata_only{false};
    };

    /** @brief One optional-name command-line specification. */
    struct SNamedSpec
    {
        std::string name{};
        std::string value{};
    };

    [[nodiscard]] SNamedSpec ParseNamedSpec(const std::string& specification,
                                            const std::string& option_name)
    {
        if (specification.empty())
        {
            throw std::invalid_argument(option_name + " expects a non-empty value.");
        }

        const size_t separator = specification.find('=');
        if (separator == std::string::npos)
        {
            return SNamedSpec{"", specification};
        }
        if (separator == 0U || separator + 1U == specification.size())
        {
            throw std::invalid_argument(option_name +
                                        " expects [tensor_name=]value with both sides non-empty.");
        }

        return SNamedSpec{specification.substr(0U, separator),
                          specification.substr(separator + 1U)};
    }

    [[nodiscard]] TNamedSpecs ResolveNamedSpecs(const std::vector<std::string>& specifications,
                                                const std::vector<infer::STensorInfo>& input_infos,
                                                const std::string& option_name)
    {
        std::unordered_set<std::string> input_names;
        for (const infer::STensorInfo& input_info : input_infos)
        {
            input_names.insert(input_info.name);
        }

        TNamedSpecs resolved;
        for (const std::string& specification : specifications)
        {
            SNamedSpec parsed = ParseNamedSpec(specification, option_name);
            if (parsed.name.empty())
            {
                if (input_infos.size() != 1U)
                {
                    throw std::invalid_argument(option_name +
                                                " requires tensor_name= for multi-input models.");
                }
                parsed.name = input_infos.front().name;
            }
            else if (!input_names.contains(parsed.name))
            {
                throw std::invalid_argument(option_name + " names unknown model input '" +
                                            parsed.name + "'.");
            }

            if (!resolved.emplace(parsed.name, std::move(parsed.value)).second)
            {
                throw std::invalid_argument(option_name + " specifies model input '" + parsed.name +
                                            "' more than once.");
            }
        }
        return resolved;
    }

    [[nodiscard]] int64_t ParsePositiveDimension(const std::string& value)
    {
        try
        {
            size_t parsed_chars = 0U;
            const long long parsed = std::stoll(value, &parsed_chars);
            if (parsed_chars != value.size() || parsed <= 0)
            {
                throw std::invalid_argument("not a positive integer");
            }
            return static_cast<int64_t>(parsed);
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("Invalid positive dimension in --shape: '" + value + "'.");
        }
    }

    [[nodiscard]] std::vector<int64_t> ParseShape(const std::string& value)
    {
        std::vector<int64_t> shape;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            if (token.empty())
            {
                throw std::invalid_argument("--shape dimensions must not be empty.");
            }
            shape.push_back(ParsePositiveDimension(token));
        }
        if (shape.empty())
        {
            throw std::invalid_argument("--shape expects at least one positive dimension.");
        }
        return shape;
    }

    [[nodiscard]] float ParseFiniteFloat(const std::string& value)
    {
        try
        {
            size_t parsed_chars = 0U;
            const float parsed = std::stof(value, &parsed_chars);
            if (parsed_chars != value.size() || !std::isfinite(parsed))
            {
                throw std::invalid_argument("not a finite float");
            }
            return parsed;
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("Invalid finite float in --fill: '" + value + "'.");
        }
    }

    [[nodiscard]] SArguments ParseArguments(const int argc, char** argv)
    {
        TCLAP::CmdLine command("Run one ONNX inference from prepared float32 "
                               "tensors. Static shapes come from model "
                               "metadata; dynamic inputs require --shape. Use "
                               "--metadata-only to inspect without "
                               "executing.",
                               ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        TCLAP::UnlabeledValueArg<std::string> model_path("model", "Path to the ONNX model artifact",
                                                         true, "", "model.onnx", command);
        TCLAP::MultiArg<std::string> inputs(
            "", "input", "Raw native-endian float32 input as [tensor_name=]path.f32", false,
            "[name=]path.f32", command);
        TCLAP::MultiArg<std::string> shapes("", "shape",
                                            "Concrete input shape as [tensor_name=]d0,d1,...",
                                            false, "[name=]d0,d1,...", command);
        TCLAP::MultiArg<std::string> fills(
            "", "fill", "Fill an input with one finite float as [tensor_name=]value", false,
            "[name=]value", command);
        TCLAP::ValueArg<std::string> targets("", "targets",
                                             "Comma-separated ORT execution-target priority", false,
                                             "", "cpu,cuda,tensorrt", command);
        TCLAP::ValueArg<int> device("", "device", "Non-negative runtime device index", false, 0,
                                    "non-negative integer", command);
        TCLAP::ValueArg<int> intra_op_threads("", "intra-op-threads",
                                              "ORT intra-operation threads; zero uses its default",
                                              false, 1, "non-negative integer", command);
        TCLAP::ValueArg<int> inter_op_threads("", "inter-op-threads",
                                              "ORT inter-operation threads; zero uses its default",
                                              false, 1, "non-negative integer", command);
        TCLAP::SwitchArg no_fallback("", "no-fallback",
                                     "Reject fallback below the requested execution targets",
                                     command, false);
        TCLAP::SwitchArg metadata_only("", "metadata-only",
                                       "Print the loaded tensor contract without executing",
                                       command, false);
        TCLAP::ValueArg<std::string> output_dir(
            "", "output-dir", "Write each output to INDEX_NAME.f32 in this directory", false, "",
            "directory", command);
        TCLAP::ValueArg<long long> max_output_values(
            "", "max-output-values", "Maximum preview values printed for each output", false, 16LL,
            "non-negative integer", command);

        command.parse(argc, argv);

        if (max_output_values.getValue() < 0 ||
            static_cast<unsigned long long>(max_output_values.getValue()) >
                std::numeric_limits<size_t>::max())
        {
            throw std::invalid_argument("--max-output-values expects a non-negative integer.");
        }

        SArguments arguments;
        arguments.model_path = model_path.getValue();
        arguments.input_specs = inputs.getValue();
        arguments.shape_specs = shapes.getValue();
        arguments.fill_specs = fills.getValue();
        arguments.max_output_values = static_cast<size_t>(max_output_values.getValue());
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

        arguments.runtime.SetDeviceId(device.getValue());
        arguments.runtime.SetThreadCounts(intra_op_threads.getValue(), inter_op_threads.getValue());
        arguments.runtime.SetAllowFallback(!no_fallback.getValue());
        arguments.runtime.SetLogId("run_ort_inference");
        if (targets.isSet())
        {
            arguments.runtime.execution_target_priority =
                infer::ParseExecutionTargetPriority(targets.getValue());
        }
        return arguments;
    }

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

    [[nodiscard]] std::vector<int64_t> ResolveShape(const infer::STensorInfo& input_info,
                                                    const TNamedSpecs& shape_specs)
    {
        std::vector<int64_t> shape = input_info.shape;
        if (const auto shape_spec = shape_specs.find(input_info.name);
            shape_spec != shape_specs.end())
        {
            shape = ParseShape(shape_spec->second);
            if (shape.size() != input_info.shape.size())
            {
                throw std::invalid_argument("--shape for input '" + input_info.name +
                                            "' has rank " + std::to_string(shape.size()) +
                                            ", but the model reports rank " +
                                            std::to_string(input_info.shape.size()) + ".");
            }
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

    [[nodiscard]] std::vector<float> ReadFloatTensor(const fs::path& path,
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

    [[nodiscard]] std::vector<infer::SFloatTensor> PrepareInputs(
        const infer::CInferenceManager& manager, const SArguments& arguments)
    {
        const std::vector<infer::STensorInfo> input_infos = manager.GetInputInfos();
        const TNamedSpecs input_specs =
            ResolveNamedSpecs(arguments.input_specs, input_infos, "--input");
        const TNamedSpecs shape_specs =
            ResolveNamedSpecs(arguments.shape_specs, input_infos, "--shape");
        const TNamedSpecs fill_specs =
            ResolveNamedSpecs(arguments.fill_specs, input_infos, "--fill");

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

            const bool has_file = input_specs.contains(input_info.name);
            const bool has_fill = fill_specs.contains(input_info.name);
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
                values = ReadFloatTensor(input_specs.at(input_info.name), element_count,
                                         input_info.name);
            }
            else
            {
                values.assign(element_count, ParseFiniteFloat(fill_specs.at(input_info.name)));
            }

            GetLogger().debug("Prepared input ", input_info.name, " shape=", FormatShape(shape),
                              " elements=", element_count);
            inputs.emplace_back(input_info.name, std::move(shape), std::move(values));
        }
        return inputs;
    }

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

    [[nodiscard]] fs::path WriteOutput(const infer::SFloatTensor& output, const size_t index,
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

    void PrintOutputs(const std::vector<infer::SFloatTensor>& outputs,
                      const size_t max_output_values, const std::optional<fs::path>& output_dir)
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

        const std::vector<infer::SFloatTensor> inputs = PrepareInputs(inference_manager, arguments);
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
