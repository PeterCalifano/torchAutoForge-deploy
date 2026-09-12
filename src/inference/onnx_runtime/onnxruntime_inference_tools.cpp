/**
 * @file onnxruntime_inference_tools.cpp
 * @brief ONNX Runtime session management behind the generic inference API.
 */

#include "onnxruntime_inference_tools.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utils/logging/CLogger.h>
#include <utils/filesystem.h>

namespace ptafdeploy::inference::onnxruntime
{
    namespace
    {
        constexpr auto kOrtLoggingLevel = ORT_LOGGING_LEVEL_ERROR;

        [[nodiscard]] ptafdeploy::logging::CLogger& GetLogger()
        {
            static ptafdeploy::logging::CLogger logger(
                "onnxruntime", ptafdeploy::logging::ELogLevel::Warning,
                ptafdeploy::logging::ELogColorMode::Disabled, std::clog, std::clog);
            static const bool environment_applied = logger.setLevelFromEnvironment();
            static_cast<void>(environment_applied);
            return logger;
        }

        [[nodiscard]] ptafdeploy::inference::ETensorElementType
        ConvertOrtType(const ONNXTensorElementDataType dtype)
        {
            switch (dtype)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return ptafdeploy::inference::ETensorElementType::boolean;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return ptafdeploy::inference::ETensorElementType::uint8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                return ptafdeploy::inference::ETensorElementType::int8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                return ptafdeploy::inference::ETensorElementType::uint16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                return ptafdeploy::inference::ETensorElementType::int16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                return ptafdeploy::inference::ETensorElementType::uint32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return ptafdeploy::inference::ETensorElementType::int32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                return ptafdeploy::inference::ETensorElementType::uint64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return ptafdeploy::inference::ETensorElementType::int64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                return ptafdeploy::inference::ETensorElementType::float16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
                return ptafdeploy::inference::ETensorElementType::bfloat16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return ptafdeploy::inference::ETensorElementType::float32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                return ptafdeploy::inference::ETensorElementType::float64;
            default:
                throw std::runtime_error("Unsupported ONNX Runtime tensor element type.");
            }
        }

        [[nodiscard]] ONNXTensorElementDataType
        ConvertToOrtType(const ptafdeploy::inference::ETensorElementType dtype)
        {
            switch (dtype)
            {
            case ptafdeploy::inference::ETensorElementType::boolean:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            case ptafdeploy::inference::ETensorElementType::uint8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case ptafdeploy::inference::ETensorElementType::int8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
            case ptafdeploy::inference::ETensorElementType::uint16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
            case ptafdeploy::inference::ETensorElementType::int16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
            case ptafdeploy::inference::ETensorElementType::uint32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
            case ptafdeploy::inference::ETensorElementType::int32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            case ptafdeploy::inference::ETensorElementType::uint64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
            case ptafdeploy::inference::ETensorElementType::int64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            case ptafdeploy::inference::ETensorElementType::float16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            case ptafdeploy::inference::ETensorElementType::bfloat16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
            case ptafdeploy::inference::ETensorElementType::float32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case ptafdeploy::inference::ETensorElementType::float64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
            }

            throw std::runtime_error("Unsupported generic tensor element type.");
        }

        [[nodiscard]] std::string NormalizeProviderName(std::string provider_name)
        {
            std::transform(provider_name.begin(), provider_name.end(), provider_name.begin(),
                           [](const unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });

            return provider_name;
        }

        [[nodiscard]] std::vector<ptafdeploy::inference::EExecutionTarget>
        ResolveExecutionTargetPriority(const ptafdeploy::inference::SInferenceOptions& options)
        {
            return options.execution_target_priority.empty()
                       ? std::vector<
                             ptafdeploy::inference::
                                 EExecutionTarget>{ptafdeploy::inference::EExecutionTarget::
                                                       tensorrt,
                                                   ptafdeploy::inference::EExecutionTarget::cuda,
                                                   ptafdeploy::inference::EExecutionTarget::cpu}
                       : options.execution_target_priority;
        }

        [[nodiscard]] bool IsProviderAvailable(const std::vector<std::string>& available_providers,
                                               const std::string& provider_name)
        {
            const std::string normalized_target = NormalizeProviderName(provider_name);
            return std::any_of(
                available_providers.begin(), available_providers.end(),
                [&](const std::string& available_provider) {
                    return NormalizeProviderName(available_provider).find(normalized_target) !=
                           std::string::npos;
                });
        }

        [[nodiscard]] std::vector<std::string>
        ApplyOrtProviders(Ort::SessionOptions& session_options,
                          const ptafdeploy::inference::SInferenceOptions& options,
                          const std::vector<std::string>& available_providers)
        {
            const std::vector<ptafdeploy::inference::EExecutionTarget> target_priority =
                ResolveExecutionTargetPriority(options);
            std::vector<std::string> applied_providers;

            for (const ptafdeploy::inference::EExecutionTarget target : target_priority)
            {
                const std::string normalized_provider = ptafdeploy::inference::ToString(target);
                if (normalized_provider == "cpu")
                {
                    applied_providers.push_back("cpu");
                    break;
                }

                if (!IsProviderAvailable(available_providers, normalized_provider))
                {
                    if (!options.allow_fallback)
                    {
                        throw std::runtime_error(
                            "Requested execution target is not available through ONNX Runtime: " +
                            normalized_provider);
                    }
                    continue;
                }

                if (normalized_provider == "cuda")
                {
                    OrtCUDAProviderOptions cuda_options{};
                    cuda_options.device_id = options.device_id;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    applied_providers.push_back("cuda");
                    continue;
                }

                if (normalized_provider == "tensorrt")
                {
                    OrtTensorRTProviderOptions tensorrt_options{};
                    tensorrt_options.device_id = options.device_id;
                    session_options.AppendExecutionProvider_TensorRT(tensorrt_options);
                    applied_providers.push_back("tensorrt");
                    continue;
                }

                if (!options.allow_fallback)
                {
                    throw std::runtime_error("Unsupported ONNX Runtime execution target request: " +
                                             normalized_provider);
                }
            }

            if (applied_providers.empty())
            {
                applied_providers.push_back("cpu");
            }

            return applied_providers;
        }

        [[nodiscard]] Ort::SessionOptions
        MakeSessionOptions(const ptafdeploy::inference::SInferenceOptions& options,
                           const std::vector<std::string>& available_providers,
                           const bool apply_requested_providers,
                           std::vector<std::string>* applied_providers)
        {
            if (options.device_id < 0)
            {
                throw std::invalid_argument("ORT device_id must be non-negative.");
            }

            Ort::SessionOptions session_options;
            if (options.intra_op_num_threads > 0)
            {
                session_options.SetIntraOpNumThreads(options.intra_op_num_threads);
            }
            if (options.inter_op_num_threads > 0)
            {
                session_options.SetInterOpNumThreads(options.inter_op_num_threads);
            }
            if (options.enable_profiling)
            {
                session_options.EnableProfiling(options.log_id.c_str());
            }
            if (apply_requested_providers)
            {
                std::vector<std::string> providers =
                    ApplyOrtProviders(session_options, options, available_providers);
                if (applied_providers != nullptr)
                {
                    *applied_providers = std::move(providers);
                }
            }
            else if (applied_providers != nullptr)
            {
                *applied_providers = {"cpu"};
            }

            return session_options;
        }

        [[nodiscard]] ptafdeploy::inference::STensorDescriptor
        ExtractTensorDescriptor(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator,
                                const size_t index, const bool is_input)
        {
            Ort::AllocatedStringPtr name_ptr =
                is_input ? session.GetInputNameAllocated(index, allocator)
                         : session.GetOutputNameAllocated(index, allocator);
            const Ort::TypeInfo type_info =
                is_input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ptafdeploy::inference::STensorDescriptor descriptor;
            descriptor.name = name_ptr.get();
            descriptor.dtype = ConvertOrtType(tensor_info.GetElementType());
            descriptor.shape = tensor_info.GetShape();
            descriptor.location = ptafdeploy::inference::EMemoryLocation::host;
            return descriptor;
        }

        [[nodiscard]] std::vector<const char*>
        BuildNamePointers(const std::vector<ptafdeploy::inference::STensorDescriptor>& descriptors)
        {
            std::vector<const char*> names;
            names.reserve(descriptors.size());
            for (const auto& descriptor : descriptors)
            {
                names.push_back(descriptor.name.c_str());
            }
            return names;
        }

        [[nodiscard]] std::vector<Ort::Value> BuildOrderedOrtInputs(
            const std::vector<ptafdeploy::inference::STensorDescriptor>& expected_inputs,
            const std::vector<ptafdeploy::inference::STensorView>& provided_inputs)
        {
            Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value> ordered_inputs;
            ordered_inputs.reserve(expected_inputs.size());
            const std::vector<const ptafdeploy::inference::STensorView*> ordered_input_views =
                ptafdeploy::inference::OrderInputViews(expected_inputs, provided_inputs);

            for (size_t i = 0; i < expected_inputs.size(); ++i)
            {
                const auto& expected_descriptor = expected_inputs[i];
                const ptafdeploy::inference::STensorView& input_view = *ordered_input_views[i];

                ptafdeploy::inference::ValidateHostTensorView(expected_descriptor, input_view,
                                                              "ONNX Runtime");

                ordered_inputs.emplace_back(Ort::Value::CreateTensor(
                    memory_info, const_cast<void*>(input_view.data), input_view.bytes,
                    input_view.descriptor.shape.data(), input_view.descriptor.shape.size(),
                    ConvertToOrtType(input_view.descriptor.dtype)));
            }

            return ordered_inputs;
        }

        [[nodiscard]] ptafdeploy::inference::STensorBuffer
        CopyOrtTensorToOwnedBuffer(const ptafdeploy::inference::STensorDescriptor& base_descriptor,
                                   const Ort::Value& tensor_value)
        {
            if (!tensor_value.IsTensor())
            {
                throw std::runtime_error("Only dense tensor outputs are supported.");
            }

            auto tensor_info = tensor_value.GetTensorTypeAndShapeInfo();
            ptafdeploy::inference::STensorDescriptor descriptor = base_descriptor;
            descriptor.dtype = ConvertOrtType(tensor_info.GetElementType());
            descriptor.shape = tensor_info.GetShape();
            descriptor.location = ptafdeploy::inference::EMemoryLocation::host;

            ptafdeploy::inference::STensorBuffer output_buffer =
                ptafdeploy::inference::MakeOwnedTensorBuffer(descriptor);
            if (!output_buffer.storage.empty())
            {
                std::memcpy(output_buffer.storage.data(), tensor_value.GetTensorRawData(),
                            output_buffer.storage.size());
            }

            return output_buffer;
        }

        [[nodiscard]] std::string JoinProviders(const std::vector<std::string>& providers)
        {
            if (providers.empty())
            {
                return "none";
            }

            std::ostringstream stream;
            for (size_t i = 0; i < providers.size(); ++i)
            {
                if (i != 0U)
                {
                    stream << ",";
                }
                stream << providers[i];
            }

            return stream.str();
        }

        [[nodiscard]] std::string MakeBackendDetail(
            const std::vector<ptafdeploy::inference::EExecutionTarget>& requested_targets,
            const std::vector<std::string>& applied_providers,
            const std::vector<std::string>& available_providers, const int device_id)
        {
            std::ostringstream stream;
            stream << "requested_targets="
                   << ptafdeploy::inference::JoinExecutionTargets(requested_targets)
                   << ";applied_ort_providers=" << JoinProviders(applied_providers)
                   << ";available_ort_providers=" << JoinProviders(available_providers)
                   << ";device_id=" << device_id;
            return stream.str();
        }
    } // namespace

    CInferenceManager_ORT::CInferenceManager_ORT(
        const fs::path& model_path, const ptafdeploy::inference::SInferenceOptions& options)
    {
        LoadModel(model_path, options);
    }

    void CInferenceManager_ORT::LoadModel(const fs::path& model_path,
                                          const ptafdeploy::inference::SInferenceOptions& options)
    {
        ptafdeploy::utils::CheckFileExistsWithExt(model_path, "onnx", true);

        model_path_ = model_path;
        options_ = options;
        model_metadata_ = {};

        const std::vector<std::string> available_providers = GetAvailableProviders();
        exec_env_ = Ort::Env(kOrtLoggingLevel, options_.log_id.c_str());

        std::vector<std::string> applied_providers;
        session_options_ =
            MakeSessionOptions(options_, available_providers, true, &applied_providers);
        try
        {
            session_ptr_ = std::make_unique<Ort::Session>(exec_env_, model_path_.string().c_str(),
                                                          session_options_);
        }
        catch (const Ort::Exception& error)
        {
            if (!options_.allow_fallback)
            {
                throw;
            }

            GetLogger().warning("Requested ORT provider session failed; retrying on CPU: ",
                                error.what());
            applied_providers = {"cpu_fallback"};
            session_options_ = MakeSessionOptions(options_, available_providers, false, nullptr);
            session_ptr_ = std::make_unique<Ort::Session>(exec_env_, model_path_.string().c_str(),
                                                          session_options_);
        }

        const size_t num_inputs = session_ptr_->GetInputCount();
        const size_t num_outputs = session_ptr_->GetOutputCount();

        model_metadata_.inputs.reserve(num_inputs);
        model_metadata_.outputs.reserve(num_outputs);
        for (size_t i = 0; i < num_inputs; ++i)
        {
            model_metadata_.inputs.push_back(
                ExtractTensorDescriptor(*session_ptr_, allocator_, i, true));
        }
        for (size_t i = 0; i < num_outputs; ++i)
        {
            model_metadata_.outputs.push_back(
                ExtractTensorDescriptor(*session_ptr_, allocator_, i, false));
        }

        model_metadata_.backend = ptafdeploy::inference::EInferenceBackend::onnxruntime;
        model_metadata_.backend_detail =
            MakeBackendDetail(ResolveExecutionTargetPriority(options_), applied_providers,
                              available_providers, options_.device_id);
        GetLogger().info("Loaded ONNX model: ", model_path_.string());
        GetLogger().debug(model_metadata_.backend_detail, ";inputs=", model_metadata_.inputs.size(),
                          ";outputs=", model_metadata_.outputs.size());
    }

    const ptafdeploy::inference::SModelMetadata&
    CInferenceManager_ORT::GetModelMetadata() const noexcept
    {
        return model_metadata_;
    }

    std::vector<std::string> CInferenceManager_ORT::GetAvailableProviders()
    {
        auto providers = Ort::GetAvailableProviders();
        std::vector<std::string> available_providers;
        available_providers.reserve(providers.size());

        for (const auto& provider_name : providers)
        {
            available_providers.push_back(static_cast<std::string>(provider_name));
        }

        return available_providers;
    }

    bool CInferenceManager_ORT::IsExecutionTargetAvailable(
        const ptafdeploy::inference::EExecutionTarget target)
    {
        return IsProviderAvailable(GetAvailableProviders(),
                                   ptafdeploy::inference::ToString(target));
    }

    std::vector<ptafdeploy::inference::STensorBuffer> CInferenceManager_ORT::Infer(
        const std::vector<ptafdeploy::inference::STensorView>& inputs) const
    {
        if (session_ptr_ == nullptr)
        {
            throw std::runtime_error("ONNX Runtime session has not been initialized.");
        }

        GetLogger().trace("Running ORT inference with ", inputs.size(), " input(s) and ",
                          model_metadata_.outputs.size(), " requested output(s).");

        const std::vector<const char*> input_names = BuildNamePointers(model_metadata_.inputs);
        const std::vector<const char*> output_names = BuildNamePointers(model_metadata_.outputs);
        std::vector<Ort::Value> ordered_inputs =
            BuildOrderedOrtInputs(model_metadata_.inputs, inputs);

        Ort::RunOptions run_options;
        run_options.SetRunTag(options_.log_id.c_str());
        std::vector<Ort::Value> ort_outputs =
            session_ptr_->Run(run_options, input_names.data(), ordered_inputs.data(),
                              ordered_inputs.size(), output_names.data(), output_names.size());

        std::vector<ptafdeploy::inference::STensorBuffer> outputs;
        outputs.reserve(ort_outputs.size());
        for (size_t i = 0; i < ort_outputs.size(); ++i)
        {
            outputs.push_back(
                CopyOrtTensorToOwnedBuffer(model_metadata_.outputs[i], ort_outputs[i]));
        }

        return outputs;
    }
} // namespace ptafdeploy::inference::onnxruntime
