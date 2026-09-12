/**
 * @file inference_manager.cpp
 * @brief Backend-neutral artifact dispatch and wrapper-safe inference facade.
 */

#include "inference_manager.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utils/logging/CLogger.h>

namespace ptafdeploy::inference
{
    namespace
    {
        [[nodiscard]] ptafdeploy::logging::CLogger& GetLogger()
        {
            static ptafdeploy::logging::CLogger logger(
                "inference_manager", ptafdeploy::logging::ELogLevel::Warning,
                ptafdeploy::logging::ELogColorMode::Disabled, std::clog, std::clog);
            static const bool environment_applied = logger.setLevelFromEnvironment();
            static_cast<void>(environment_applied);
            return logger;
        }

        [[nodiscard]] EModelArtifact DetectArtifactTypeFromPath(const fs::path& model_path)
        {
            const std::string extension = model_path.extension().string();
            if (extension == ".onnx")
            {
                return EModelArtifact::onnx;
            }
            if (extension == ".engine" || extension == ".plan")
            {
                return EModelArtifact::tensorrt_engine;
            }

            throw std::invalid_argument("Unsupported model artifact extension: " +
                                        model_path.string());
        }

        [[nodiscard]] SInferenceOptions MakeInferenceOptions(const SRuntimeConfig& runtime_config)
        {
            SInferenceOptions options;
            options.backend = runtime_config.backend;
            options.artifact = runtime_config.artifact;
            options.allow_fallback = runtime_config.allow_fallback;
            options.execution_target_priority = runtime_config.execution_target_priority;
            options.device_id = runtime_config.device_id;
            options.intra_op_num_threads = runtime_config.intra_op_num_threads;
            options.inter_op_num_threads = runtime_config.inter_op_num_threads;
            options.enable_profiling = runtime_config.enable_profiling;
            options.log_id = runtime_config.log_id;
            options.tensorrt_optimization_profile_index =
                runtime_config.tensorrt_optimization_profile_index;
            return options;
        }

        void ValidateInferenceOptions(const SInferenceOptions& options)
        {
            if (options.device_id < 0)
            {
                throw std::invalid_argument("Runtime device_id must be non-negative.");
            }
            if (options.intra_op_num_threads < 0 || options.inter_op_num_threads < 0)
            {
                throw std::invalid_argument("Runtime thread counts must be non-negative.");
            }
            if (options.tensorrt_optimization_profile_index < 0)
            {
                throw std::invalid_argument(
                    "TensorRT optimization profile index must be non-negative.");
            }
        }

        [[nodiscard]] EModelArtifact ResolveArtifactType(const fs::path& model_path,
                                                         const SInferenceOptions& options)
        {
            const EModelArtifact detected_artifact = DetectArtifactTypeFromPath(model_path);
            if (options.artifact != EModelArtifact::auto_artifact &&
                options.artifact != detected_artifact)
            {
                throw std::invalid_argument(
                    "artifact selection does not match file extension: selected=" +
                    ToString(options.artifact) + ", detected=" + ToString(detected_artifact));
            }

            return detected_artifact;
        }

        void ValidateBackendArtifactCompatibility(const EInferenceBackend backend,
                                                  const EModelArtifact artifact)
        {
            const bool incompatible_onnx =
                backend == EInferenceBackend::onnxruntime && artifact != EModelArtifact::onnx;
            const bool incompatible_tensorrt = backend == EInferenceBackend::tensorrt_engine &&
                                               artifact != EModelArtifact::tensorrt_engine;
            if (incompatible_onnx || incompatible_tensorrt)
            {
                throw std::invalid_argument("incompatible backend and artifact: backend=" +
                                            ToString(backend) + ", artifact=" + ToString(artifact));
            }
        }

        [[nodiscard]] const STensorDescriptor&
        GetTensorDescriptorAt(const std::vector<STensorDescriptor>& descriptors, const size_t index,
                              const char* kind)
        {
            if (index >= descriptors.size())
            {
                throw std::out_of_range(std::string(kind) + " tensor index is out of range.");
            }

            return descriptors[index];
        }

        [[nodiscard]] std::vector<STensorInfo>
        MakeTensorInfos(const std::vector<STensorDescriptor>& descriptors)
        {
            std::vector<STensorInfo> infos;
            infos.reserve(descriptors.size());
            for (const STensorDescriptor& descriptor : descriptors)
            {
                infos.push_back(MakeTensorInfo(descriptor));
            }

            return infos;
        }

        [[nodiscard]] STensorView MakeFloatTensorView(const SFloatTensor& input)
        {
            STensorView view;
            view.descriptor.name = input.name;
            view.descriptor.dtype = ETensorElementType::float32;
            view.descriptor.shape = input.shape;
            view.descriptor.location = EMemoryLocation::host;
            view.data = input.values.empty() ? nullptr : input.values.data();
            view.bytes = input.values.size() * sizeof(float);
            return view;
        }

        [[nodiscard]] SFloatTensor MakeFloatTensor(const STensorBuffer& buffer)
        {
            if (buffer.descriptor.dtype != ETensorElementType::float32)
            {
                throw std::runtime_error(
                    "Wrapper float inference only supports float32 tensor outputs.");
            }
            if ((buffer.storage.size() % sizeof(float)) != 0U)
            {
                throw std::runtime_error(
                    "Float tensor output byte count is not divisible by sizeof(float).");
            }

            SFloatTensor output;
            output.name = buffer.descriptor.name;
            output.shape = buffer.descriptor.shape;
            output.values.resize(buffer.storage.size() / sizeof(float));
            if (!output.values.empty())
            {
                std::memcpy(output.values.data(), buffer.storage.data(), buffer.storage.size());
            }

            return output;
        }
    } // namespace

    CInferenceManager::CInferenceManager(const fs::path& model_path,
                                         const SInferenceOptions& options)
    {
        LoadModel(model_path, options);
    }

    CInferenceManager::CInferenceManager(const std::string& model_path)
    {
        LoadModel(model_path);
    }

    void CInferenceManager::LoadModel(const fs::path& model_path, const SInferenceOptions& options)
    {
        ValidateInferenceOptions(options);
        const EModelArtifact artifact = ResolveArtifactType(model_path, options);
        ValidateBackendArtifactCompatibility(options.backend, artifact);
        GetLogger().debug(
            "Loading artifact=", model_path.string(), ", detected_type=", ToString(artifact),
            ", requested_backend=", ToString(options.backend), ", device_id=", options.device_id,
            ", allow_fallback=", options.allow_fallback);

        switch (options.backend)
        {
        case EInferenceBackend::auto_backend:
            if (artifact == EModelArtifact::onnx)
            {
                backend_.emplace<ptafdeploy::inference::onnxruntime::CInferenceManager_ORT>();
            }
            else
            {
                backend_
                    .emplace<ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine>();
            }
            break;
        case EInferenceBackend::onnxruntime:
            backend_.emplace<ptafdeploy::inference::onnxruntime::CInferenceManager_ORT>();
            break;
        case EInferenceBackend::tensorrt_engine:
            backend_.emplace<ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine>();
            break;
        }

        std::visit(
            [&](auto& backend)
            {
                using TBackend = std::decay_t<decltype(backend)>;
                if constexpr (!std::is_same_v<TBackend, std::monostate>)
                {
                    backend.LoadModel(model_path, options);
                }
            },
            backend_);
        const SModelMetadata& metadata = GetModelMetadata();
        GetLogger().info("Loaded inference artifact with ", metadata.inputs.size(), " input(s), ",
                         metadata.outputs.size(), " output(s); ", metadata.backend_detail);
    }

    void CInferenceManager::LoadModel(const std::string& model_path)
    {
        LoadModel(fs::path{model_path}, SInferenceOptions{});
    }

    void CInferenceManager::LoadModelWithRuntimeConfig(const std::string& model_path,
                                                       const SRuntimeConfig& runtime_config)
    {
        LoadModel(fs::path{model_path}, MakeInferenceOptions(runtime_config));
    }

    void CInferenceManager::LoadModelWithThreadCounts(const std::string& model_path,
                                                      const int intra_op_num_threads,
                                                      const int inter_op_num_threads)
    {
        SInferenceOptions options;
        options.intra_op_num_threads = intra_op_num_threads;
        options.inter_op_num_threads = inter_op_num_threads;
        LoadModel(fs::path{model_path}, options);
    }

    const SModelMetadata& CInferenceManager::GetModelMetadata() const
    {
        return std::visit(
            [](const auto& backend) -> const SModelMetadata&
            {
                using TBackend = std::decay_t<decltype(backend)>;
                if constexpr (std::is_same_v<TBackend, std::monostate>)
                {
                    throw std::runtime_error("No inference backend has been loaded.");
                }
                else
                {
                    return backend.GetModelMetadata();
                }
            },
            backend_);
    }

    std::string CInferenceManager::GetBackendDetail() const
    {
        return GetModelMetadata().backend_detail;
    }

    size_t CInferenceManager::GetNumInputs() const
    {
        return GetModelMetadata().inputs.size();
    }

    size_t CInferenceManager::GetNumOutputs() const
    {
        return GetModelMetadata().outputs.size();
    }

    STensorInfo CInferenceManager::GetInputInfo(const size_t index) const
    {
        return MakeTensorInfo(GetTensorDescriptorAt(GetModelMetadata().inputs, index, "Input"));
    }

    STensorInfo CInferenceManager::GetOutputInfo(const size_t index) const
    {
        return MakeTensorInfo(GetTensorDescriptorAt(GetModelMetadata().outputs, index, "Output"));
    }

    std::vector<STensorInfo> CInferenceManager::GetInputInfos() const
    {
        return MakeTensorInfos(GetModelMetadata().inputs);
    }

    std::vector<STensorInfo> CInferenceManager::GetOutputInfos() const
    {
        return MakeTensorInfos(GetModelMetadata().outputs);
    }

    std::vector<STensorBuffer>
    CInferenceManager::Infer(const std::vector<STensorView>& inputs) const
    {
        GetLogger().trace("Dispatching inference with ", inputs.size(), " input tensor(s).");
        return std::visit(
            [&](const auto& backend) -> std::vector<STensorBuffer>
            {
                using TBackend = std::decay_t<decltype(backend)>;
                if constexpr (std::is_same_v<TBackend, std::monostate>)
                {
                    throw std::runtime_error("No inference backend has been loaded.");
                }
                else
                {
                    return backend.Infer(inputs);
                }
            },
            backend_);
    }

    std::vector<SFloatTensor>
    CInferenceManager::InferFloatTensors(const std::vector<SFloatTensor>& inputs) const
    {
        std::vector<STensorView> input_views;
        input_views.reserve(inputs.size());
        for (const SFloatTensor& input : inputs)
        {
            input_views.push_back(MakeFloatTensorView(input));
        }

        const std::vector<STensorBuffer> output_buffers = Infer(input_views);
        std::vector<SFloatTensor> outputs;
        outputs.reserve(output_buffers.size());
        for (const STensorBuffer& buffer : output_buffers)
        {
            outputs.push_back(MakeFloatTensor(buffer));
        }

        return outputs;
    }

    SFloatTensor CInferenceManager::InferSingleFloatTensor(const SFloatTensor& input) const
    {
        std::vector<SFloatTensor> outputs = InferFloatTensors({input});
        if (outputs.size() != 1U)
        {
            throw std::runtime_error("Expected exactly one model output tensor.");
        }

        return std::move(outputs.front());
    }

    std::vector<float>
    CInferenceManager::InferSingleFloatInput(const std::vector<float>& values,
                                             const std::vector<int64_t>& shape) const
    {
        return InferSingleFloatTensor(SFloatTensor{"", shape, values}).values;
    }

    EModelArtifact CInferenceManager::DetectArtifactType(const fs::path& model_path)
    {
        return DetectArtifactTypeFromPath(model_path);
    }
} // namespace ptafdeploy::inference
