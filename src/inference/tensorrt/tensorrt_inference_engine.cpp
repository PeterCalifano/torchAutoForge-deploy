/**
 * @file tensorrt_inference_engine.cpp
 * @brief Optional serialized TensorRT engine execution backend.
 */

#include "tensorrt_inference_engine.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utils/logging/CLogger.h>

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#endif

namespace ptafdeploy::inference::tensorrt
{
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    namespace
    {
        class CTensorRtLogger final : public nvinfer1::ILogger
        {
          public:
            void log(const Severity severity, const char* message) noexcept override
            {
                if (severity <= Severity::kERROR && message != nullptr)
                {
                    last_error_ = message;
                }
            }

            [[nodiscard]] const std::string& LastError() const noexcept
            {
                return last_error_;
            }

          private:
            std::string last_error_{};
        };

        class CCudaStream final
        {
          public:
            CCudaStream()
            {
                Check(cudaStreamCreate(&stream_), "create CUDA stream");
            }

            ~CCudaStream()
            {
                if (stream_ != nullptr)
                {
                    cudaStreamDestroy(stream_);
                }
            }

            CCudaStream(const CCudaStream&) = delete;
            CCudaStream& operator=(const CCudaStream&) = delete;

            [[nodiscard]] cudaStream_t Get() const noexcept
            {
                return stream_;
            }

            static void Check(const cudaError_t error, const std::string& operation)
            {
                if (error != cudaSuccess)
                {
                    throw std::runtime_error("CUDA failed to " + operation + ": " +
                                             cudaGetErrorString(error));
                }
            }

          private:
            cudaStream_t stream_{nullptr};
        };

        class CDeviceBuffer final
        {
          public:
            ~CDeviceBuffer()
            {
                if (data_ != nullptr)
                {
                    cudaFree(data_);
                }
            }

            CDeviceBuffer() = default;
            CDeviceBuffer(const CDeviceBuffer&) = delete;
            CDeviceBuffer& operator=(const CDeviceBuffer&) = delete;

            CDeviceBuffer(CDeviceBuffer&& other) noexcept
                : data_(other.data_), capacity_bytes_(other.capacity_bytes_)
            {
                other.data_ = nullptr;
                other.capacity_bytes_ = 0;
            }

            CDeviceBuffer& operator=(CDeviceBuffer&& other) noexcept
            {
                if (this != &other)
                {
                    if (data_ != nullptr)
                    {
                        cudaFree(data_);
                    }
                    data_ = other.data_;
                    capacity_bytes_ = other.capacity_bytes_;
                    other.data_ = nullptr;
                    other.capacity_bytes_ = 0;
                }
                return *this;
            }

            void EnsureCapacity(const size_t bytes)
            {
                if (bytes <= capacity_bytes_)
                {
                    return;
                }

                if (data_ != nullptr)
                {
                    CCudaStream::Check(cudaFree(data_), "free TensorRT device buffer");
                    data_ = nullptr;
                    capacity_bytes_ = 0;
                }

                if (bytes > 0U)
                {
                    CCudaStream::Check(cudaMalloc(&data_, bytes),
                                       "allocate TensorRT device buffer");
                    capacity_bytes_ = bytes;
                }
            }

            [[nodiscard]] void* Data() const noexcept
            {
                return data_;
            }

          private:
            void* data_{nullptr};
            size_t capacity_bytes_{0};
        };

        [[nodiscard]] std::vector<char> ReadBinaryFile(const fs::path& path)
        {
            if (!fs::exists(path) || !fs::is_regular_file(path))
            {
                throw std::invalid_argument("Could not open TensorRT engine file: " +
                                            path.string());
            }

            const std::string extension = path.extension().string();
            if (extension != ".engine" && extension != ".plan")
            {
                throw std::invalid_argument(
                    "TensorRT standalone backend expects a .engine or .plan artifact: " +
                    path.string());
            }

            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            if (!stream)
            {
                throw std::invalid_argument("Could not open TensorRT engine file: " +
                                            path.string());
            }

            const std::streamsize file_size = stream.tellg();
            if (file_size <= 0)
            {
                throw std::invalid_argument("TensorRT engine file is empty: " + path.string());
            }

            std::vector<char> bytes(static_cast<size_t>(file_size));
            stream.seekg(0, std::ios::beg);
            if (!stream.read(bytes.data(), file_size))
            {
                throw std::runtime_error("Failed to read TensorRT engine file: " + path.string());
            }

            return bytes;
        }

        [[nodiscard]] ptafdeploy::inference::ETensorElementType
        ConvertTensorRtType(const nvinfer1::DataType dtype)
        {
            switch (dtype)
            {
            case nvinfer1::DataType::kFLOAT:
                return ptafdeploy::inference::ETensorElementType::float32;
            case nvinfer1::DataType::kHALF:
                return ptafdeploy::inference::ETensorElementType::float16;
            case nvinfer1::DataType::kINT8:
                return ptafdeploy::inference::ETensorElementType::int8;
            case nvinfer1::DataType::kINT32:
                return ptafdeploy::inference::ETensorElementType::int32;
            case nvinfer1::DataType::kBOOL:
                return ptafdeploy::inference::ETensorElementType::boolean;
            case nvinfer1::DataType::kUINT8:
                return ptafdeploy::inference::ETensorElementType::uint8;
            case nvinfer1::DataType::kBF16:
                return ptafdeploy::inference::ETensorElementType::bfloat16;
            case nvinfer1::DataType::kINT64:
                return ptafdeploy::inference::ETensorElementType::int64;
            default:
                throw std::runtime_error("Unsupported TensorRT tensor element type.");
            }
        }

        [[nodiscard]] std::vector<int64_t> ConvertDims(const nvinfer1::Dims& dims)
        {
            if (dims.nbDims < 0)
            {
                throw std::runtime_error("TensorRT returned invalid tensor dimensions.");
            }

            std::vector<int64_t> shape;
            shape.reserve(static_cast<size_t>(dims.nbDims));
            for (int32_t i = 0; i < dims.nbDims; ++i)
            {
                shape.push_back(static_cast<int64_t>(dims.d[i]));
            }
            return shape;
        }

        [[nodiscard]] nvinfer1::Dims MakeTensorRtDims(const std::vector<int64_t>& shape,
                                                      const std::string& tensor_name)
        {
            if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
            {
                throw std::invalid_argument(
                    "Tensor rank exceeds TensorRT Dims::MAX_DIMS for tensor: " + tensor_name);
            }

            nvinfer1::Dims dims{};
            dims.nbDims = static_cast<int32_t>(shape.size());
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] < 0 ||
                    shape[i] > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
                {
                    throw std::invalid_argument(
                        "TensorRT runtime input shape must be concrete int32 dimensions: " +
                        tensor_name);
                }
                dims.d[i] = static_cast<int32_t>(shape[i]);
            }
            return dims;
        }

        [[nodiscard]] bool RequiresHostAddress(const nvinfer1::ICudaEngine& engine,
                                               const std::string& tensor_name)
        {
            return engine.getTensorLocation(tensor_name.c_str()) == nvinfer1::TensorLocation::kHOST;
        }

        [[nodiscard]] ptafdeploy::inference::STensorDescriptor
        ExtractTensorDescriptor(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name)
        {
            ptafdeploy::inference::STensorDescriptor descriptor;
            descriptor.name = tensor_name;
            descriptor.dtype = ConvertTensorRtType(engine.getTensorDataType(tensor_name.c_str()));
            descriptor.shape = ConvertDims(engine.getTensorShape(tensor_name.c_str()));
            descriptor.location = ptafdeploy::inference::EMemoryLocation::host;
            return descriptor;
        }

        [[nodiscard]] std::string
        MakeBackendDetail(const ptafdeploy::inference::SInferenceOptions& options)
        {
            std::ostringstream stream;
            stream << "backend=tensorrt_engine" << ";requested_targets="
                   << ptafdeploy::inference::JoinExecutionTargets(options.execution_target_priority)
                   << ";device_id=" << options.device_id
                   << ";optimization_profile=" << options.tensorrt_optimization_profile_index
                   << ";io=host_staged" << ";tensorrt_version=" << getInferLibVersion();
            return stream.str();
        }

        void ValidateExecutionTargets(const ptafdeploy::inference::SInferenceOptions& options)
        {
            if (options.execution_target_priority.empty())
            {
                return;
            }

            const auto& targets = options.execution_target_priority;
            const bool has_gpu_target =
                std::any_of(targets.begin(), targets.end(),
                            [](const ptafdeploy::inference::EExecutionTarget target)
                            {
                                return target == ptafdeploy::inference::EExecutionTarget::cuda ||
                                       target == ptafdeploy::inference::EExecutionTarget::tensorrt;
                            });
            if (!has_gpu_target)
            {
                throw std::invalid_argument(
                    "TensorRT standalone backend requires cuda or tensorrt execution target.");
            }
        }

        void ApplyTensorRtRuntimeOptions(const ptafdeploy::inference::SInferenceOptions& options,
                                         const nvinfer1::ICudaEngine& engine,
                                         nvinfer1::IExecutionContext& context)
        {
            const int32_t num_profiles = engine.getNbOptimizationProfiles();
            if (options.tensorrt_optimization_profile_index < 0 ||
                options.tensorrt_optimization_profile_index >= num_profiles)
            {
                throw std::invalid_argument("TensorRT optimization profile index is out of range.");
            }

            if (options.tensorrt_optimization_profile_index != 0)
            {
                CCudaStream profile_stream;
                if (!context.setOptimizationProfileAsync(
                        options.tensorrt_optimization_profile_index, profile_stream.Get()))
                {
                    throw std::runtime_error("TensorRT rejected optimization profile index.");
                }
                CCudaStream::Check(cudaStreamSynchronize(profile_stream.Get()),
                                   "synchronize TensorRT optimization profile selection");
            }
        }
    } // namespace

    struct STensorRtState
    {
        CTensorRtLogger logger{};
        std::unique_ptr<nvinfer1::IRuntime> runtime{};
        std::unique_ptr<nvinfer1::ICudaEngine> engine{};
        std::unique_ptr<nvinfer1::IExecutionContext> context{};
        mutable std::vector<CDeviceBuffer> input_device_buffers{};
        mutable std::vector<CDeviceBuffer> output_device_buffers{};
    };
#else
    struct STensorRtState
    {
    };
#endif

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    namespace
    {
        [[nodiscard]] ptafdeploy::logging::CLogger& GetLogger()
        {
            static ptafdeploy::logging::CLogger logger(
                "tensorrt_engine", ptafdeploy::logging::ELogLevel::Warning,
                ptafdeploy::logging::ELogColorMode::Disabled, std::clog, std::clog);
            static const bool environment_applied = logger.setLevelFromEnvironment();
            static_cast<void>(environment_applied);
            return logger;
        }
    } // namespace
#endif

    CInferenceManager_TensorRT_Engine::CInferenceManager_TensorRT_Engine() = default;

    CInferenceManager_TensorRT_Engine::~CInferenceManager_TensorRT_Engine() = default;

    CInferenceManager_TensorRT_Engine::CInferenceManager_TensorRT_Engine(
        const fs::path& model_path, const ptafdeploy::inference::SInferenceOptions& options)
    {
        LoadModel(model_path, options);
    }

    void CInferenceManager_TensorRT_Engine::LoadModel(
        const fs::path& model_path, const ptafdeploy::inference::SInferenceOptions& options)
    {
        model_path_ = model_path;
        options_ = options;
        metadata_ = {};
        metadata_.backend = ptafdeploy::inference::EInferenceBackend::tensorrt_engine;

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
        if (options_.device_id < 0)
        {
            throw std::invalid_argument("TensorRT device_id must be non-negative.");
        }
        ValidateExecutionTargets(options_);

        CCudaStream::Check(cudaSetDevice(options_.device_id), "set CUDA device for TensorRT");

        std::vector<char> engine_bytes = ReadBinaryFile(model_path_);
        state_ = std::make_unique<STensorRtState>();
        if (!initLibNvInferPlugins(&state_->logger, ""))
        {
            throw std::runtime_error("Failed to initialize TensorRT plugins.");
        }

        state_->runtime.reset(nvinfer1::createInferRuntime(state_->logger));
        if (state_->runtime == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        state_->engine.reset(
            state_->runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
        if (state_->engine == nullptr)
        {
            const std::string detail =
                state_->logger.LastError().empty() ? "" : ": " + state_->logger.LastError();
            throw std::runtime_error("Failed to deserialize TensorRT engine" + detail);
        }

        state_->context.reset(state_->engine->createExecutionContext());
        if (state_->context == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }
        ApplyTensorRtRuntimeOptions(options_, *state_->engine, *state_->context);

        const int32_t num_io_tensors = state_->engine->getNbIOTensors();
        for (int32_t i = 0; i < num_io_tensors; ++i)
        {
            const char* tensor_name_ptr = state_->engine->getIOTensorName(i);
            if (tensor_name_ptr == nullptr)
            {
                throw std::runtime_error("TensorRT engine returned a null IO tensor name.");
            }

            const std::string tensor_name{tensor_name_ptr};
            const nvinfer1::TensorIOMode io_mode =
                state_->engine->getTensorIOMode(tensor_name.c_str());
            if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            {
                metadata_.inputs.push_back(ExtractTensorDescriptor(*state_->engine, tensor_name));
            }
            else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                metadata_.outputs.push_back(ExtractTensorDescriptor(*state_->engine, tensor_name));
            }
        }

        if (metadata_.inputs.empty() || metadata_.outputs.empty())
        {
            throw std::runtime_error(
                "TensorRT engine must expose at least one input and one output tensor.");
        }

        state_->input_device_buffers.resize(metadata_.inputs.size());
        state_->output_device_buffers.resize(metadata_.outputs.size());
        metadata_.backend_detail = MakeBackendDetail(options_);
        GetLogger().info("Loaded TensorRT engine: ", model_path_.string());
        GetLogger().debug(metadata_.backend_detail, ";inputs=", metadata_.inputs.size(),
                          ";outputs=", metadata_.outputs.size());
#else
        metadata_.backend_detail =
            "TensorRT standalone backend was not built. Reconfigure with ENABLE_TENSORRT=ON.";
        ThrowNotImplemented();
#endif
    }

    const ptafdeploy::inference::SModelMetadata&
    CInferenceManager_TensorRT_Engine::GetModelMetadata() const noexcept
    {
        return metadata_;
    }

    std::vector<ptafdeploy::inference::STensorBuffer> CInferenceManager_TensorRT_Engine::Infer(
        const std::vector<ptafdeploy::inference::STensorView>& inputs) const
    {
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
        std::lock_guard<std::mutex> lock(inference_mutex_);
        if (state_ == nullptr || state_->engine == nullptr || state_->context == nullptr)
        {
            throw std::runtime_error("TensorRT engine has not been initialized.");
        }

        GetLogger().trace("Running TensorRT inference with ", inputs.size(), " input tensor(s).");

        CCudaStream::Check(cudaSetDevice(options_.device_id),
                           "set CUDA device for TensorRT inference");
        CCudaStream stream;

        const std::vector<const ptafdeploy::inference::STensorView*> ordered_inputs =
            ptafdeploy::inference::OrderInputViews(metadata_.inputs, inputs);
        for (size_t i = 0; i < metadata_.inputs.size(); ++i)
        {
            const ptafdeploy::inference::STensorDescriptor& expected_descriptor =
                metadata_.inputs[i];
            const ptafdeploy::inference::STensorView& input_view = *ordered_inputs[i];
            ptafdeploy::inference::ValidateHostTensorView(expected_descriptor, input_view,
                                                          "TensorRT");

            const nvinfer1::Dims input_dims =
                MakeTensorRtDims(input_view.descriptor.shape, expected_descriptor.name);
            if (!state_->context->setInputShape(expected_descriptor.name.c_str(), input_dims))
            {
                throw std::invalid_argument("TensorRT rejected runtime shape for input tensor: " +
                                            expected_descriptor.name);
            }

            void* tensor_address = const_cast<void*>(input_view.data);
            if (!RequiresHostAddress(*state_->engine, expected_descriptor.name))
            {
                state_->input_device_buffers[i].EnsureCapacity(input_view.bytes);
                if (input_view.bytes > 0U)
                {
                    CCudaStream::Check(cudaMemcpyAsync(state_->input_device_buffers[i].Data(),
                                                       input_view.data, input_view.bytes,
                                                       cudaMemcpyHostToDevice, stream.Get()),
                                       "copy TensorRT input to device");
                }
                tensor_address = state_->input_device_buffers[i].Data();
            }

            if (!state_->context->setTensorAddress(expected_descriptor.name.c_str(),
                                                   tensor_address))
            {
                throw std::runtime_error("TensorRT rejected input tensor address: " +
                                         expected_descriptor.name);
            }
        }

        std::vector<ptafdeploy::inference::STensorBuffer> outputs;
        outputs.reserve(metadata_.outputs.size());
        for (size_t i = 0; i < metadata_.outputs.size(); ++i)
        {
            ptafdeploy::inference::STensorDescriptor descriptor = metadata_.outputs[i];
            descriptor.shape =
                ConvertDims(state_->context->getTensorShape(descriptor.name.c_str()));
            descriptor.location = ptafdeploy::inference::EMemoryLocation::host;
            if (ptafdeploy::inference::HasDynamicShape(descriptor.shape))
            {
                throw std::runtime_error("TensorRT output shape is unresolved for tensor: " +
                                         descriptor.name);
            }

            outputs.push_back(ptafdeploy::inference::MakeOwnedTensorBuffer(descriptor));
            void* tensor_address = outputs.back().data();
            if (!RequiresHostAddress(*state_->engine, descriptor.name))
            {
                state_->output_device_buffers[i].EnsureCapacity(outputs.back().bytes());
                tensor_address = state_->output_device_buffers[i].Data();
            }

            if (!state_->context->setTensorAddress(descriptor.name.c_str(), tensor_address))
            {
                throw std::runtime_error("TensorRT rejected output tensor address: " +
                                         descriptor.name);
            }
        }

        if (!state_->context->enqueueV3(stream.Get()))
        {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            if (!RequiresHostAddress(*state_->engine, outputs[i].descriptor.name) &&
                outputs[i].bytes() > 0U)
            {
                CCudaStream::Check(
                    cudaMemcpyAsync(outputs[i].data(), state_->output_device_buffers[i].Data(),
                                    outputs[i].bytes(), cudaMemcpyDeviceToHost, stream.Get()),
                    "copy TensorRT output to host");
            }
        }

        CCudaStream::Check(cudaStreamSynchronize(stream.Get()), "synchronize TensorRT stream");
        return outputs;
#else
        (void)inputs;
        ThrowNotImplemented();
#endif
    }

    void CInferenceManager_TensorRT_Engine::ThrowNotImplemented()
    {
        throw std::runtime_error(
            "TensorRT standalone backend was not built. Reconfigure with ENABLE_TENSORRT=ON.");
    }
} // namespace ptafdeploy::inference::tensorrt
