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
                    try
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        last_error_ = message;
                    }
                    catch (...)
                    {
                        // TensorRT requires a nonthrowing logging callback, even on allocation
                        // failure.
                    }
                }
            }

            [[nodiscard]] std::string LastError() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return last_error_;
            }

          private:
            mutable std::mutex mutex_;
            std::string last_error_{};
        };

        // Plugin registration retains a logger independently of individual engine states.
        void InitializeTensorRtPlugins()
        {
            // The process-global plugin registry can outlive static backend instances.
            // Keep this single logger alive through process teardown.
            static auto* plugin_logger = new CTensorRtLogger;
            static const bool initialized = initLibNvInferPlugins(plugin_logger, "");
            if (!initialized)
                throw std::runtime_error("Failed to initialize TensorRT plugins.");
        }

        /**
         * @brief Select a device for a scope and restore the caller's device on exit.
         * @param device_id CUDA device owning the resources used in this scope.
         * @throws std::runtime_error If reading or selecting the current device fails.
         */
        class CCudaDeviceScope final
        {
          public:
            explicit CCudaDeviceScope(int device_id)
            {
                if (cudaGetDevice(&previous_device_) != cudaSuccess ||
                    cudaSetDevice(device_id) != cudaSuccess)
                    throw std::runtime_error("Cannot select CUDA device for TensorRT");
            }
            ~CCudaDeviceScope()
            {
                cudaSetDevice(previous_device_);
            }
            CCudaDeviceScope(const CCudaDeviceScope&) = delete;
            CCudaDeviceScope& operator=(const CCudaDeviceScope&) = delete;

          private:
            int previous_device_{};
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
        MakeBackendDetail(const ptafdeploy::inference::SInferenceOptions& options,
                          size_t selected_target_index)
        {
            std::ostringstream stream;
            stream << "backend=tensorrt_engine" << ";requested_targets="
                   << ptafdeploy::inference::JoinExecutionTargets(options.execution_target_priority)
                   << ";selected_target="
                   << (options.execution_target_priority.empty()
                           ? "cuda"
                           : ptafdeploy::inference::ToString(
                                 options.execution_target_priority[selected_target_index]))
                   << ";skipped_target_count=" << selected_target_index
                   << ";device_id=" << options.device_id
                   << ";optimization_profile=" << options.tensorrt_optimization_profile_index
                   << ";io=host_staged" << ";tensorrt_version=" << getInferLibVersion();
            return stream.str();
        }

        [[nodiscard]] size_t
        SelectExecutionTarget(const ptafdeploy::inference::SInferenceOptions& options)
        {
            const auto& targets = options.execution_target_priority;
            if (targets.empty())
                return 0; // The standalone backend defaults to CUDA.

            for (size_t index = 0; index < targets.size(); ++index)
            {
                if (targets[index] == EExecutionTarget::cuda ||
                    targets[index] == EExecutionTarget::tensorrt)
                    return index;
                if (!options.allow_fallback)
                    break;
            }
            throw std::invalid_argument(
                "TensorRT requires cuda or tensorrt as the first target when fallback is disabled; "
                "CPU-only execution is unsupported.");
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
        explicit STensorRtState(int device) : device_id(device)
        {
        }

        ~STensorRtState()
        {
            // Destroy device allocations and TensorRT objects on their owning device.
            int previous_device = device_id;
            cudaGetDevice(&previous_device);
            cudaSetDevice(device_id);
            output_device_buffers.clear();
            input_device_buffers.clear();
            context.reset();
            engine.reset();
            runtime.reset();
            cudaSetDevice(previous_device);
        }

        int device_id{};
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
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto replacement_path = model_path;
        auto replacement_options = options;
        SModelMetadata replacement_metadata;
        replacement_metadata.backend = EInferenceBackend::tensorrt_engine;
        if (replacement_options.device_id < 0)
        {
            throw std::invalid_argument("TensorRT device_id must be non-negative.");
        }
        const auto selected_target = SelectExecutionTarget(replacement_options);

        CCudaDeviceScope device_scope(replacement_options.device_id);

        std::vector<char> engine_bytes = ReadBinaryFile(replacement_path);
        InitializeTensorRtPlugins();
        auto replacement_state = std::make_unique<STensorRtState>(replacement_options.device_id);

        replacement_state->runtime.reset(nvinfer1::createInferRuntime(replacement_state->logger));
        if (replacement_state->runtime == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        replacement_state->engine.reset(replacement_state->runtime->deserializeCudaEngine(
            engine_bytes.data(), engine_bytes.size()));
        if (replacement_state->engine == nullptr)
        {
            const auto last_error = replacement_state->logger.LastError();
            const std::string detail = last_error.empty() ? "" : ": " + last_error;
            throw std::runtime_error("Failed to deserialize TensorRT engine" + detail);
        }

        replacement_state->context.reset(replacement_state->engine->createExecutionContext());
        if (replacement_state->context == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }
        ApplyTensorRtRuntimeOptions(replacement_options, *replacement_state->engine,
                                    *replacement_state->context);

        const int32_t num_io_tensors = replacement_state->engine->getNbIOTensors();
        for (int32_t i = 0; i < num_io_tensors; ++i)
        {
            const char* tensor_name_ptr = replacement_state->engine->getIOTensorName(i);
            if (tensor_name_ptr == nullptr)
            {
                throw std::runtime_error("TensorRT engine returned a null IO tensor name.");
            }

            const std::string tensor_name{tensor_name_ptr};
            const nvinfer1::TensorIOMode io_mode =
                replacement_state->engine->getTensorIOMode(tensor_name.c_str());
            if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            {
                replacement_metadata.inputs.push_back(
                    ExtractTensorDescriptor(*replacement_state->engine, tensor_name));
            }
            else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                replacement_metadata.outputs.push_back(
                    ExtractTensorDescriptor(*replacement_state->engine, tensor_name));
            }
        }

        if (replacement_metadata.inputs.empty() || replacement_metadata.outputs.empty())
        {
            throw std::runtime_error(
                "TensorRT engine must expose at least one input and one output tensor.");
        }

        replacement_state->input_device_buffers.resize(replacement_metadata.inputs.size());
        replacement_state->output_device_buffers.resize(replacement_metadata.outputs.size());
        replacement_metadata.backend_detail =
            MakeBackendDetail(replacement_options, selected_target);
        // Finish potentially throwing diagnostics before exchanging the loaded state.
        GetLogger().info("Loaded TensorRT engine: ", replacement_path.string());
        GetLogger().debug(replacement_metadata.backend_detail,
                          ";inputs=", replacement_metadata.inputs.size(),
                          ";outputs=", replacement_metadata.outputs.size());

        // Commit only complete state. The displaced state retains its device ownership.
        using std::swap;
        swap(model_path_, replacement_path);
        swap(options_, replacement_options);
        swap(metadata_, replacement_metadata);
        swap(state_, replacement_state);
#else
        (void)model_path;
        (void)options;
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

        CCudaDeviceScope device_scope(options_.device_id);
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
