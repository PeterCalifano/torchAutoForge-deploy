/**
 * @file inference_common.h
 * @brief Backend-neutral tensor, model, and runtime value contracts.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Runtime backend family requested by user-facing config.
     *
     * `auto_backend` resolves from artifact type. Concrete values select the
     * backend explicitly and fail early when incompatible with the artifact.
     */
    enum class EInferenceBackend
    {
        auto_backend,
        onnxruntime,
        tensorrt_engine
    };

    /**
     * @brief Serialized model artifact type.
     *
     * Keep this separate from backend: an ONNX artifact may still run through
     * different ORT execution providers, while a TensorRT engine is a standalone
     * serialized plan handled by the TensorRT backend.
     */
    enum class EModelArtifact
    {
        auto_artifact,
        onnx,
        tensorrt_engine
    };

    /**
     * @brief Execution target priority inside a backend.
     *
     * ORT maps these to execution providers. TensorRT engine uses CUDA device
     * execution directly.
     */
    enum class EExecutionTarget
    {
        cpu,
        cuda,
        tensorrt
    };

    /**
     * @brief High-level model role for facade/adapters.
     *
     * Roles describe model semantics above raw tensor inference. They must not
     * leak backend-specific handles into wrappers.
     */
    enum class EModelRole
    {
        raw_tensor,
        centroiding,
        object_detection,
        feature_matching,
        tracking,
        optical_flow,
        custom
    };

    /** @brief Backend-neutral dense tensor element types with byte-addressable storage. */
    enum class ETensorElementType
    {
        boolean,
        uint8,
        int8,
        uint16,
        int16,
        uint32,
        int32,
        uint64,
        int64,
        float16,
        bfloat16,
        float32,
        float64
    };

    /**
     * @brief Memory-location tag carried by tensor descriptors.
     *
     * Current inference entry points accept `host`. `device_cuda` is retained
     * as an explicit rejection/extension tag and is not an execution claim.
     */
    enum class EMemoryLocation
    {
        host,
        device_cuda
    };

    /**
     * @brief Backend-neutral tensor descriptor.
     *
     * Shapes use `-1` for dynamic dimensions reported by backends. Runtime
     * input views must provide concrete shapes before inference.
     */
    struct STensorDescriptor
    {
        std::string name{};
        ETensorElementType dtype{ETensorElementType::float32};
        std::vector<int64_t> shape{};
        EMemoryLocation location{EMemoryLocation::host};
    };

    /**
     * @brief Wrapper-safe tensor metadata.
     *
     * Strings are intentional here: MATLAB/Python bindings should not need to
     * understand C++ enum values to inspect loaded model contracts.
     */
    struct STensorInfo
    {
        std::string name{};
        std::string dtype{};
        std::vector<int64_t> shape{};
        std::string location{};
    };

    /**
     * @brief Owned float32 tensor used by wrapper-safe inference helpers.
     *
     * This is intentionally simple and value-based for Python/MATLAB. Lower
     * layers use `STensorView` to avoid input copies.
     */
    struct SFloatTensor
    {
        std::string name{};
        std::vector<int64_t> shape{};
        std::vector<float> values{};

        SFloatTensor() = default;
        SFloatTensor(std::string tensor_name,
                     std::vector<int64_t> tensor_shape,
                     std::vector<float> tensor_values)
            : name(std::move(tensor_name)),
              shape(std::move(tensor_shape)),
              values(std::move(tensor_values))
        {
        }
    };

    /**
     * @brief Non-owning dense tensor view.
     *
     * The caller owns memory. Backends validate name, dtype, shape, byte count,
     * and memory location before execution.
     */
    struct STensorView
    {
        STensorDescriptor descriptor{};
        const void *data{nullptr};
        size_t bytes{0};
    };

    /**
     * @brief Owned dense tensor output buffer.
     *
     * Storage is byte-oriented so any supported dtype can be returned without
     * templating the backend interface.
     */
    struct STensorBuffer
    {
        STensorDescriptor descriptor{};
        std::vector<std::byte> storage{};

        [[nodiscard]] const void *data() const noexcept
        {
            return storage.empty() ? nullptr : storage.data();
        }

        [[nodiscard]] void *data() noexcept
        {
            return storage.empty() ? nullptr : storage.data();
        }

        [[nodiscard]] size_t bytes() const noexcept
        {
            return storage.size();
        }

        [[nodiscard]] STensorView AsView() const noexcept
        {
            return STensorView{descriptor, data(), bytes()};
        }
    };

    /** @brief Full backend metadata for a loaded model. */
    struct SModelMetadata
    {
        std::vector<STensorDescriptor> inputs{};
        std::vector<STensorDescriptor> outputs{};
        EInferenceBackend backend{EInferenceBackend::auto_backend};
        std::string backend_detail{};
    };

    /**
     * @brief Internal backend load options.
     *
     * Use `SRuntimeConfig` at public config/wrapper boundaries. This type is the
     * backend-facing equivalent used by `CInferenceManager`.
     */
    struct SInferenceOptions
    {
        EInferenceBackend backend{EInferenceBackend::auto_backend};
        EModelArtifact artifact{EModelArtifact::auto_artifact};
        bool allow_fallback{true};
        std::vector<EExecutionTarget> execution_target_priority{};
        int device_id{0};
        int intra_op_num_threads{1};
        int inter_op_num_threads{1};
        bool enable_profiling{false};
        std::string log_id{"autoforge_deploy"};
        int tensorrt_optimization_profile_index{0};
    };

    /**
     * @brief Public runtime configuration for manifests, wrappers, and facade.
     *
     * Setters validate values that can otherwise crash or misconfigure backend
     * setup, especially device id and thread counts.
     */
    struct SRuntimeConfig
    {
        EInferenceBackend backend{EInferenceBackend::auto_backend};
        EModelArtifact artifact{EModelArtifact::auto_artifact};
        bool allow_fallback{true};
        std::vector<EExecutionTarget> execution_target_priority{};
        int device_id{0};
        int intra_op_num_threads{1};
        int inter_op_num_threads{1};
        bool enable_profiling{false};
        std::string log_id{"autoforge_deploy"};
        int tensorrt_optimization_profile_index{0};

        [[nodiscard]] EInferenceBackend GetBackend() const noexcept
        {
            return backend;
        }

        void SetBackend(const EInferenceBackend value) noexcept
        {
            backend = value;
        }

        [[nodiscard]] EModelArtifact GetArtifact() const noexcept
        {
            return artifact;
        }

        void SetArtifact(const EModelArtifact value) noexcept
        {
            artifact = value;
        }

        [[nodiscard]] bool GetAllowFallback() const noexcept
        {
            return allow_fallback;
        }

        void SetAllowFallback(const bool value) noexcept
        {
            allow_fallback = value;
        }

        [[nodiscard]] int GetDeviceId() const noexcept
        {
            return device_id;
        }

        /** @brief Set the non-negative backend device index. */
        void SetDeviceId(const int value)
        {
            if (value < 0)
            {
                throw std::invalid_argument("Runtime device_id must be non-negative.");
            }
            device_id = value;
        }

        [[nodiscard]] int GetIntraOpNumThreads() const noexcept
        {
            return intra_op_num_threads;
        }

        [[nodiscard]] int GetInterOpNumThreads() const noexcept
        {
            return inter_op_num_threads;
        }

        /** @brief Set non-negative intra/inter-operation thread counts. */
        void SetThreadCounts(const int intra_op_threads,
                             const int inter_op_threads)
        {
            if (intra_op_threads < 0 || inter_op_threads < 0)
            {
                throw std::invalid_argument("Runtime thread counts must be non-negative.");
            }
            intra_op_num_threads = intra_op_threads;
            inter_op_num_threads = inter_op_threads;
        }

        [[nodiscard]] bool GetEnableProfiling() const noexcept
        {
            return enable_profiling;
        }

        void SetEnableProfiling(const bool value) noexcept
        {
            enable_profiling = value;
        }

        [[nodiscard]] std::string GetLogId() const
        {
            return log_id;
        }

        void SetLogId(std::string value)
        {
            log_id = std::move(value);
        }

        [[nodiscard]] int GetTensorRtOptimizationProfileIndex() const noexcept
        {
            return tensorrt_optimization_profile_index;
        }

        /** @brief Select a non-negative TensorRT optimization profile. */
        void SetTensorRtOptimizationProfileIndex(const int value)
        {
            if (value < 0)
            {
                throw std::invalid_argument("TensorRT optimization profile index must be non-negative.");
            }
            tensorrt_optimization_profile_index = value;
        }

        /** @brief Clear explicit target priority so the backend applies its default. */
        void ClearExecutionTargetPriority()
        {
            execution_target_priority.clear();
        }

        /** @brief Append one execution target to the requested priority order. */
        void AddExecutionTarget(const EExecutionTarget target)
        {
            execution_target_priority.push_back(target);
        }

        /** @brief Request CPU execution without fallback. */
        void UseCpuOnly()
        {
            execution_target_priority = {EExecutionTarget::cpu};
            allow_fallback = false;
        }

        /** @brief Request CUDA followed by CPU fallback. */
        void UseCudaWithCpuFallback()
        {
            execution_target_priority = {EExecutionTarget::cuda, EExecutionTarget::cpu};
            allow_fallback = true;
        }

        /** @brief Request ORT TensorRT, CUDA, then CPU fallback. */
        void UseTensorRtWithCudaFallback()
        {
            execution_target_priority = {EExecutionTarget::tensorrt, EExecutionTarget::cuda, EExecutionTarget::cpu};
            allow_fallback = true;
        }
    };

    /**
     * @brief Convert a backend value to its stable manifest/CLI name.
     * @param backend Backend enum value.
     * @return Stable lowercase backend name.
     * @throws std::runtime_error When `backend` is not a declared value.
     */
    [[nodiscard]] inline std::string ToString(const EInferenceBackend backend)
    {
        switch (backend)
        {
        case EInferenceBackend::auto_backend:
            return "auto";
        case EInferenceBackend::onnxruntime:
            return "onnxruntime";
        case EInferenceBackend::tensorrt_engine:
            return "tensorrt_engine";
        }

        throw std::runtime_error("Unsupported inference backend.");
    }

    /** @brief Convert an artifact value to its stable manifest/CLI name. */
    [[nodiscard]] inline std::string ToString(const EModelArtifact artifact)
    {
        switch (artifact)
        {
        case EModelArtifact::auto_artifact:
            return "auto";
        case EModelArtifact::onnx:
            return "onnx";
        case EModelArtifact::tensorrt_engine:
            return "tensorrt_engine";
        }

        throw std::runtime_error("Unsupported model artifact.");
    }

    /** @brief Convert an execution target to its stable manifest/CLI name. */
    [[nodiscard]] inline std::string ToString(const EExecutionTarget target)
    {
        switch (target)
        {
        case EExecutionTarget::cpu:
            return "cpu";
        case EExecutionTarget::cuda:
            return "cuda";
        case EExecutionTarget::tensorrt:
            return "tensorrt";
        }

        throw std::runtime_error("Unsupported execution target.");
    }

    /** @brief Convert a recognized model role to its stable manifest name. */
    [[nodiscard]] inline std::string ToString(const EModelRole role)
    {
        switch (role)
        {
        case EModelRole::raw_tensor:
            return "raw_tensor";
        case EModelRole::centroiding:
            return "centroiding";
        case EModelRole::object_detection:
            return "object_detection";
        case EModelRole::feature_matching:
            return "feature_matching";
        case EModelRole::tracking:
            return "tracking";
        case EModelRole::optical_flow:
            return "optical_flow";
        case EModelRole::custom:
            return "custom";
        }

        throw std::runtime_error("Unsupported model role.");
    }

    /** @brief Convert a tensor element type to its stable metadata name. */
    [[nodiscard]] inline std::string ToString(const ETensorElementType dtype)
    {
        switch (dtype)
        {
        case ETensorElementType::boolean:
            return "bool";
        case ETensorElementType::uint8:
            return "uint8";
        case ETensorElementType::int8:
            return "int8";
        case ETensorElementType::uint16:
            return "uint16";
        case ETensorElementType::int16:
            return "int16";
        case ETensorElementType::uint32:
            return "uint32";
        case ETensorElementType::int32:
            return "int32";
        case ETensorElementType::uint64:
            return "uint64";
        case ETensorElementType::int64:
            return "int64";
        case ETensorElementType::float16:
            return "float16";
        case ETensorElementType::bfloat16:
            return "bfloat16";
        case ETensorElementType::float32:
            return "float32";
        case ETensorElementType::float64:
            return "float64";
        }

        throw std::runtime_error("Unsupported tensor element type.");
    }

    /** @brief Convert a memory-location tag to its stable metadata name. */
    [[nodiscard]] inline std::string ToString(const EMemoryLocation location)
    {
        switch (location)
        {
        case EMemoryLocation::host:
            return "host";
        case EMemoryLocation::device_cuda:
            return "device_cuda";
        }

        throw std::runtime_error("Unsupported tensor memory location.");
    }

    /**
     * @brief Convert an internal descriptor to wrapper-safe metadata.
     * @param descriptor Backend-neutral tensor descriptor.
     * @return Value-only tensor metadata suitable for wrappers.
     */
    [[nodiscard]] inline STensorInfo MakeTensorInfo(const STensorDescriptor &descriptor)
    {
        return STensorInfo{descriptor.name,
                           ToString(descriptor.dtype),
                           descriptor.shape,
                           ToString(descriptor.location)};
    }

    /**
     * @brief Return the byte width of a supported tensor element type.
     * @param dtype Byte-addressable tensor element type.
     * @return Number of bytes per element.
     * @throws std::runtime_error When `dtype` is not a declared value.
     */
    [[nodiscard]] inline size_t GetTensorElementTypeSize(const ETensorElementType dtype)
    {
        switch (dtype)
        {
        case ETensorElementType::boolean:
        case ETensorElementType::uint8:
        case ETensorElementType::int8:
            return 1;
        case ETensorElementType::uint16:
        case ETensorElementType::int16:
        case ETensorElementType::float16:
        case ETensorElementType::bfloat16:
            return 2;
        case ETensorElementType::uint32:
        case ETensorElementType::int32:
        case ETensorElementType::float32:
            return 4;
        case ETensorElementType::uint64:
        case ETensorElementType::int64:
        case ETensorElementType::float64:
            return 8;
        }

        throw std::runtime_error("Unsupported tensor element type.");
    }

    /**
     * @brief Check whether a backend-reported shape contains dynamic dimensions.
     * @param shape Tensor shape using negative values for dynamic dimensions.
     * @return True when at least one dimension is dynamic.
     */
    [[nodiscard]] inline bool HasDynamicShape(const std::vector<int64_t> &shape) noexcept
    {
        for (const int64_t dim : shape)
        {
            if (dim < 0)
            {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Match runtime shape against metadata shape.
     *
     * Non-negative metadata dimensions are exact; negative dimensions accept
     * any concrete runtime value.
     * @param expected_shape Model metadata shape.
     * @param actual_shape Concrete runtime shape.
     * @return True when ranks and every fixed dimension match.
     */
    [[nodiscard]] inline bool IsShapeCompatible(const std::vector<int64_t> &expected_shape,
                                                const std::vector<int64_t> &actual_shape)
    {
        if (expected_shape.size() != actual_shape.size())
        {
            return false;
        }

        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            if (expected_shape[i] >= 0 && expected_shape[i] != actual_shape[i])
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Multiply two unsigned values with overflow checking.
     * @tparam T Unsigned integral type.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @param error_message Exception message used on overflow.
     * @return Checked product.
     * @throws std::overflow_error When the product exceeds `T`.
     */
    template <std::unsigned_integral T>
    [[nodiscard]] inline T CheckedMultiply(const T lhs,
                                           const T rhs,
                                           const std::string &error_message)
    {
        if (rhs != 0U && lhs > (std::numeric_limits<T>::max() / rhs))
        {
            throw std::overflow_error(error_message);
        }

        return lhs * rhs;
    }

    /**
     * @brief Compute a checked element count for a concrete tensor shape.
     * @param shape Concrete non-negative dimensions.
     * @return Product of the dimensions.
     * @throws std::invalid_argument When a dimension is dynamic.
     * @throws std::overflow_error When the product exceeds `size_t`.
     */
    [[nodiscard]] inline size_t ComputeElementCount(const std::vector<int64_t> &shape)
    {
        size_t num_elements = 1;
        for (const int64_t dim : shape)
        {
            if (dim < 0)
            {
                throw std::invalid_argument("Dynamic dimensions require concrete runtime shapes.");
            }

            const size_t dim_count = static_cast<size_t>(dim);
            num_elements = CheckedMultiply(num_elements, dim_count, "Tensor element count overflows size_t.");
        }

        return num_elements;
    }

    /**
     * @brief Compute the checked byte count for a dense tensor descriptor.
     * @param descriptor Concrete dense tensor descriptor.
     * @return Storage size in bytes.
     * @throws std::exception When shape or element-size validation fails.
     */
    [[nodiscard]] inline size_t ComputeByteCount(const STensorDescriptor &descriptor)
    {
        const size_t num_elements = ComputeElementCount(descriptor.shape);
        const size_t element_size = GetTensorElementTypeSize(descriptor.dtype);
        return CheckedMultiply(num_elements, element_size, "Tensor byte count overflows size_t.");
    }

    /**
     * @brief Allocate an owned output buffer for a concrete tensor descriptor.
     * @param descriptor Concrete dense tensor descriptor copied into the result.
     * @return Zero-initialized owned byte storage of the required size.
     */
    [[nodiscard]] inline STensorBuffer MakeOwnedTensorBuffer(const STensorDescriptor &descriptor)
    {
        STensorBuffer buffer;
        buffer.descriptor = descriptor;
        buffer.storage.resize(ComputeByteCount(descriptor));
        return buffer;
    }

    /**
     * @brief Validate a host input view before handing memory to a backend.
     * @param expected_descriptor Model input contract.
     * @param input_view Caller-owned runtime tensor view.
     * @param backend_name Name included in actionable errors.
     * @throws std::invalid_argument When dtype, memory, shape, size, or pointer
     *         validation fails.
     */
    inline void ValidateHostTensorView(const STensorDescriptor &expected_descriptor,
                                       const STensorView &input_view,
                                       const std::string &backend_name)
    {
        if (input_view.descriptor.dtype != expected_descriptor.dtype)
        {
            throw std::invalid_argument("Input dtype mismatch for tensor: " + expected_descriptor.name);
        }

        if (input_view.descriptor.location != EMemoryLocation::host)
        {
            throw std::invalid_argument("Only host memory inputs are supported in the " + backend_name + " backend.");
        }

        if (!IsShapeCompatible(expected_descriptor.shape, input_view.descriptor.shape))
        {
            throw std::invalid_argument("Input shape mismatch for tensor: " + expected_descriptor.name);
        }

        const size_t expected_bytes = ComputeByteCount(input_view.descriptor);
        if (input_view.bytes != expected_bytes)
        {
            throw std::invalid_argument("Input byte count mismatch for tensor: " + expected_descriptor.name);
        }

        if (expected_bytes > 0U && input_view.data == nullptr)
        {
            throw std::invalid_argument("Input tensor data pointer is null for tensor: " + expected_descriptor.name);
        }
    }

    /**
     * @brief Order input views by model metadata.
     *
     * Named inputs are matched by tensor name. Unnamed inputs preserve caller
     * order. Mixing named and unnamed tensors is rejected to avoid silent
     * misrouting.
     * @param expected_inputs Model input descriptors in backend order.
     * @param provided_inputs Caller-provided input views.
     * @return Pointers into `provided_inputs` ordered like `expected_inputs`.
     * @throws std::invalid_argument When input naming or cardinality is invalid.
     */
    [[nodiscard]] inline std::vector<const STensorView *> OrderInputViews(
        const std::vector<STensorDescriptor> &expected_inputs,
        const std::vector<STensorView> &provided_inputs)
    {
        bool has_named_inputs = false;
        bool has_unnamed_inputs = false;
        for (const STensorView &input : provided_inputs)
        {
            has_named_inputs = has_named_inputs || !input.descriptor.name.empty();
            has_unnamed_inputs = has_unnamed_inputs || input.descriptor.name.empty();
        }

        if (has_named_inputs && has_unnamed_inputs)
        {
            throw std::invalid_argument("Mixed named and unnamed input tensors are not supported.");
        }

        if (has_named_inputs)
        {
            for (size_t i = 0; i < provided_inputs.size(); ++i)
            {
                for (size_t j = i + 1; j < provided_inputs.size(); ++j)
                {
                    if (provided_inputs[i].descriptor.name == provided_inputs[j].descriptor.name)
                    {
                        throw std::invalid_argument("Duplicate input tensor name: " +
                                                    provided_inputs[i].descriptor.name);
                    }
                }
            }
        }

        if (provided_inputs.size() != expected_inputs.size())
        {
            throw std::invalid_argument("Input tensor count does not match model metadata.");
        }

        std::vector<const STensorView *> ordered_inputs;
        ordered_inputs.reserve(expected_inputs.size());

        if (!has_named_inputs)
        {
            for (const STensorView &input : provided_inputs)
            {
                ordered_inputs.push_back(&input);
            }
            return ordered_inputs;
        }

        for (const STensorDescriptor &expected_descriptor : expected_inputs)
        {
            const STensorView *matched_input = nullptr;
            for (const STensorView &provided_input : provided_inputs)
            {
                if (provided_input.descriptor.name == expected_descriptor.name)
                {
                    matched_input = &provided_input;
                    break;
                }
            }

            if (matched_input == nullptr)
            {
                throw std::invalid_argument("Missing input tensor: " + expected_descriptor.name);
            }

            ordered_inputs.push_back(matched_input);
        }

        return ordered_inputs;
    }

    /**
     * @brief Join execution-target names for backend diagnostics.
     * @param targets Priority-ordered execution targets.
     * @return Comma-separated names, or `none` when empty.
     */
    [[nodiscard]] inline std::string JoinExecutionTargets(const std::vector<EExecutionTarget> &targets)
    {
        if (targets.empty())
        {
            return "none";
        }

        std::string joined_targets;
        for (const EExecutionTarget target : targets)
        {
            if (!joined_targets.empty())
            {
                joined_targets += ",";
            }
            joined_targets += ToString(target);
        }

        return joined_targets;
    }
}
