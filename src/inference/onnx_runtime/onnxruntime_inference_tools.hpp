#pragma once

// ORT
#include <onnxruntime_cxx_api.h>
// STL
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Autoforge deploy
#include <auxiliary/common_defs.h>
#include <auxiliary/common_ops.h>

// MACROS
#define DEBUG false
#define ORT_LOGGING_LEVEL ORT_LOGGING_LEVEL_ERROR

#define printd(s)                   \
    do                              \
    {                               \
        if (DEBUG || !NDEBUG)       \
        {                           \
            std::cout << s << "\n"; \
        }                           \
    } while (0)
#define get_array_size(v) sizeof(v) / sizeof(v[0])
#define print_info(string)                       \
    do                                           \
    {                                            \
        std::cout << "INFO: " << string << "\n"; \
    } while (0)

namespace deploy_ort
{
    // DOUBT Can this be determined at compile time?
    // Mapping: ONNXTensorElementDataType → sizeof(type)
    constexpr size_t GetONNXTypeSize(ONNXTensorElementDataType dtype)
    {
        switch (dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return sizeof(double);
        default:
            throw std::runtime_error("Unsupported ONNX data type in output.");
        }
    }

    /**
     * @brief A class to manage ONNX Runtime environment setup and inference.
     * @author Peter C.
     *
     */
    class CInferenceManager_ORT
    {
      public:
        // CONSTRUCTORS
        CInferenceManager_ORT() = default;
        CInferenceManager_ORT(const std::string session_config_path); // TODO implement parser
        CInferenceManager_ORT(const bool inplace_init,
                              const std::string &model_path,
                              const Ort::SessionOptions session_options = Ort::SessionOptions());

        // DESTRUCTOR
        ~CInferenceManager_ORT() = default;

      public:
        // GETTERS

        // SETTERS

        // METHODS
        template <typename infer_type>
        void initialize();

        template <typename infer_type>
        void infer();

      protected:
        // DATA MEMBERS
        fs::path model_path_{};
        // ORT environment and session
        Ort::Env exec_env_{};
        Ort::SessionOptions session_options_{};
        std::unique_ptr<Ort::Session> session_ptr_{nullptr};
        // Memory management
        Ort::AllocatorWithDefaultOptions allocator_{};
        Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

        // Tensor allocation
        Ort::Value input_tensor_{nullptr};
        Ort::Value output_tensor_{nullptr};

        // Input/Output specifications
        std::shared_ptr<deploy_common_defs::SInputOutputSpecs<int64_t, int64_t>> input_output_specs_{};
    };

    // Template method definitions
    /**
     * @brief Initializes the ONNX Runtime environment and session.
     *
     */
    template <typename infer_type>
    void CInferenceManager_ORT::initialize()
    {
        // Initialize ORT environment
        exec_env_ = Ort::Env(ORT_LOGGING_LEVEL, "ONNXModel");

        // Define ort session
        if (session_ptr_ == nullptr)
        {
            print_info("Creating Ort::Session with model path: " + model_path_.string());
            session_ptr_ = std::make_unique<Ort::Session>(exec_env_,
                                                          static_cast<const char *>(model_path_.string().c_str()),
                                                          session_options_);

            // Define input/output specifications
            print_info("Defining input/output specifications for the model...");
            input_output_specs_ = std::make_shared<deploy_common_defs::SInputOutputSpecs<int64_t, int64_t>>(); // TODO

            // Define memory info
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemType::OrtMemTypeDefault); // TODO

            // Allocate input and output tensors
            // FIXME
            input_tensor_ = Ort::Value::CreateTensor<infer_type>(allocator_,
                                                                 input_output_specs_->input_shapes.data(),
                                                                 input_output_specs_->num_elements_linear_input_array);

            output_tensor_ = Ort::Value::CreateTensor<infer_type>(allocator_,
                                                                  input_output_specs_->output_shapes.data(),
                                                                  input_output_specs_->num_elements_linear_output_array);
        }
        else
        {
            print_info("Ort::Session already initialized.");
        }
    };

    template <typename infer_type>
    void CInferenceManager_ORT::infer()
    {
#if (VERBOSE)
        // Placeholder for inference logic
        print_info("Running inference with model: " + model_path_.string());
#endif

        // Run session inference
        Ort::RunOptions run_options;
        run_options.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL);
        run_options.SetRunTag("InferenceRun");

        // Run the session
        session_ptr_->Run(run_options,
                          input_output_specs_->input_names.data(),   // Input names
                          &input_tensor_,                            // Input tensor
                          input_output_specs_->input_names.size(),   // Number of inputs
                          input_output_specs_->output_names.data(),  // Output names
                          &output_tensor_,                           // Output tensor
                          input_output_specs_->output_names.size()); // Number of outputs
    }
};