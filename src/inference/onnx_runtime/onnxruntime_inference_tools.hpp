// ORT
#include <onnxruntime_cxx_api.h>
// STL
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

    class CInferenceManager_ORT
    {
      public:
        // Constructor
        CInferenceManager_ORT() = default;
        // TODO: do not use strings to define paths! Scott Meyer's tip.
        CInferenceManager_ORT(std::string model_path)
        {
            print_info("Initintializing CInferenceManager_ORT with model path: " + model_path);
            // Initialize ORT environment
            Ort::Env exec_env(ORT_LOGGING_LEVEL, "ONNXModel");
        }
    };

};