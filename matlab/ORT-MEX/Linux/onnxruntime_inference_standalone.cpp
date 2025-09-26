/*

ORT-MEX (Standalone): Run ONNX in Matlab with ONNXRuntime, application src

Author: Giacomo Battaglia, Pietro Califano Politecnico di Milano

Usage: ./this_file <onnx_file> <input_dims> <input_names> <output_names>


*/

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
// #include <opencv2/opencv.hpp>

// #define __TRYCATCH
#define DEBUG false
#define ORT_LOGGING_LEVEL ORT_LOGGING_LEVEL_ERROR

#define printd(s)                        \
    do                                   \
    {                                    \
        if (DEBUG)                       \
        {                                \
            std::cout << s << std::endl; \
        }                                \
    } while (0)
#define array_size(v) sizeof(v) / sizeof(v[0])

/**
 * @brief Function computing the product of all elements in a vector of integers.
 * @author Giacomo Battaglia
 * @param vec
 * @return int
 */
int vecprodi(std::vector<int> &vec)
{
    int prod = 1;
    for (auto &v : vec)
    {
        prod *= v;
    }
    return prod;
}

/**
 * @brief Loads a binary file into a byte array.
 *
 * @param filename The name of the file to load.
 * @param out_size The size of the loaded data.
 * @return uint8_t* A pointer to the loaded data.
 */
uint8_t *loadByteArrayBinary(const std::string &filename, size_t &out_size)
{
    // Open the file in binary mode
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in)
    {
        throw std::runtime_error("Cannot open file for reading.");
    }

    // Get file size in bytes
    std::streamsize file_size = in.tellg();
    if (file_size < 0)
    {
        throw std::runtime_error("Failed to determine file size.");
    }

    out_size = static_cast<size_t>(file_size);
    uint8_t *arr = new uint8_t[out_size];

    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(arr), file_size);

    if (!in)
    {
        delete[] arr;
        throw std::runtime_error("Error reading data from file.");
    }

    in.close();
    return arr;
}

// Mapping: ONNXTensorElementDataType → sizeof(type)
size_t GetONNXTypeSize(const ONNXTensorElementDataType dtype)
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
 * @brief 
 * 
 * @param model_path 
 * @param input_data 
 * @param input_shapes 
 * @param input_types 
 * @param input_names 
 * @param output_names 
 * @param n_inputs 
 * @param n_outputs 
 * @param output_data_size 
 * @return uint8_t* 
 */
uint8_t *run_onnx_inference(const wchar_t *model_path,
                            const uint8_t *input_data[],
                            const std::vector<std::vector<int64_t>> &input_shapes,
                            const ONNXTensorElementDataType input_types[],
                            const char *input_names[],
                            const char *output_names[],
                            size_t n_inputs,
                            size_t n_outputs,
                            size_t &output_data_size)
{
    printd("Init Ort::Env");

    // Initialize ORT environment
    static Ort::Env env(ORT_LOGGING_LEVEL, "ONNXModel");

    printd("Init Ort::Session");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Get available execution provider for acceleration
    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();

    if (cuda_available)
    {
        printd("CUDAExecutionProvider is available. Enabling GPU inference.");
        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(options);
    }
    else
    {
        printd("CUDAExecutionProvider not available. Falling back to CPU.");
    }

    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Create input Ort::Value tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(n_inputs);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < n_inputs; ++i)
    {
        const auto &shape = input_shapes[i];
        size_t elem_count = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
        size_t type_size;

        switch (input_types[i])
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_size = 1;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_size = 2;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_size = 4;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_size = 8;
            break;
        default:
            std::cerr << "Unsupported input tensor data type.\n";
            return nullptr;
        }

        // Create tensor based on dtype
        Ort::Value tensor{nullptr};
        switch (input_types[i])
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                     reinterpret_cast<float *>(const_cast<uint8_t *>(input_data[i])),
                                                     elem_count,
                                                     shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            tensor = Ort::Value::CreateTensor<double>(memory_info,
                                                      reinterpret_cast<double *>(const_cast<uint8_t *>(input_data[i])),
                                                      elem_count,
                                                      shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                                                       reinterpret_cast<int64_t *>(const_cast<uint8_t *>(input_data[i])),
                                                       elem_count,
                                                       shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            tensor = Ort::Value::CreateTensor<int32_t>(memory_info,
                                                       reinterpret_cast<int32_t *>(const_cast<uint8_t *>(input_data[i])),
                                                       elem_count,
                                                       shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            tensor = Ort::Value::CreateTensor<int16_t>(memory_info,
                                                       reinterpret_cast<int16_t *>(const_cast<uint8_t *>(input_data[i])),
                                                       elem_count,
                                                       shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            tensor = Ort::Value::CreateTensor<int8_t>(memory_info,
                                                      reinterpret_cast<int8_t *>(const_cast<uint8_t *>(input_data[i])),
                                                      elem_count,
                                                      shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            tensor = Ort::Value::CreateTensor<uint16_t>(memory_info,
                                                        reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(input_data[i])),
                                                        elem_count,
                                                        shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            tensor = Ort::Value::CreateTensor<uint8_t>(memory_info,
                                                       const_cast<uint8_t *>(input_data[i]),
                                                       elem_count,
                                                       shape.data(), shape.size());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            tensor = Ort::Value::CreateTensor<bool>(memory_info,
                                                    reinterpret_cast<bool *>(const_cast<uint8_t *>(input_data[i])),
                                                    elem_count,
                                                    shape.data(), shape.size());
            break;
        default:
            std::cerr << "Data type not implemented for tensor creation.\n";
            return nullptr;
        }

        input_tensors.push_back(std::move(tensor));
    }

#if (DEBUG)
    if (DEBUG)
    {
        printd("Number of inputs: " << input_tensors.size());
        for (size_t i = 0; i < input_tensors.size(); ++i)
        {
            Ort::Value &tensor = input_tensors[i];

            if (tensor.IsTensor())
            {
                Ort::TensorTypeAndShapeInfo shape_info = tensor.GetTensorTypeAndShapeInfo();

                // Data type
                ONNXTensorElementDataType type = shape_info.GetElementType();

                // Shape
                std::vector<int64_t> shape = shape_info.GetShape();

                // Number of elements
                size_t num_elements = shape_info.GetElementCount();

                // Optional: Size of dimensions
                std::cout << "Input " << i << " shape: [";
                for (size_t j = 0; j < shape.size(); ++j)
                {
                    std::cout << shape[j] << (j < shape.size() - 1 ? ", " : "");
                }
                std::cout << "]" << std::endl;

                printd("Num elements: " << num_elements);
                printd("Data type enum: " << type); // e.g., ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            }
        }
    }
#endif

    printd("Perform inference");
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names,
                                      input_tensors.data(),
                                      n_inputs,
                                      output_names,
                                      n_outputs);

    printd("Packing outputs");
    size_t offset = 0;

    if (output_tensors.size() != n_outputs)
    {
        throw std::runtime_error("Output number not coherent with network output.");
    }

#if (DEBUG)
    if (DEBUG)
    {
        printd("Number of outputs: " << output_tensors.size());
        for (size_t i = 0; i < output_tensors.size(); ++i)
        {
            Ort::Value &tensor = output_tensors[i];

            if (tensor.IsTensor())
            {
                Ort::TensorTypeAndShapeInfo shape_info = tensor.GetTensorTypeAndShapeInfo();

                // Data type
                ONNXTensorElementDataType type = shape_info.GetElementType();

                // Shape
                std::vector<int64_t> shape = shape_info.GetShape();

                // Number of elements
                size_t num_elements = shape_info.GetElementCount();

                // Optional: Size of dimensions
                std::cout << "Output " << i << " shape: [";
                for (size_t j = 0; j < shape.size(); ++j)
                {
                    std::cout << shape[j] << (j < shape.size() - 1 ? ", " : "");
                }
                std::cout << "]" << std::endl;

                printd("Num elements: " << num_elements);
                printd("Data type enum: " << type); // e.g., ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            }
        }
    }
#endif

    // Prepare metadata
    printd("Prepare metadata");

    std::vector<std::vector<size_t>> all_shapes;
    std::vector<ONNXTensorElementDataType> all_types;
    std::vector<size_t> all_counts;
    size_t total_header_entries = 1; // num_outputs
    size_t total_data_bytes = 0;

    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
        auto tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();

        auto shape = tensor_info.GetShape();
        std::vector<size_t> shape_u(shape.begin(), shape.end());
        all_shapes.push_back(shape_u);

        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        all_types.push_back(dtype);

        size_t count = tensor_info.GetElementCount();
        all_counts.push_back(count);

        total_data_bytes += count * GetONNXTypeSize(dtype);
        total_header_entries += 2 + shape_u.size(); // dtype + rank + dims
    }

    printd("Prepare header");

    // Total header size in bytes (each entry stored as float)
    size_t header_bytes = total_header_entries * sizeof(float);

    // Allocate unified buffer: header + raw tensor data
    output_data_size = header_bytes + total_data_bytes;
    uint8_t *output_data = new uint8_t[output_data_size];

    //

    // Populate header (write float entries as bytes)
    float *header_ptr = reinterpret_cast<float *>(output_data);
    size_t header_offset = 0;
    header_ptr[header_offset++] = static_cast<float>(output_tensors.size());

    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
        const auto &shape = all_shapes[i];
        header_ptr[header_offset++] = static_cast<float>(all_types[i]); // Data type enum
        header_ptr[header_offset++] = static_cast<float>(shape.size()); // Rank
        for (size_t d : shape)
        {
            header_ptr[header_offset++] = static_cast<float>(d); // Shape dims
        }
    }

    printd("Prepare output");

    // Copy raw tensor data directly after the header
    size_t data_offset = header_bytes;
    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
        void *src = output_tensors[i].GetTensorMutableData<void>();
        size_t bytes = all_counts[i] * GetONNXTypeSize(all_types[i]);
        std::memcpy(output_data + data_offset, src, bytes);
        data_offset += bytes;
    }

    printd("End run_onnx_inference");

    return output_data;
}

// float* loadFloatArrayBinary(const std::string& filename, size_t size) {
//     std::ifstream in(filename, std::ios::binary);
//     if (!in) throw std::runtime_error("Cannot open file for reading.");
//
//     float* arr = new float[size];
//     in.read(reinterpret_cast<char*>(arr), size * sizeof(float));
//
//     if (!in) {
//         delete[] arr;
//         throw std::runtime_error("Error reading data from file.");
//     }
//
//     in.close();
//     return arr;
// }

std::vector<int> parseShapeArg(const std::string &arg)
{
    if (arg.front() != '[' || arg.back() != ']')
    {
        throw std::invalid_argument("Input must be in format [N,C,H,W]");
    }

    std::string inner = arg.substr(1, arg.size() - 2); // remove [ and ]
    std::stringstream ss(inner);
    std::string token;
    std::vector<int> shape;

    while (std::getline(ss, token, ','))
    {
        shape.push_back(std::stoi(token));
    }

    if (shape.size() != 4)
    {
        throw std::invalid_argument("Exactly 4 dimensions expected: [N,C,H,W]");
    }

    return shape;
}

// Parse "[input1,input2]" → const char*[]
void parseBracketedListToCStrings(const std::string &input, std::vector<std::string> &storage, std::vector<const char *> &out_ptrs)
{
    if (input.empty() || input.front() != '[' || input.back() != ']')
    {
        throw std::runtime_error("Expected string format: [input1,input2,...]");
    }

    std::string inner = input.substr(1, input.size() - 2); // Remove [ and ]
    std::stringstream ss(inner);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        // Optional: trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        storage.push_back(item); // Keep the string alive
    }

    out_ptrs.reserve(storage.size());
    for (auto &str : storage)
    {
        out_ptrs.push_back(str.c_str());
    }
}

// MAIN FUNCTION
int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: onnx_infer <onnx_file> <input_names> <output_names> <add_batch>\n";
        return 1;
    }

    std::cout << "Starting ONNX inference session..." << "\n";

    // Get model path
    std::string model_path{argv[1]};
    std::wstring wide_input(model_path.begin(), model_path.end());

    std::string add_batch_str(argv[4]);
    bool add_batch;

    if (add_batch_str == "true" || add_batch_str == "1")
    {
        add_batch = true;
    }
    else if (add_batch_str == "false" || add_batch_str == "0")
    {
        add_batch = false;
    }
    else
    {
        std::cerr << "Invalid value for <add_batch>. Use 'true' or 'false'.\n";
        return 1;
    }

    printd("Allocating buffers...");

    // Step 1: Load the binary file
    size_t total_input_size = 0;
    uint8_t *input_buffer = loadByteArrayBinary("./onnxruntime_inference_input.bin", total_input_size);

    // Step 2: Parse the header and extract each input
    const float *header = reinterpret_cast<const float *>(input_buffer);
    size_t offset = 0;

    size_t num_inputs = static_cast<size_t>(header[offset++]);

    struct InputTensor
    {
        ONNXTensorElementDataType dtype;
        std::vector<int64_t> shape;
        std::vector<uint8_t> data; // raw bytes
    };

    std::vector<InputTensor> inputs(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        int dtype_int = static_cast<int>(header[offset++]);
        inputs[i].dtype = static_cast<ONNXTensorElementDataType>(dtype_int);

        size_t rank = static_cast<size_t>(header[offset++]);
        std::vector<int64_t> shape(rank);

        size_t element_count = 1;
        for (size_t j = 0; j < rank; ++j)
        {
            int dim = static_cast<int>(header[offset++]);
            shape[j] = dim;
            element_count *= dim;
        }

        // Reverse dimensions to match ONNX layout (e.g., from NHWC to NCHW)
        std::reverse(shape.begin(), shape.end());

        // Add batch dimension if missing (e.g., [C, H, W] -> [1, C, H, W])
        if (add_batch)
        {
            shape.insert(shape.begin(), 1);
        }

        inputs[i].shape = std::move(shape);

        size_t type_size;
        switch (inputs[i].dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_size = 1;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_size = 2;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_size = 4;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_size = 8;
            break;
        default:
            std::cerr << "Unsupported input tensor data type.\n";
            exit(1);
        }

        size_t byte_size = element_count * type_size;
        const uint8_t *raw_ptr = reinterpret_cast<const uint8_t *>(header + offset);
        inputs[i].data.assign(raw_ptr, raw_ptr + byte_size);
        offset += (byte_size + sizeof(float) - 1) / sizeof(float); // align to float boundary
    }

    // Collect type and shape info for inference
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<const uint8_t *> input_data_ptrs;
    std::vector<std::vector<int64_t>> input_shapes;

    for (const auto &input : inputs)
    {
        input_types.push_back(input.dtype);
        input_data_ptrs.push_back(input.data.data());
        input_shapes.push_back(input.shape);
    }

    std::string input_str(argv[2]);
    std::vector<std::string> input_storage;
    std::vector<const char *> input_c_strings;
    parseBracketedListToCStrings(input_str, input_storage, input_c_strings);

    std::string output_str(argv[3]);
    std::vector<std::string> output_storage;
    std::vector<const char *> output_c_strings;
    parseBracketedListToCStrings(output_str, output_storage, output_c_strings);

    printd("Running ONNX inference...");

    size_t output_data_size = 0;
    uint8_t *output_data = nullptr;
#ifdef __TRYCATCH
    try
    {
#endif
        // uint8_t* output_data = run_onnx_inference(wide_input.c_str(), input_data, input_dims,
        //                                          input_c_strings.data(), output_c_strings.data(),
        //                                          input_c_strings.size(), output_c_strings.size(),
        //                                          output_data_size);

        output_data = run_onnx_inference(
            wide_input.c_str(),
            input_data_ptrs.data(),
            input_shapes,
            input_types.data(),
            input_c_strings.data(),
            output_c_strings.data(),
            input_data_ptrs.size(),
            output_c_strings.size(),
            output_data_size);

        // Write output
        printd("Writing output to file...");
        std::ofstream outfile("./onnxruntime_inference_output.bin", std::ios::binary);
        outfile.write(reinterpret_cast<char *>(output_data), output_data_size * sizeof(uint8_t));
        outfile.close();

#ifdef __TRYCATCH
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
#endif

    // delete[] input_data;
    delete[] output_data;

    printd("End");

    return 0;
}
