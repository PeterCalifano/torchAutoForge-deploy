/*

ORT-MEX (Standalone): Run ONNX in Matlab with ONNXRuntime, application src

Author: Giacomo Battaglia, Politecnico di Milano

Usage: onnx_infer <onnx_file> <input_dims> <input_names> <output_names>

Compile with:
(Windows) cl.exe^
           /I"C:\path\to\onnxruntime-win-x64-gpu-X.X.X\include"^
           /I"C:\path\to\opencv\build\include"^
           /EHsc^
           onnxruntime_inference_standalone.cpp /Fe"%cd%\onnx_infer.exe"^
           /link^
           /LIBPATH:"C:\path\to\onnxruntime-win-x64-gpu-X.X.X\lib" onnxruntime.lib^
           /LIBPATH:"C:\path\to\onnxruntime-win-x64-gpu-1.22.0\lib" onnxruntime_providers_cuda.lib

*/

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <codecvt>
#include <locale>
#include <memory>
#include <algorithm>
#include <sstream>
#include <stdexcept>
// #include <opencv2/opencv.hpp>

// #define __TRYCATCH
#define DEBUG                       false
#define ORT_LOGGING_LEVEL           ORT_LOGGING_LEVEL_ERROR

#define printd(s)       do{if (DEBUG) {std::cout << s << std::endl;}}while(0)
#define array_size(v)   sizeof(v)/sizeof(v[0])

int vecprodi(std::vector<int>& vec) {
    int prod = 1;
    for (auto &v: vec) {
        prod *= v;
    }
    return prod;
}

// Mapping: ONNXTensorElementDataType → sizeof(type)
inline size_t GetONNXTypeSize(ONNXTensorElementDataType dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return sizeof(double);
        default:
            throw std::runtime_error("Unsupported ONNX data type in output.");
    }
}

__declspec(dllexport) uint8_t* run_onnx_inference(const wchar_t* model_path, float* input_data, std::vector<int> input_dims, 
                                                const char* input_names[], const char* output_names[],
                                                size_t n_inputs, size_t n_outputs,
                                                size_t& output_data_size) {
    printd("Init Ort::Env");
     
    static Ort::Env env(ORT_LOGGING_LEVEL, "ONNXModel");

    printd("Init Ort::Session");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Check available execution providers
    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();
    
    if (cuda_available) {
        printd("CUDAExecutionProvider is available. Enabling GPU inference.");
        OrtCUDAProviderOptions options;
        options.device_id = 0;
        // options.arena_extend_strategy = -1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
        // options.gpu_mem_limit = 8L * 1024 * 1024 * 1024;
        // options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
        options.do_copy_in_default_stream = 1;
        options.user_compute_stream = nullptr;
        options.default_memory_arena_cfg = nullptr;
        session_options.AppendExecutionProvider_CUDA(options);
    } else {
        printd("CUDAExecutionProvider not available. Falling back to CPU.");
    }

    Ort::Session session(env, model_path, session_options);

    printd("Init allocator");
    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    std::vector<int64_t> input_shape = {input_dims[0], input_dims[1], input_dims[2], input_dims[3]};

    size_t input_tensor_size = vecprodi(input_dims);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_tensor_size, input_shape.data(), input_shape.size());

    // Inference
    printd("Perform inference");

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, n_inputs, output_names, n_outputs);
    
    printd("Number of outputs: " << output_tensors.size());
    
    printd("Packing outputs");
    size_t offset = 0;

    if (output_tensors.size() != n_outputs) {
        throw std::runtime_error("Output number not coherent with network output.");
    }

#if (DEBUG)
    if (DEBUG) {
        printd("Number of outputs: " << output_tensors.size());
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            Ort::Value& tensor = output_tensors[i];

            if (tensor.IsTensor()) {
                Ort::TensorTypeAndShapeInfo shape_info = tensor.GetTensorTypeAndShapeInfo();

                // Data type
                ONNXTensorElementDataType type = shape_info.GetElementType();

                // Shape
                std::vector<int64_t> shape = shape_info.GetShape();

                // Number of elements
                size_t num_elements = shape_info.GetElementCount();

                // Optional: Size of dimensions
                std::cout << "Output " << i << " shape: [";
                for (size_t j = 0; j < shape.size(); ++j) {
                    std::cout << shape[j] << (j < shape.size() - 1 ? ", " : "");
                }
                std::cout << "]" << std::endl;

                printd("Num elements: " << num_elements);
                printd("Data type enum: " << type);  // e.g., ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            }
        }
    }
#endif

    // Prepare metadata
    std::vector<std::vector<size_t>> all_shapes;
    std::vector<ONNXTensorElementDataType> all_types;
    std::vector<size_t> all_counts;
    size_t total_header_entries = 1;  // num_outputs
    size_t total_data_bytes = 0;
    
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        auto tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
    
        auto shape = tensor_info.GetShape();
        std::vector<size_t> shape_u(shape.begin(), shape.end());
        all_shapes.push_back(shape_u);
    
        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        all_types.push_back(dtype);
    
        size_t count = tensor_info.GetElementCount();
        all_counts.push_back(count);
    
        total_data_bytes += count * GetONNXTypeSize(dtype);
        total_header_entries += 2 + shape_u.size();  // dtype + rank + dims
    }
    
    // Total header size in bytes (each entry stored as float)
    size_t header_bytes = total_header_entries * sizeof(float);
    
    // Allocate unified buffer: header + raw tensor data
    output_data_size = header_bytes + total_data_bytes;
    uint8_t* output_data = new uint8_t[output_data_size];
    
    // Populate header (write float entries as bytes)
    float* header_ptr = reinterpret_cast<float*>(output_data);
    size_t header_offset = 0;
    header_ptr[header_offset++] = static_cast<float>(output_tensors.size());
    
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        const auto& shape = all_shapes[i];
        header_ptr[header_offset++] = static_cast<float>(all_types[i]);    // Data type enum
        header_ptr[header_offset++] = static_cast<float>(shape.size());    // Rank
        for (size_t d : shape) {
            header_ptr[header_offset++] = static_cast<float>(d);           // Shape dims
        }
    }
    
    // Copy raw tensor data directly after the header
    size_t data_offset = header_bytes;
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        void* src = output_tensors[i].GetTensorMutableData<void>();
        size_t bytes = all_counts[i] * GetONNXTypeSize(all_types[i]);
        std::memcpy(output_data + data_offset, src, bytes);
        data_offset += bytes;
    }

    return output_data;
}

float* loadFloatArrayBinary(const std::string& filename, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading.");

    float* arr = new float[size];
    in.read(reinterpret_cast<char*>(arr), size * sizeof(float));

    if (!in) {
        delete[] arr;
        throw std::runtime_error("Error reading data from file.");
    }

    in.close();
    return arr;
}

std::vector<int> parseShapeArg(const std::string& arg) {
    if (arg.front() != '[' || arg.back() != ']') {
        throw std::invalid_argument("Input must be in format [N,C,H,W]");
    }

    std::string inner = arg.substr(1, arg.size() - 2); // remove [ and ]
    std::stringstream ss(inner);
    std::string token;
    std::vector<int> shape;

    while (std::getline(ss, token, ',')) {
        shape.push_back(std::stoi(token));
    }

    if (shape.size() != 4) {
        throw std::invalid_argument("Exactly 4 dimensions expected: [N,C,H,W]");
    }

    return shape;
}

// Parse "[input1,input2]" → const char*[]
void parseBracketedListToCStrings(const std::string& input, std::vector<std::string>& storage, std::vector<const char*>& out_ptrs) {
    if (input.empty() || input.front() != '[' || input.back() != ']') {
        throw std::runtime_error("Expected string format: [input1,input2,...]");
    }

    std::string inner = input.substr(1, input.size() - 2);  // Remove [ and ]
    std::stringstream ss(inner);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Optional: trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        storage.push_back(item);                  // Keep the string alive
    }

    out_ptrs.reserve(storage.size());
    for (auto & str: storage) {
        out_ptrs.push_back(str.c_str());
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: onnx_infer.exe <onnx_file> <input_dims> <input_names> <output_names>\n";
        return 1;
    }

    printd("Starting ONNX inference...");

    std::string model_path{argv[1]};

    printd("Allocating buffers...");
    
    std::vector<int> input_dims = parseShapeArg(argv[2]);
    
    int input_size = vecprodi(input_dims);
    float* input_data = loadFloatArrayBinary("./onnxruntime_inference_input.bin", input_size);
    
    std::wstring wide_input(model_path.begin(), model_path.end());

    std::string input_str(argv[3]);
    std::vector<std::string> input_storage;                   
    std::vector<const char*> input_c_strings;           
    parseBracketedListToCStrings(input_str, input_storage, input_c_strings);

    std::string output_str(argv[4]);
    std::vector<std::string> output_storage;                   
    std::vector<const char*> output_c_strings;           
    parseBracketedListToCStrings(output_str, output_storage, output_c_strings);

    printd("Running ONNX inference...");

    size_t output_data_size = 0;
#ifdef __TRYCATCH
    uint8_t* output_data;
    try {
#endif
    uint8_t* output_data = run_onnx_inference(wide_input.c_str(), input_data, input_dims, 
                                             input_c_strings.data(), output_c_strings.data(), 
                                             input_c_strings.size(), output_c_strings.size(),
                                             output_data_size);

    // Write output
    printd("Writing output to file...");
    std::ofstream outfile("./onnxruntime_inference_output.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_data), output_data_size * sizeof(uint8_t));
    outfile.close();

#ifdef __TRYCATCH
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
#endif

    delete[] input_data;
    delete[] output_data;

    printd("End");

    return 0;
}
