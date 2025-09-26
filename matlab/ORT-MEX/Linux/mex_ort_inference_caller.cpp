/*

ORT-MEX: Run ONNX in Matlab with ONNXRuntime

Author: Giacomo Battaglia, Politecnico di Milano

Function: [varargout] = onnxruntime_inference(onnx_file, input_tensor, [input_names, output_names])

Supports 4-dim input tensor, variable outputs (number, types, sizes)

Compile with:
(Windows) mex('-v', ...
              '-I"C:\path\to\onnxruntime-win-x64-gpu-X.X.X\include"', ...
              'onnxruntime_inference.cpp', ...
              '-L"C:\path\to\onnxruntime-win-x64-gpu-X.X.X\lib"', ...
              'onnxruntime.lib');

*/

#include "mex.h"
#include <algorithm>
#include <codecvt>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <string>
#include <vector>

mwSize vecprod_mwSize(std::vector<mwSize> &vec)
{
    mwSize prod = 1;
    for (auto &v : vec)
    {
        prod *= v;
    }
    return prod;
}

uint8_t *loadByteArrayBinary(const std::string &filename, size_t &out_size)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("Cannot open file for reading.");

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

/**
 * @brief
 *
 * @param nchw_data
 * @param matlab_data
 * @param N
 * @param C
 * @param H
 * @param W
 */
void copyNCHWToMxArray(const float *nchw_data, float *matlab_data,
                       size_t N, size_t C, size_t H, size_t W)
{
    // MATLAB layout: W x H x C x N (column-major)
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    int idx_cpp = n * C * H * W + c * H * W + h * W + w;
                    int idx_matlab = w + h * W + c * W * H + n * W * H * C;
                    matlab_data[idx_matlab] = nchw_data[idx_cpp];
                }
            }
        }
    }
}

/**
 * @brief Copies data from a MATLAB array to an NCHW formatted array.
 *
 * @param matlab_data The input MATLAB array (W x H x C x N).
 * @param nchw_data The output NCHW array (N x C x H x W).
 * @param N The batch size.
 * @param C The number of channels.
 * @param H D1
 * @param W D2
 */
void copyMxArrayToNCHW(const float *matlab_data, float *nchw_data,
                       size_t N, size_t C, size_t H, size_t W)
{
    // MATLAB layout: W x H x C x N (column-major)
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    int idx_cpp = n * C * H * W + c * H * W + h * W + w;
                    int idx_matlab = w + h * W + c * W * H + n * W * H * C;
                    nchw_data[idx_cpp] = matlab_data[idx_matlab];
                }
            }
        }
    }
}

void copyNCLToMxArray(const float *ncl_data, float *matlab_data,
                      int N, int C, int L)
{
    // MATLAB layout: L x C x N (column-major)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int l = 0; l < L; ++l)
            {
                int idx_cpp = n * C * L + c * L + l;
                int idx_matlab = l + c * L + n * L * C;
                matlab_data[idx_matlab] = ncl_data[idx_cpp];
            }
        }
    }
}

const char *mxArrayToCString(const mxArray *arr, std::string &buffer)
{
    if (!mxIsChar(arr))
    {
        mexErrMsgTxt("First input must be a string (e.g., model path).");
    }

    char *c_str = mxArrayToString(arr); // Allocates a new buffer (UTF-8)
    if (!c_str)
    {
        mexErrMsgTxt("Failed to convert MATLAB string to C string.");
    }

    buffer = std::string(c_str); // Copy to std::string to manage memory
    mxFree(c_str);               // Free MATLAB-allocated memory

    return buffer.c_str(); // Return const char* (valid as long as buffer is alive)
}

std::string dimsToString(const mwSize *dims, mwSize ndims)
{
    std::ostringstream oss;
    oss << "[";
    for (mwSize i = 0; i < ndims; ++i)
    {
        oss << dims[i];
        if (i != ndims - 1)
            oss << ",";
    }
    oss << "]";
    return oss.str();
}

// Converts MATLAB cell array or string array to "[str1,str2,...]" format
std::string mxArrayToBracketedStringList(const mxArray *arr)
{
    if (!mxIsCell(arr) && !mxIsChar(arr))
    {
        mexErrMsgTxt("Expected a char array or cell array of strings.");
    }

    std::ostringstream result;
    result << "[";

    if (mxIsChar(arr))
    {
        // Single string input
        char *str = mxArrayToString(arr);
        result << str;
        mxFree(str);
    }
    else
    {
        // Cell array
        mwSize num_elems = mxGetNumberOfElements(arr);
        for (mwSize i = 0; i < num_elems; ++i)
        {
            const mxArray *cell = mxGetCell(arr, i);
            if (!mxIsChar(cell))
            {
                mexErrMsgTxt("Each cell must contain a string.");
            }
            char *str = mxArrayToString(cell);
            result << str;
            if (i != num_elems - 1)
            {
                result << ",";
            }
            mxFree(str);
        }
    }

    result << "]";
    return result.str();
}

std::string mwSizeArrayToString(const mwSize *dims, mwSize ndims)
{
    std::ostringstream oss;

    for (mwSize i = 0; i < ndims; ++i)
    {
        oss << dims[i];
        if (i < ndims - 1)
        {
            oss << ",";
        }
    }

    return oss.str();
}

std::vector<std::vector<mwSize>> parseOutputDimsCellArray(const mxArray *cell_input)
{
    if (!mxIsCell(cell_input))
    {
        mexErrMsgTxt("Expected a cell array of output shapes.");
    }

    size_t num_outputs = mxGetNumberOfElements(cell_input);
    std::vector<std::vector<mwSize>> output_shapes;

    for (size_t i = 0; i < num_outputs; ++i)
    {
        const mxArray *shape_arr = mxGetCell(cell_input, i);

        if (!mxIsDouble(shape_arr) || mxIsComplex(shape_arr))
        {
            mexErrMsgTxt("Each output shape must be a real double array.");
        }

        size_t num_elems = mxGetNumberOfElements(shape_arr);
        if (num_elems < 1 || num_elems > 4)
        {
            mexErrMsgTxt("Each shape must have 1 to 4 elements ([M,], [M,L], [N,C,L], [N,C,H,W]).");
        }

        double *dims_ptr = mxGetPr(shape_arr);
        std::vector<mwSize> shape(num_elems);
        for (size_t j = 0; j < num_elems; ++j)
        {
            shape[j] = static_cast<mwSize>(dims_ptr[j]);
        }

        output_shapes.push_back(shape);
    }

    return output_shapes;
}

void unpackOutputsToPlhs(const uint8_t *output_with_header, mxArray **plhs)
{
    // Step 1: Read header as floats
    const float *header = reinterpret_cast<const float *>(output_with_header);
    size_t offset = 0;

    size_t num_outputs = static_cast<size_t>(header[offset++]);

    struct OutputMeta
    {
        ONNXTensorElementDataType dtype;
        std::vector<mwSize> shape;
    };

    std::vector<OutputMeta> outputs(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i)
    {
        outputs[i].dtype = static_cast<ONNXTensorElementDataType>(static_cast<int>(header[offset++]));

        size_t rank = static_cast<size_t>(header[offset++]);
        if (rank < 1 || rank > 4)
        {
            mexErrMsgTxt("Only ranks 1 to 4 are supported.");
        }

        outputs[i].shape.resize(rank);
        for (size_t j = 0; j < rank; ++j)
        {
            outputs[i].shape[j] = static_cast<mwSize>(header[offset++]);
        }
    }

    // Step 2: Point to raw binary data
    const uint8_t *raw_data = reinterpret_cast<const uint8_t *>(header + offset);

    size_t byte_offset = 0;
    for (size_t i = 0; i < num_outputs; ++i)
    {
        const auto &meta = outputs[i];
        const auto &dims = meta.shape;
        size_t rank = dims.size();

        size_t elem_count = 1;
        for (mwSize d : dims)
            elem_count *= d;

        mxClassID class_id;
        size_t type_size;

        switch (meta.dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            class_id = mxLOGICAL_CLASS;
            type_size = 1;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            class_id = mxUINT8_CLASS;
            type_size = 1;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            class_id = mxINT8_CLASS;
            type_size = 1;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            class_id = mxUINT16_CLASS;
            type_size = 2;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            class_id = mxINT16_CLASS;
            type_size = 2;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            class_id = mxINT32_CLASS;
            type_size = 4;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            class_id = mxINT64_CLASS;
            type_size = 8;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            class_id = mxSINGLE_CLASS;
            type_size = 4;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            class_id = mxDOUBLE_CLASS;
            type_size = 8;
            break;
        default:
            mexErrMsgTxt("Unsupported ONNX tensor data type.");
        }

        // Transpose from ONNX layout (e.g., NCHW) to MATLAB layout if needed
        std::vector<mwSize> matlab_dims(dims.rbegin(), dims.rend());

        // Always create 2D minimum for MATLAB
        if (matlab_dims.size() == 1)
        {
            matlab_dims.push_back(1);
        }

        plhs[i] = mxCreateNumericArray(matlab_dims.size(), matlab_dims.data(), class_id, mxREAL);
        void *dst = mxGetData(plhs[i]);

        const void *src = raw_data + byte_offset;
        std::memcpy(dst, src, elem_count * type_size);
        byte_offset += elem_count * type_size;
    }
}

#define DEBUG false
#define printd(s)                        \
    do                                   \
    {                                    \
        if (DEBUG)                       \
        {                                \
            std::cout << s << std::endl; \
        }                                \
    } while (0)

// TODO rework mex function 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5)
    {
        mexErrMsgTxt("Usage: onnxruntime_inference(application_filepath, model_path, input_names, output_names, add_batch, input_tensor)");
    }
    std::string input_file_str = "./onnxruntime_inference_input.bin";
    std::string output_file = "./onnxruntime_inference_output.bin";

    // Convert input
    // Parse input/output names
    std::string application_filepath = mxGetString(prhs[0]); // FIXME
    std::string model_path;
    mxArrayToCString(prhs[1], model_path);

    std::string input_names = mxArrayToBracketedStringList(prhs[2]);
    std::string output_names = mxArrayToBracketedStringList(prhs[3]);

    // Check that the 4th argument is a scalar logical or convertible to one
    bool add_batch = false;
    if (!mxIsLogicalScalar(prhs[4]) && !mxIsDouble(prhs[4]))
    {
        mexErrMsgTxt("add_batch must be a logical scalar or numeric scalar.");
    }
    else
    {
        add_batch = mxIsLogicalScalarTrue(prhs[4]) || (mxIsDouble(prhs[4]) && mxGetScalar(prhs[4]) != 0.0);
    }

    // Convert to string
    std::string add_batch_str = add_batch ? "true" : "false";

    size_t num_inputs = static_cast<size_t>(mxGetNumberOfElements(prhs[2]));
    if (num_inputs != static_cast<size_t>(nrhs - 4))
    {
        mexErrMsgTxt("Number of input tensors does not match number of input names.");
    }

    std::ofstream input_file(input_file_str, std::ios::binary);

    // Write input count
    float num_inputs_f = static_cast<float>(num_inputs);
    input_file.write(reinterpret_cast<char *>(&num_inputs_f), sizeof(float)); // DEVNOTE replace reinterpret cast (!)

    // For each input, define type and shape
    for (size_t i = 0; i < num_inputs; ++i)
    {
        const mxArray *input_array = prhs[4 + i];

        ONNXTensorElementDataType dtype;
        mxClassID class_id = mxGetClassID(input_array);
        size_t type_size;

        switch (class_id)
        {
        case mxLOGICAL_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            type_size = 1;
            break;
        case mxUINT8_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            type_size = 1;
            break;
        case mxINT8_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
            type_size = 1;
            break;
        case mxUINT16_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
            type_size = 2;
            break;
        case mxINT16_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
            type_size = 2;
            break;
        case mxINT32_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            type_size = 4;
            break;
        case mxINT64_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            type_size = 8;
            break;
        case mxSINGLE_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            type_size = 4;
            break;
        case mxDOUBLE_CLASS:
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
            type_size = 8;
            break;
        default:
            mexErrMsgTxt("Unsupported input type.");
        }

        // Write dtype
        float dtype_f = static_cast<float>(dtype);
        input_file.write(reinterpret_cast<char *>(&dtype_f), sizeof(float));

        // Get dimensions
        const mwSize *dims = mxGetDimensions(input_array);
        mwSize ndim = mxGetNumberOfDimensions(input_array);
        float rank_f = static_cast<float>(ndim);
        input_file.write(reinterpret_cast<char *>(&rank_f), sizeof(float));

        // Write each dimension
        for (mwSize d = 0; d < ndim; ++d)
        {
            float dim_f = static_cast<float>(dims[d]);
            input_file.write(reinterpret_cast<char *>(&dim_f), sizeof(float));
        }

        // Write actual data
        void *data_ptr = mxGetData(input_array);
        size_t elem_count = mxGetNumberOfElements(input_array);
        input_file.write(reinterpret_cast<char *>(data_ptr), elem_count * type_size);
    }

    input_file.close();
    std::string command = "./" + application_filepath + " " + model_path + " " + input_names + " " + output_names + " " + add_batch_str;
    printd("Running command: " + command);

    // TODO: add code to launch a second process to run the ONNX inference

    // TODO write inputs to pipe as text

    // TODO read back the outputs from the pipe
    
    printd("Reading output");
    // Read output
    size_t output_size = 0;
    uint8_t *output = loadByteArrayBinary(output_file, output_size);
    // Unpack to MATLAB outputs
    unpackOutputsToPlhs(output, plhs);

    std::remove(input_file_str.c_str());
    std::remove(output_file.c_str());

    // delete[] input_data;
}
