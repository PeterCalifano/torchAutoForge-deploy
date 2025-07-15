#include <iostream>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>

using deploy_ort::CInferenceManager_ORT;

int main()
{
    CInferenceManager_ORT inference_manager("path/to/model.onnx");
    std::cout << "This is a placeholder main function." << "\n";
    return 0;
}