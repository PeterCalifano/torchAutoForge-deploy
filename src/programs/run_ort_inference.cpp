#include <filesystem>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <iostream>

using deploy_ort::CInferenceManager_ORT;

int main()
{

    std::filesystem::path model_path = "path/to/model.onnx";
    bool in_place_init = false;

    CInferenceManager_ORT inference_manager(in_place_init, model_path.string());
    std::cout << "This is a placeholder main function." << "\n";
    return 0;
}