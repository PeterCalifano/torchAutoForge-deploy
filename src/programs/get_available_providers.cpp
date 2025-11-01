#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <iostream>

int main()
{
    // Check providers
    std::ignore = deploy_ort::CInferenceManager_ORT::GetAvailableProviders();
    return 0;
}