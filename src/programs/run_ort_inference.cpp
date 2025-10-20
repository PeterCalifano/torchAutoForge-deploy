#include <filesystem>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <iostream>

// TODO add tclap for argument parsing

using deploy_ort::CInferenceManager_ORT;

int main()
{
    std::cout << "This is a placeholder main function." << "\n";

    std::filesystem::path model_path = "/home/peterc/devDir/ML-repos/torchAutoForge-deploy/tests/.test_samples/zealous-cow-471_epoch_2893_0.onnx";
    bool in_place_init = true;

    CInferenceManager_ORT inference_manager(in_place_init, model_path.string());

    // TODO add inference calls

    // TODO any cleanup needed?
    return 0;
}