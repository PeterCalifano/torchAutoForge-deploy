#include "onnxruntime_inference_tools.hpp"

namespace deploy_ort
{

    // CInferenceManager_ORT
    CInferenceManager_ORT::CInferenceManager_ORT(const std::string &model_path, const bool inplace_init) : model_path_(model_path)
    {
        print_info("Initializing CInferenceManager_ORT with model path: " + model_path);



        // Initialize manager if in-place
        if (inplace_init)
        {
            initialize();
        }
        else
        {
            print_info("CInferenceManager_ORT not initialized. Call initialize() method to set up the environment.");
        }
    }

    void CInferenceManager_ORT::initialize()
    {
        // Initialize ORT environment
        Ort::Env exec_env(ORT_LOGGING_LEVEL, "ONNXModel");
    }
};