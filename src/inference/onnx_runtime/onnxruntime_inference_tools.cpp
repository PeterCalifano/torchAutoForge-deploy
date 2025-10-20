#include "onnxruntime_inference_tools.hpp"

namespace deploy_ort
{

    // CInferenceManager_ORT constructors
    CInferenceManager_ORT::CInferenceManager_ORT(const bool inplace_init,
                                                 const std::string &model_path,
                                                 Ort::SessionOptions session_options)
        : model_path_(model_path), session_options_(std::move(session_options))
    {
        print_info("Initializing CInferenceManager_ORT with model path: " + model_path);
        // Define and validate attributes
        if (model_path.empty())
        {
            throw std::invalid_argument("Model path cannot be empty.");
        }

        [[maybe_unused]] bool file_exists = deploy_aux::CheckFileExistsWithExt(model_path, "onnx", true);
        model_path_ = fs::path(model_path);

        // Initialize manager if in-place
        if (inplace_init)
        {
            initialize<float>(); // How to do with the type? :/
        }
        else
        {
            print_info("CInferenceManager_ORT not initialized. Call initialize() method to set up the environment.");
        }
    }

    // From yml configuration file
    CInferenceManager_ORT::CInferenceManager_ORT(const std::string session_config_path)
    {
        // Check file exists
        bool config_exists = deploy_aux::CheckFileExistsWithExt(session_config_path, "yml", true);
        if (config_exists)
        {
            print_info("Session configuration file found: " + session_config_path);
            // Load session configuration from the file
            // TODO

            std::string model_path;

            // Initialize calling the other constructor
            CInferenceManager_ORT(true, model_path, Ort::SessionOptions());
        }
    }

};