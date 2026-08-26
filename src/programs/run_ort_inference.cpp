/**
 * @file run_ort_inference.cpp
 * @brief Command-line ONNX model metadata probe through the generic facade.
 */

#include <filesystem>
#include <inference/inference_manager.h>
#include <iostream>
#include <utils/logging/CLogger.h>

namespace
{
    namespace logging = ptafdeploy::logging;

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("run_ort_inference", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }
} // namespace

/**
 * @brief Load an ONNX artifact and print its generic metadata summary.
 * @return Zero on success, nonzero for usage or model-load errors.
 */
int main(int argc, char** argv)
{
    GetLogger().info("ONNX Runtime inference metadata probe");
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << "\n";
        return 1;
    }

    try
    {
        const std::filesystem::path model_path = argv[1];
        GetLogger().debug("Loading ONNX artifact: ", model_path.string());
        ptafdeploy::inference::CInferenceManager inference_manager(model_path);
        const ptafdeploy::inference::SModelMetadata& metadata =
            inference_manager.GetModelMetadata();

        std::cout << "Inputs: " << metadata.inputs.size() << "\n";
        std::cout << "Outputs: " << metadata.outputs.size() << "\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        GetLogger().error(error.what());
        return 2;
    }
}
