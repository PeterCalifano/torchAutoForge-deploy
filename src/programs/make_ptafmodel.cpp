/**
 * @file make_ptafmodel.cpp
 * @brief Create and validate JSON model manifests without loading inference sessions.
 */
#include <inference/inference_config_parsing.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <tclap/CmdLine.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;

    /** @brief Publish a new file exclusively; remove an incomplete write owned by this call. */
    void WriteNewManifest(const fs::path& path, const std::string& json)
    {
        FILE* stream = std::fopen(path.string().c_str(), "wx");
        if (!stream)
            throw std::runtime_error("Cannot create new manifest: " + path.string());

        const bool written = std::fwrite(json.data(), 1, json.size(), stream) == json.size();
        const bool closed = std::fclose(stream) == 0;
        if (!written || !closed)
        {
            std::error_code cleanup_error;
            fs::remove(path, cleanup_error);
            throw std::runtime_error("Failed writing manifest: " + path.string());
        }
    }
} // namespace

/**
 * @brief Run manifest tooling without creating inference sessions.
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments owned by the calling process.
 * @return Zero on success, nonzero on invalid input or IO failure.
 */
int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || std::string(argv[1]) == "--help")
        {
            std::cout << "make-ptafmodel init ARTIFACT --task TASK --output MANIFEST [options]\n"
                         "make-ptafmodel validate MANIFEST [--check-artifact]\n"
                         "make-ptafmodel schema\n";
            return argc < 2 ? 1 : 0;
        }
        const std::string operation = argv[1];
        if (operation == "schema")
        {
            if (argc != 2)
                throw std::invalid_argument("schema accepts no arguments");
            std::cout << infer::GetPtafModelSchemaJson();
            return 0;
        }
        if (operation != "init" && operation != "validate")
            throw std::invalid_argument("Expected init, validate, or schema");

        TCLAP::CmdLine command("Create or validate a PTAF JSON manifest", ' ',
                               PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);
        TCLAP::UnlabeledValueArg<std::string> input("input", "Artifact or manifest path", true, "",
                                                    "PATH", command);
        TCLAP::ValueArg<std::string> task("", "task", "Model task", false, "", "TASK", command);
        TCLAP::ValueArg<std::string> output("", "output", "New manifest path", false, "", "PATH",
                                            command);
        TCLAP::ValueArg<std::string> targets("", "targets", "Execution target list", false, "cpu",
                                             "cpu,cuda,tensorrt", command);
        TCLAP::ValueArg<std::string> backend("", "backend", "Backend override", false, "auto",
                                             "BACKEND", command);
        TCLAP::ValueArg<std::string> preprocessing(
            "", "preprocessing", "Declarative preprocessing label", false, "", "LABEL", command);
        TCLAP::ValueArg<std::string> postprocessing(
            "", "postprocessing", "Declarative postprocessing label", false, "", "LABEL", command);
        TCLAP::ValueArg<int> device("", "device", "Device index", false, 0, "N", command);
        TCLAP::ValueArg<int> intra_threads("", "intra-op-threads", "Intra-operation threads", false,
                                           1, "N", command);
        TCLAP::ValueArg<int> inter_threads("", "inter-op-threads", "Inter-operation threads", false,
                                           1, "N", command);
        TCLAP::SwitchArg fallback("", "allow-fallback", "Allow automatic runtime fallback", command,
                                  false);
        TCLAP::SwitchArg check_artifact("", "check-artifact",
                                        "Check artifact existence and extension", command, false);
        command.parse(argc - 1, argv + 1);

        if (operation == "validate")
        {
            if (task.isSet() || output.isSet() || targets.isSet() || backend.isSet() ||
                preprocessing.isSet() || postprocessing.isSet() || device.isSet() ||
                intra_threads.isSet() || inter_threads.isSet() || fallback.isSet())
                throw std::invalid_argument("Creation options are not accepted by validate");
            infer::ValidatePtafModelConfig(input.getValue(), check_artifact.getValue());
            std::cout << "Valid manifest: " << input.getValue() << '\n';
            return 0;
        }
        if (!task.isSet() || !output.isSet() || check_artifact.isSet())
            throw std::invalid_argument(
                "init requires --task and --output; --check-artifact is for validate");

        const auto destination = fs::absolute(output.getValue());
        if (destination.extension() != ".ptafmodel")
            throw std::invalid_argument("Manifest output must have .ptafmodel extension");
        const auto artifact = fs::absolute(input.getValue());
        const auto extension = artifact.extension().string();
        if (extension != ".onnx" && extension != ".engine" && extension != ".plan")
            throw std::invalid_argument("Unsupported artifact extension: " + extension);
        std::error_code path_error;
        auto relative_artifact = fs::relative(artifact, destination.parent_path(), path_error);
        if (path_error || relative_artifact.empty())
            relative_artifact = artifact;

        // Build typed JSON, then validate it through the same parser used by model loading.
        rapidjson::Document document(rapidjson::kObjectType);
        auto& allocator = document.GetAllocator();
        const auto add_string = [&](const char* key, const std::string& value)
        {
            document.AddMember(rapidjson::Value(key, allocator),
                               rapidjson::Value(value.c_str(), allocator), allocator);
        };
        document.AddMember("schema_version", 1, allocator);
        add_string("artifact_path", relative_artifact.generic_string());
        add_string("task", task.getValue());
        document.AddMember("allow_fallback", fallback.getValue(), allocator);
        rapidjson::Value priority(rapidjson::kArrayType);
        for (auto target : infer::ParseExecutionTargetPriority(targets.getValue()))
            priority.PushBack(rapidjson::Value(infer::ToString(target).c_str(), allocator),
                              allocator);
        document.AddMember("execution_target_priority", priority, allocator);
        if (backend.isSet())
            add_string("backend", backend.getValue());
        if (preprocessing.isSet())
            add_string("preprocessing", preprocessing.getValue());
        if (postprocessing.isSet())
            add_string("postprocessing", postprocessing.getValue());
        if (device.isSet())
            document.AddMember("device_id", device.getValue(), allocator);
        if (intra_threads.isSet())
            document.AddMember("intra_op_num_threads", intra_threads.getValue(), allocator);
        if (inter_threads.isSet())
            document.AddMember("inter_op_num_threads", inter_threads.getValue(), allocator);

        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        const std::string json = std::string(buffer.GetString()) + '\n';
        infer::ValidatePtafModelJson(json, destination.string());

        // Exclusive creation protects an existing manifest, including concurrent creation.
        WriteNewManifest(destination, json);
        std::cout << "Created manifest: " << destination.string() << '\n';
        return 0;
    }
    catch (const TCLAP::ExitException& error)
    {
        return error.getExitStatus();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
