/** @file prepare_image_sequence.cpp
 * @brief Prepare a reproducible local image selection; model and dataset policies are external.
 */

#include <fstream>
#include <iostream>
#include <tclap/CmdLine.h>
#include <utils/filesystem.h>
#include <utils/images/images.h>
#include <utils/inference_output/inference_output.h>
#include <utils/sampling.h>

/** @brief Prepare inputs and write provenance; return nonzero for invalid arguments or IO errors.
 */
int main(int argc, char** argv)
{
    try
    {
        namespace images = ptafdeploy::utils::images;
        namespace inference_output = ptafdeploy::utils::inference_output;
        namespace filesystem = std::filesystem;
        using inference_output::SJsonValue;

        TCLAP::CmdLine command_line("Select images without modifying source datasets", ' ', "1");
        TCLAP::UnlabeledValueArg<std::string> input_path_option("input", "Image or image directory",
                                                                true, "", "path", command_line);
        TCLAP::UnlabeledValueArg<std::string> output_path_option(
            "destination", "New or empty selection directory", true, "", "path", command_line);
        TCLAP::ValueArg<int> max_frame_count_option("", "count", "Maximum image count", false, 100,
                                                    "count", command_line);
        TCLAP::ValueArg<std::string> storage_mode_option("", "mode", "symlink or copy", false,
                                                         "symlink", "mode", command_line);

        // Validate the requested selection and storage policy before touching the output.
        command_line.parse(argc, argv);
        if (storage_mode_option.getValue() != "symlink" && storage_mode_option.getValue() != "copy")
            throw std::invalid_argument("--mode must be symlink or copy");
        if (max_frame_count_option.getValue() <= 0)
            throw std::invalid_argument("--count must be positive");

        // Enumerate once; creating an output directory cannot change this input sequence.
        const auto ordered_image_paths = images::SelectFrames(input_path_option.getValue());
        const auto selected_indices = ptafdeploy::utils::SelectEvenlySpacedIndices(
            ordered_image_paths.size(), max_frame_count_option.getValue());
        const filesystem::path output_directory = output_path_option.getValue();
        inference_output::PrepareOutput(output_directory, input_path_option.getValue());

        // Inspect one selected image at a time; retain only paths and compact metadata.
        std::vector<filesystem::path> selected_source_paths;
        SJsonValue::Array selection_records;

        for (auto source_index : selected_indices)
        {
            const auto source_path = filesystem::absolute(ordered_image_paths[source_index]);
            selected_source_paths.push_back(source_path);
            const auto decoded_image = images::DecodeForOverlay(source_path);
            selection_records.emplace_back(SJsonValue::Object{
                {"source_index", source_index},
                {"source", source_path.string()},
                {"image_size",
                 SJsonValue::Object{{"width", decoded_image.cols}, {"height", decoded_image.rows}}},
                {"sample_bits", decoded_image.elemSize1() * 8},
                {"channels", decoded_image.channels()},
                {"selected",
                 (filesystem::path("images") / source_path.filename()).generic_string()}});
        }

        // Preserve source files and reject existing output before writing the manifest.
        ptafdeploy::utils::CopyOrLinkFiles(selected_source_paths, output_directory / "images",
                                           storage_mode_option.getValue() == "copy"
                                               ? ptafdeploy::utils::EFileCreationMode::copy
                                               : ptafdeploy::utils::EFileCreationMode::symlink);

        // Record the original selection order and decoded image properties.
        std::ofstream selection_manifest;
        selection_manifest.exceptions(std::ios::badbit | std::ios::failbit);
        selection_manifest.open(output_directory / "selection.json", std::ios::binary);
        selection_manifest << inference_output::Serialize(SJsonValue::Object{
                                  {"schema_version", 1},
                                  {"input",
                                   filesystem::absolute(input_path_option.getValue()).string()},
                                  {"ordering", "natural_filename"},
                                  {"selection", "evenly_spaced_endpoints"},
                                  {"available_count", ordered_image_paths.size()},
                                  {"mode", storage_mode_option.getValue()},
                                  {"files", std::move(selection_records)}})
                           << '\n';
        selection_manifest.close();

        std::cout << "selected=" << selected_source_paths.size()
                  << "\nimages=" << (output_directory / "images").string() << '\n';
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
    return 0;
}
