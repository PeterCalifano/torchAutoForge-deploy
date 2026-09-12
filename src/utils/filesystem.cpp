/** @file filesystem.cpp
 * @brief Non-overwriting file copy and symbolic-link operations.
 */

#include "filesystem.h"
#include <set>

namespace ptafdeploy::utils
{
    void CopyOrLinkFiles(std::span<const std::filesystem::path> source_paths,
                         const std::filesystem::path& destination_directory, EFileCreationMode creation_mode)
    {
        namespace fs = std::filesystem;

        if (source_paths.empty())
            throw std::invalid_argument("Cannot copy or link an empty file list");
        if (creation_mode != EFileCreationMode::symlink && creation_mode != EFileCreationMode::copy)
            throw std::invalid_argument("Unknown file creation mode");
        if (fs::exists(destination_directory) &&
            (!fs::is_directory(destination_directory) || !fs::is_empty(destination_directory)))
            throw std::invalid_argument("File destination must be new or empty");

        // Validate the complete list before creating any destination entries.
        std::set<fs::path> source_filenames;
        for (const auto& source_path : source_paths)
        {
            if (!fs::is_regular_file(source_path))
                throw std::invalid_argument("Not a regular input: " + source_path.string());
            if (!source_filenames.insert(source_path.filename()).second)
                throw std::invalid_argument("Duplicate input basename: " +
                                            source_path.filename().string());
        }

        // Create each copy or absolute link without overwriting an existing entry.
        fs::create_directories(destination_directory);
        for (const auto& source_path : source_paths)
        {
            const auto destination_path = destination_directory / source_path.filename();
            if (creation_mode == EFileCreationMode::symlink)
                fs::create_symlink(fs::absolute(source_path), destination_path);
            else
                fs::copy_file(source_path, destination_path, fs::copy_options::none);
        }
    }
} // namespace ptafdeploy::utils
