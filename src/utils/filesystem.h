/**
 * @file filesystem.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief File-path validation and non-overwriting copy/link operations.
 * @version 0.1
 * @date 2025-07-20
 */

#pragma once

#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ptafdeploy::utils
{
    namespace fs = std::filesystem;

    /** @brief Create independent file copies or absolute symbolic links. */
    enum class EFileCreationMode
    {
        copy,
        symlink
    };

    /** @brief Copy or link files into a new or empty directory without altering their sources.
     * @param source_paths Regular files with distinct basenames.
     * @param destination_directory New or empty output directory.
     * @param creation_mode Copy file contents or create absolute symlinks; no fallback between modes.
     * @throws std::invalid_argument For empty inputs, duplicate basenames, or invalid destinations.
     * @throws fs::filesystem_error For filesystem failures.
     * @note Partial outputs are retained on failure. Concurrent writers are unsupported.
     */
    void CopyOrLinkFiles(std::span<const fs::path> source_paths, const fs::path& destination_directory,
                        EFileCreationMode creation_mode);

    /** @brief Accept types implicitly convertible to a filesystem path. */
    template <typename PathType>
    concept IsValidPathType = std::is_convertible_v<PathType, fs::path>;

    /**
     * @brief Check that a path exists and is not a directory.
     * @tparam PathType Path-convertible input type.
     * @param path Path to inspect.
     * @param throw_if_not_exists Throw instead of returning false for an absent path.
     * @return Whether the non-directory path exists.
     * @throws std::invalid_argument For a directory or a required missing path.
     * @throws fs::filesystem_error If the filesystem query fails.
     */
    template <IsValidPathType PathType>
    bool CheckFileExists(const PathType& path, const bool throw_if_not_exists = false)
    {
        fs::path file_path(path);
        if (fs::exists(file_path))
        {
            if (fs::is_directory(file_path))
            {
                throw std::invalid_argument("Path is a directory, not a file: " + file_path.string());
            }

#if (VERBOSE)
            if (fs::is_regular_file(file_path))
            {
                std::cout << "File path " << file_path << " exists and is a regular file.\n";
            }
            else
            {
                std::cout << "File path " << file_path
                          << " exists but is neither a regular file nor a directory.\n";
            }
#endif
            return true;
        }
        else
        {

#if (VERBOSE)
            if (!throw_if_not_exists)
                std::cout << "File path " << file_path << " does not exist\n";
#endif
            if (throw_if_not_exists)
            {
                throw std::invalid_argument("File does not exist: " + file_path.string());
            }
            return false;
        }
    }

    /**
     * @brief Check existence and a case-sensitive filename extension.
     * @tparam PathType Path-convertible input type.
     * @param path Path to inspect.
     * @param expected_extension Expected extension without its leading dot.
     * @param throw_if_not_exists Throw for an absent path or extension mismatch.
     * @return Whether the non-directory path exists with the expected extension.
     * @throws std::invalid_argument For a directory or a requested strict-check failure.
     * @throws fs::filesystem_error If the filesystem query fails.
     */
    template <IsValidPathType PathType>
    bool CheckFileExistsWithExt(const PathType& path, const std::string& expected_extension,
                                const bool throw_if_not_exists = false)
    {
        fs::path file_path(path);
        bool file_exists = CheckFileExists(path, throw_if_not_exists);
        if (file_exists)
        {
            if (file_path.extension() != ("." + expected_extension) && throw_if_not_exists)
            {
                throw std::invalid_argument("File does not have the expected extension: " +
                                            file_path.string());
            }
            else if (file_path.extension() != ("." + expected_extension))
            {
                std::cerr << "File exists but does not have the expected extension: " << file_path.string()
                          << "\n";
                return false;
            }
        }
        return file_exists;
    }
}; // namespace ptafdeploy::utils
