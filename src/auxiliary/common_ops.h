/**
 * @file common_ops.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief File-path validation helpers used by inference callers.
 * @version 0.1
 * @date 2025-07-20
 */
#pragma once

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fs = std::filesystem;

namespace deploy_aux
{
    /** @brief Accept types implicitly convertible to a filesystem path. */
    template <typename T>
    concept IsValidPathType = std::is_convertible_v<T, fs::path>;

    /**
     * @brief Check that a path exists and is not a directory.
     * @tparam T Path-convertible input type.
     * @param path Path to inspect.
     * @param throw_if_not_exists Throw instead of returning false for an absent path.
     * @return Whether the non-directory path exists.
     * @throws std::invalid_argument For a directory or a required missing path.
     * @throws fs::filesystem_error If the filesystem query fails.
     */
    template <IsValidPathType T>
    bool CheckFileExists(const T &path,
                         const bool throw_if_not_exists = false)
    {
        // Define path object from string
        fs::path p(path);
        if (fs::exists(p))
        {
            if (fs::is_directory(p))
            {
                throw std::invalid_argument("Path is a directory, not a file: " + p.string());
            }

#if (VERBOSE)
                if (fs::is_regular_file(p))
            {
                std::cout << "File path " << p << " exists and is a regular file.\n";
            }
            else
            {
                std::cout << "File path " << p << " exists but is neither a regular file nor a directory.\n";
            }
#endif
            return true;
        }
        else
        {

#if (VERBOSE)
    if (!throw_if_not_exists)
        std::cout << "File path " << p << " does not exist\n";
#endif
            if (throw_if_not_exists)
            {
                throw std::invalid_argument("File does not exist: " + p.string());
            }
            return false;
        }
    }

    /**
     * @brief Check existence and a case-sensitive filename extension.
     * @tparam T Path-convertible input type.
     * @param path Path to inspect.
     * @param ext Expected extension without its leading dot.
     * @param throw_if_not_exists Throw for an absent path or extension mismatch.
     * @return Whether the non-directory path exists with the expected extension.
     * @throws std::invalid_argument For a directory or a requested strict-check failure.
     * @throws fs::filesystem_error If the filesystem query fails.
     */
    template <IsValidPathType T>
    bool CheckFileExistsWithExt(const T &path,
                                const std::string &ext,
                                const bool throw_if_not_exists = false)
    {
        // Check if the file exists and has the specified extension
        fs::path p(path);
        bool file_exists = CheckFileExists(path, throw_if_not_exists);
        if (file_exists)
        {
            if (p.extension() != ("." + ext) && throw_if_not_exists)
            {
                throw std::invalid_argument("File does not have the expected extension: " + p.string());
            }
            else if (p.extension() != ("." + ext))
            {
                std::cerr << "File exists but does not have the expected extension: " << p.string() << "\n";
                return false;
            }
        }
        return file_exists;
    }
};
