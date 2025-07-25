/**
 * @file common_ops.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-20
 */
#pragma once

#include <exception>
#include <execution>  // Optional, for parallel policies
#include <filesystem> // C++17, for filesystem operations
#include <iostream>
#include <numeric> // for std::reduce
#include <vector>

namespace fs = std::filesystem;

namespace deploy_aux
{
#if (PARALLELEXEC)
    /**
     * @brief Computes the product of elements in a vector in parallel.
     *
     * @tparam T The type of elements in the vector.
     * @param v The input vector.
     * @return T The product of the elements.
     */
    template <typename T>
    T AccumProduct(const std::vector<T> &v)
    {

        return std::reduce(
            std::execution::par,
            v.begin(), v.end(),
            static_cast<T>(1),
            std::multiplies<T>());
    }
#else

    /**
     * @brief Computes the product of elements in a vector serially.
     *
     * @tparam T The type of elements in the vector.
     * @param v The input vector.
     * @return T The product of the elements.
     */
    template <typename T>
    T AccumProduct(const std::vector<T> &v)
    {
        // serial:
        return std::reduce(
            v.begin(), v.end(),
            static_cast<T>(1),
            std::multiplies<T>());
    }
#endif

    template <typename T>
    concept IsValidPathType = std::is_convertible_v<T, fs::path>;

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
