/** @file test_filesystem.cpp
 * @brief Sampling and copy/link behavior without image dependencies.
 */

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <limits>
#include <utils/filesystem.h>
#include <utils/sampling.h>

TEST_CASE("Index sampling and file copying preserve inputs and reject collisions",
          "[utils][filesystem]")
{
    REQUIRE(ptafdeploy::utils::SelectEvenlySpacedIndices(10, 4) == std::vector<size_t>{0, 3, 6, 9});
    REQUIRE(ptafdeploy::utils::SelectEvenlySpacedIndices(3, 100) == std::vector<size_t>{0, 1, 2});
    REQUIRE(ptafdeploy::utils::SelectEvenlySpacedIndices(10, 1) == std::vector<size_t>{0});
    REQUIRE(ptafdeploy::utils::SelectEvenlySpacedIndices(std::numeric_limits<size_t>::max(), 3)
                .back() == std::numeric_limits<size_t>::max() - 1);
    REQUIRE_THROWS_AS(ptafdeploy::utils::SelectEvenlySpacedIndices(0, 2), std::invalid_argument);

    // Use ordinary text files so the general utilities are tested without image libraries.
    namespace filesystem = std::filesystem;
    const auto test_directory =
        filesystem::temp_directory_path() /
        ("ptaf-select-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(filesystem::create_directory(test_directory));

    struct CTemporaryDirectoryCleanup
    {
        filesystem::path path;

        ~CTemporaryDirectoryCleanup()
        {
            std::error_code ignored;
            filesystem::remove_all(path, ignored);
        }
    } directory_cleanup{test_directory};

    const auto source = test_directory / "source.txt";
    const std::string content = "unchanged source contents";
    {
        std::ofstream file(source);
        file << content;
    }

    // Exercise both storage modes and verify the copied payload.
    const std::vector<filesystem::path> sources{source};
    ptafdeploy::utils::CopyOrLinkFiles(sources, test_directory / "linked",
                                       ptafdeploy::utils::EFileCreationMode::symlink);
    ptafdeploy::utils::CopyOrLinkFiles(sources, test_directory / "copied",
                                       ptafdeploy::utils::EFileCreationMode::copy);

    REQUIRE(filesystem::is_symlink(test_directory / "linked/source.txt"));
    REQUIRE_FALSE(filesystem::is_symlink(test_directory / "copied/source.txt"));

    std::ifstream copied(test_directory / "copied/source.txt");
    const std::string contents{std::istreambuf_iterator<char>(copied), {}};
    REQUIRE(contents == content);

    REQUIRE_THROWS_AS(
        ptafdeploy::utils::CopyOrLinkFiles(sources, test_directory / "copied",
                                           ptafdeploy::utils::EFileCreationMode::copy),
        std::invalid_argument);

    REQUIRE_THROWS_AS(
        ptafdeploy::utils::CopyOrLinkFiles(std::vector<filesystem::path>{source, source},
                                           test_directory / "duplicates",
                                           ptafdeploy::utils::EFileCreationMode::copy),
        std::invalid_argument);
}
