/** @file testFilesystem.cpp
 * @brief Preserve observable filesystem-helper behavior after utility consolidation.
 */
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <utils/filesystem.h>
namespace fs = std::filesystem;
TEST_CASE("Filesystem checks distinguish files directories and extension mismatches", "[utils]")
{
    const auto root = fs::temp_directory_path() /
                      ("ptaf-filesystem-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(root));
    struct Cleanup
    {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    } cleanup{root};
    const auto file = root / "model.onnx";
    {
        std::ofstream stream(file);
        stream << "fixture";
    }
    REQUIRE(ptafdeploy::utils::CheckFileExists(file));
    REQUIRE(ptafdeploy::utils::CheckFileExistsWithExt(file, "onnx"));
    REQUIRE_FALSE(ptafdeploy::utils::CheckFileExistsWithExt(file, "ONNX"));
    REQUIRE_FALSE(ptafdeploy::utils::CheckFileExists(root / "missing"));
    REQUIRE_THROWS_AS(ptafdeploy::utils::CheckFileExists(root), std::invalid_argument);
    REQUIRE_THROWS_AS(ptafdeploy::utils::CheckFileExists(root / "missing", true),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ptafdeploy::utils::CheckFileExistsWithExt(file, "engine", true),
                      std::invalid_argument);
}
