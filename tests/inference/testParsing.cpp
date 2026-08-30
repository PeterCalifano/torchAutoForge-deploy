/**
 * @file testParsing.cpp
 * @brief Target-owned tests for reusable deployment value and tensor parsing.
 */

#include <inference/inference_tensor_parsing.h>
#include <utils/value_parsing.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace parsing = ptafdeploy::parsing;
} // namespace

TEST_CASE("value_parsing_splits_optional_names", "[parsing][value]")
{
    const parsing::SNamedValue named =
        parsing::ParseNamedValue("image=tensors/input=0.f32", "--input");
    REQUIRE(named.name == "image");
    REQUIRE(named.value == "tensors/input=0.f32");

    const parsing::SNamedValue positional =
        parsing::ParseNamedValue("tensors/input.f32", "--input");
    REQUIRE(positional.name.empty());
    REQUIRE(positional.value == "tensors/input.f32");
}

TEST_CASE("value_parsing_rejects_malformed_optional_names", "[parsing][value]")
{
    REQUIRE_THROWS_WITH(parsing::ParseNamedValue("", "--input"),
                        Catch::Matchers::ContainsSubstring("--input"));
    REQUIRE_THROWS_AS(
        parsing::ParseNamedValue("=tensor.f32", "--input"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseNamedValue("image=", "--input"),
        std::invalid_argument);
}

TEST_CASE("value_parsing_reads_strict_integer_lists", "[parsing][value]")
{
    REQUIRE(parsing::ParseIntegerList(" -2, 0, +17 ", ',', "dimensions") ==
            std::vector<int64_t>{-2, 0, 17});

    REQUIRE_THROWS_AS(
        parsing::ParseIntegerList("1,,3", ',', "dimensions"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseIntegerList("1,2x", ',', "dimensions"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseIntegerList("1,2", '\0', "dimensions"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseIntegerList(std::to_string(std::numeric_limits<int64_t>::max()) + "0", ',',
                                  "dimensions"),
        std::invalid_argument);
}

TEST_CASE("value_parsing_reads_only_finite_floats", "[parsing][value]")
{
    REQUIRE(parsing::ParseFiniteFloat(" -1.25e2 ", "--fill") == Catch::Approx(-125.0F));
    REQUIRE(parsing::ParseFiniteFloat("+0.25", "--fill") == Catch::Approx(0.25F));
    REQUIRE(parsing::ParseFiniteFloat("0x1p2", "--fill") == Catch::Approx(4.0F));
    REQUIRE(parsing::ParseFiniteFloat("-0X1p1", "--fill") == Catch::Approx(-2.0F));

    REQUIRE_THROWS_AS(
        parsing::ParseFiniteFloat("1.0suffix", "--fill"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseFiniteFloat("1e1000", "--fill"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseFiniteFloat("nan", "--fill"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        parsing::ParseFiniteFloat("inf", "--fill"),
        std::invalid_argument);
}

TEST_CASE("tensor_shape_parsing_requires_positive_dimensions", "[parsing][inference]")
{
    REQUIRE(infer::ParseTensorShape("1, 3,224,224", "--shape") ==
            std::vector<int64_t>{1, 3, 224, 224});

    REQUIRE_THROWS_WITH(infer::ParseTensorShape("1,0,224", "--shape"),
                        Catch::Matchers::ContainsSubstring("--shape"));
    REQUIRE_THROWS_AS(
        infer::ParseTensorShape("1,-1,224", "--shape"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        infer::ParseTensorShape("1,224,", "--shape"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        infer::ParseTensorShape("", "--shape"),
        std::invalid_argument);
}

TEST_CASE("tensor_value_resolution_follows_model_input_order", "[parsing][inference]")
{
    const std::vector<infer::STensorInfo> inputs{
        {"image", "float32", {1, 3, 224, 224}, "host"},
        {"state", "float32", {1, 4}, "host"},
    };
    const std::vector<std::string> specifications{
        "state=state.f32",
        "image=image.f32",
    };

    const std::vector<parsing::SNamedValue> resolved =
        infer::ResolveNamedTensorValues(specifications, inputs, "--input");

    REQUIRE(resolved.size() == 2U);
    REQUIRE(resolved[0].name == "image");
    REQUIRE(resolved[0].value == "image.f32");
    REQUIRE(resolved[1].name == "state");
    REQUIRE(resolved[1].value == "state.f32");
}

TEST_CASE("tensor_value_resolution_enforces_unambiguous_inputs", "[parsing][inference]")
{
    const std::vector<infer::STensorInfo> one_input{
        {"image", "float32", {1, 3, 224, 224}, "host"},
    };
    const std::vector<infer::STensorInfo> two_inputs{
        {"image", "float32", {1, 3, 224, 224}, "host"},
        {"state", "float32", {1, 4}, "host"},
    };

    const std::vector<std::string> positional_specifications{"image.f32"};
    const std::vector<parsing::SNamedValue> positional =
        infer::ResolveNamedTensorValues(positional_specifications, one_input, "--input");
    const std::vector<parsing::SNamedValue> expected_positional{
        {"image", "image.f32"},
    };
    REQUIRE(positional == expected_positional);

    const std::vector<std::string> ambiguous_specifications{"image.f32"};
    REQUIRE_THROWS_AS(
        infer::ResolveNamedTensorValues(ambiguous_specifications, two_inputs, "--input"),
        std::invalid_argument);

    const std::vector<std::string> unknown_specifications{"other=other.f32"};
    REQUIRE_THROWS_AS(
        infer::ResolveNamedTensorValues(unknown_specifications, two_inputs, "--input"),
        std::invalid_argument);

    const std::vector<std::string> duplicate_specifications{
        "image=first.f32",
        "image=second.f32",
    };
    REQUIRE_THROWS_AS(
        infer::ResolveNamedTensorValues(duplicate_specifications, two_inputs, "--input"),
        std::invalid_argument);

    const std::vector<infer::STensorInfo> duplicate_names{
        {"image", "float32", {1}, "host"},
        {"image", "float32", {1}, "host"},
    };
    REQUIRE_THROWS_AS(infer::ResolveNamedTensorValues({}, duplicate_names, "--input"),
                      std::invalid_argument);

    const std::vector<infer::STensorInfo> empty_name{
        {"", "float32", {1}, "host"},
    };
    REQUIRE_THROWS_AS(infer::ResolveNamedTensorValues({}, empty_name, "--input"),
                      std::invalid_argument);
}
