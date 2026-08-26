/**
 * @file testTaskAdapters.cpp
 * @brief Target-owned generic inference task-adapter tests.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <inference/inference_matlab_adapters.h>
#include <inference/task_adapters.h>
#include <inference/task_value_types.h>

#include <limits>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    using Catch::Matchers::ContainsSubstring;
} // namespace

TEST_CASE("task_value_types_convert_center_size_and_corner_boxes", "[inference][task_adapters]")
{
    const infer::SBoundingBox2D center_size =
        infer::MakeBoundingBox2DFromCenterSize(10.0F, 20.0F, 4.0F, 6.0F);

    REQUIRE(center_size.center.x == Catch::Approx(10.0F));
    REQUIRE(center_size.center.y == Catch::Approx(20.0F));
    REQUIRE(center_size.size.width == Catch::Approx(4.0F));
    REQUIRE(center_size.size.height == Catch::Approx(6.0F));
    REQUIRE(infer::GetBoundingBox2DXyxy(center_size) ==
            std::vector<float>{8.0F, 17.0F, 12.0F, 23.0F});

    const infer::SBoundingBox2D corners = infer::MakeBoundingBox2DFromXyxy(2.0F, 4.0F, 8.0F, 10.0F);
    REQUIRE(corners.center.x == Catch::Approx(5.0F));
    REQUIRE(corners.center.y == Catch::Approx(7.0F));
    REQUIRE(corners.size.width == Catch::Approx(6.0F));
    REQUIRE(corners.size.height == Catch::Approx(6.0F));
}

TEST_CASE("task_value_types_reject_invalid_boxes", "[inference][task_adapters]")
{
    REQUIRE_THROWS_WITH(infer::MakeBoundingBox2DFromCenterSize(0.0F, 0.0F, -1.0F, 1.0F),
                        ContainsSubstring("non-negative"));
    REQUIRE_THROWS_WITH(infer::MakeBoundingBox2DFromXyxy(4.0F, 0.0F, 3.0F, 1.0F),
                        ContainsSubstring("ordered"));
    REQUIRE_THROWS_WITH(infer::MakeBoundingBox2DFromCenterSize(
                            std::numeric_limits<float>::quiet_NaN(), 0.0F, 1.0F, 1.0F),
                        ContainsSubstring("finite"));

    const infer::SBoundingBox2D zero_area =
        infer::MakeBoundingBox2DFromXyxy(2.0F, 3.0F, 2.0F, 3.0F);
    REQUIRE(zero_area.size.width == Catch::Approx(0.0F));
    REQUIRE(zero_area.size.height == Catch::Approx(0.0F));

    const float maximum = std::numeric_limits<float>::max();
    const infer::SBoundingBox2D maximum_point =
        infer::MakeBoundingBox2DFromXyxy(maximum, maximum, maximum, maximum);
    REQUIRE(maximum_point.center.x == Catch::Approx(maximum));
    REQUIRE(maximum_point.center.y == Catch::Approx(maximum));
}

TEST_CASE("task_adapters_prepare_nchw_image_tensors", "[inference][task_adapters]")
{
    const infer::SFloatTensor tensor = infer::MakeNchwFloatTensorFromHwcFloat(
        "images", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, 1, 2, 3, 0.5F, true);

    REQUIRE(tensor.name == "images");
    REQUIRE(tensor.shape == std::vector<int64_t>{1, 3, 1, 2});
    REQUIRE(tensor.values == std::vector<float>{1.5F, 3.0F, 1.0F, 2.5F, 0.5F, 2.0F});
    REQUIRE_THROWS_WITH(infer::MakeNchwFloatTensorFromHwcFloat("bad", {1.0F}, 1, 1, 3, 1.0F, false),
                        ContainsSubstring("HWC image value count"));
}

TEST_CASE("task_adapters_decode_feature_rows_on_configured_axis", "[inference][task_adapters]")
{
    infer::SFeatureRowSchema schema;
    schema.attribute_axis = 1;
    schema.x_index = 0;
    schema.y_index = 1;
    schema.score_index = 2;

    const infer::SFloatTensor output{
        "features", {1, 3, 2}, {10.0F, 20.0F, 11.0F, 21.0F, 0.5F, 0.75F}};
    const std::vector<infer::SFeature2D> features = infer::DecodeFeatureRows(output, schema);

    REQUIRE(features.size() == 2);
    REQUIRE(features[0].position.x == Catch::Approx(10.0F));
    REQUIRE(features[0].position.y == Catch::Approx(11.0F));
    REQUIRE(features[0].score == Catch::Approx(0.5F));
    REQUIRE(features[1].position.x == Catch::Approx(20.0F));
    REQUIRE(features[1].position.y == Catch::Approx(21.0F));
    REQUIRE(features[1].score == Catch::Approx(0.75F));

    schema.attribute_axis = -1;
    schema.score_index = -1;
    const std::vector<infer::SFeature2D> unscored = infer::DecodeFeatureRows(
        infer::SFloatTensor{"features", {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}}, schema);
    REQUIRE(unscored.size() == 2);
    REQUIRE(unscored[1].position.x == Catch::Approx(3.0F));
    REQUIRE(unscored[1].position.y == Catch::Approx(4.0F));
    REQUIRE(unscored[1].score == Catch::Approx(1.0F));
}

TEST_CASE("task_adapters_validate_feature_row_contracts", "[inference][task_adapters]")
{
    infer::SFeatureRowSchema schema;
    schema.x_index = 0;
    schema.y_index = 1;

    REQUIRE_THROWS_WITH(
        infer::DecodeFeatureRows(infer::SFloatTensor{"bad", {2, 2}, {1.0F, 2.0F}}, schema),
        ContainsSubstring("value count"));

    schema.attribute_axis = 2;
    REQUIRE_THROWS_WITH(
        infer::DecodeFeatureRows(infer::SFloatTensor{"bad", {1, 2}, {1.0F, 2.0F}}, schema),
        ContainsSubstring("attribute axis"));

    schema.attribute_axis = -1;
    schema.y_index = 2;
    REQUIRE_THROWS_WITH(
        infer::DecodeFeatureRows(infer::SFloatTensor{"bad", {1, 2}, {1.0F, 2.0F}}, schema),
        ContainsSubstring("schema index"));
}

TEST_CASE("task_adapters_decode_center_size_detections", "[inference][task_adapters]")
{
    infer::SDetectionRowSchema schema;
    schema.attribute_axis = -1;
    schema.box_encoding = infer::EBoundingBoxEncoding::center_xywh;
    schema.box_coordinate_0_index = 0;
    schema.box_coordinate_1_index = 1;
    schema.box_coordinate_2_index = 2;
    schema.box_coordinate_3_index = 3;
    schema.objectness_index = 4;
    schema.first_class_score_index = 5;
    schema.class_score_count = 2;

    const infer::SFloatTensor output{"detections",
                                     {1, 3, 7},
                                     {
                                         10.0F, 20.0F, 30.0F, 40.0F, 0.9F, 0.1F, 0.8F,
                                         1.0F,  2.0F,  3.0F,  4.0F,  0.5F, 0.9F, 0.1F,
                                         7.0F,  8.0F,  9.0F,  10.0F, 0.8F, 0.7F, 0.1F,
                                     }};
    const std::vector<infer::SDetection2D> detections =
        infer::DecodeDetectionRows(output, schema, 0.5F, 2);

    REQUIRE(detections.size() == 2);
    REQUIRE(detections[0].bounds.center.x == Catch::Approx(10.0F));
    REQUIRE(detections[0].bounds.size.width == Catch::Approx(30.0F));
    REQUIRE(detections[0].classification.score == Catch::Approx(0.72F));
    REQUIRE(detections[0].classification.class_id == 1);
    REQUIRE(detections[1].bounds.center.x == Catch::Approx(7.0F));
    REQUIRE(detections[1].classification.score == Catch::Approx(0.56F));
    REQUIRE(detections[1].classification.class_id == 0);
}

TEST_CASE("task_adapters_decode_corner_detections_without_objectness", "[inference][task_adapters]")
{
    infer::SDetectionRowSchema schema;
    schema.box_encoding = infer::EBoundingBoxEncoding::corners_xyxy;
    schema.box_coordinate_0_index = 0;
    schema.box_coordinate_1_index = 1;
    schema.box_coordinate_2_index = 2;
    schema.box_coordinate_3_index = 3;
    schema.objectness_index = -1;
    schema.first_class_score_index = 4;
    schema.class_score_count = 2;

    const std::vector<infer::SDetection2D> detections = infer::DecodeDetectionRows(
        infer::SFloatTensor{"detections", {1, 6}, {2.0F, 4.0F, 8.0F, 10.0F, 0.6F, 0.4F}}, schema,
        0.6F, 0);

    REQUIRE(detections.size() == 1);
    REQUIRE(detections[0].bounds.center.x == Catch::Approx(5.0F));
    REQUIRE(detections[0].bounds.center.y == Catch::Approx(7.0F));
    REQUIRE(detections[0].classification.score == Catch::Approx(0.6F));
    REQUIRE(detections[0].classification.class_id == 0);
}

TEST_CASE("task_adapters_validate_detection_contracts", "[inference][task_adapters]")
{
    infer::SDetectionRowSchema schema;
    schema.box_coordinate_0_index = 0;
    schema.box_coordinate_1_index = 1;
    schema.box_coordinate_2_index = 2;
    schema.box_coordinate_3_index = 3;
    schema.objectness_index = 4;
    schema.first_class_score_index = 5;
    schema.class_score_count = 1;

    const infer::SFloatTensor output{"detections", {1, 6}, {1.0F, 2.0F, 3.0F, 4.0F, 0.8F, 0.9F}};
    REQUIRE_THROWS_WITH(
        infer::DecodeDetectionRows(output, schema, std::numeric_limits<float>::quiet_NaN(), 0),
        ContainsSubstring("finite"));

    schema.class_score_count = 0;
    REQUIRE_THROWS_WITH(infer::DecodeDetectionRows(output, schema, 0.1F, 0),
                        ContainsSubstring("class score"));

    schema.class_score_count = 1;
    schema.first_class_score_index = 4;
    REQUIRE_THROWS_WITH(infer::DecodeDetectionRows(output, schema, 0.1F, 0),
                        ContainsSubstring("overlap"));

    schema.first_class_score_index = 5;
    const infer::SFloatTensor invalid_filtered_box{
        "detections", {1, 6}, {1.0F, 2.0F, -3.0F, 4.0F, 0.1F, 0.1F}};
    REQUIRE_THROWS_WITH(infer::DecodeDetectionRows(invalid_filtered_box, schema, 0.5F, 0),
                        ContainsSubstring("non-negative"));
}

TEST_CASE("matlab_adapters_serialize_generic_detection_rows", "[inference][task_adapters][matlab]")
{
    infer::SDetectionRowSchema schema;
    schema.objectness_index = 4;
    schema.first_class_score_index = 5;
    schema.class_score_count = 2;

    const gtsam::Matrix detections = infer::DecodeDetectionRowsMatrix(
        infer::SFloatTensor{"detections", {1, 7}, {10.0F, 20.0F, 30.0F, 40.0F, 0.5F, 0.2F, 0.8F}},
        schema, 0.1, 0);

    REQUIRE(detections.rows() == 1);
    REQUIRE(detections.cols() == 6);
    REQUIRE(detections(0, 0) == Catch::Approx(10.0));
    REQUIRE(detections(0, 1) == Catch::Approx(20.0));
    REQUIRE(detections(0, 2) == Catch::Approx(30.0));
    REQUIRE(detections(0, 3) == Catch::Approx(40.0));
    REQUIRE(detections(0, 4) == Catch::Approx(0.4));
    REQUIRE(detections(0, 5) == Catch::Approx(1.0));
}
