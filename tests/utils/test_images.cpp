/** @file test_images.cpp
 * @brief Image utility behavior independent of learned models.
 */
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utils/images/images.h>
namespace images = ptafdeploy::utils::images;

TEST_CASE("Image markers preserve other pixels and caller coordinates", "[utils][images]")
{
    cv::Mat image(32, 32, CV_16UC4, cv::Scalar(10, 20, 30, 40));
    const cv::Point2d point(16, 16);
    images::DrawCrosshair(image, point, cv::Scalar(65535, 65535, 65535, 65535), 9, 1);
    REQUIRE(image.at<cv::Vec<uint16_t, 4>>(0, 0) == cv::Vec<uint16_t, 4>(10, 20, 30, 40));
    REQUIRE(image.at<cv::Vec<uint16_t, 4>>(16, 16)[3] == 65535);
    REQUIRE(point == cv::Point2d(16, 16));
    images::DrawCrosshair(image, {-1, 8}, cv::Scalar(50, 60, 70, 80), 5, 3);
    REQUIRE(image.at<cv::Vec<uint16_t, 4>>(8, 0)[3] == 80);
    const auto before = image.clone();
    images::DrawCrosshair(image, {std::numeric_limits<double>::max(), 16}, cv::Scalar(0), 9, 1);
    REQUIRE(cv::norm(image, before, cv::NORM_INF) == 0);
    REQUIRE_THROWS_AS(images::DrawCrosshair(image, {0, 0}, cv::Scalar(0), 0, 1),
                      std::invalid_argument);
}

TEST_CASE("Image operations use explicit interpolation and natural ordering", "[utils][images]")
{
    const cv::Mat source(2, 3, CV_8UC1, cv::Scalar(42));
    const auto resized = images::Resize(source, {6, 4}, cv::INTER_NEAREST);
    REQUIRE(resized.size() == cv::Size(6, 4));
    REQUIRE(resized.at<uint8_t>(3, 5) == 42);
    REQUIRE(images::NaturalLess("frame2.png", "frame10.png"));
    REQUIRE(images::NaturalLess("frame02.png", "frame2.png"));
    REQUIRE(images::NaturalLess("f99999999999999999999", "f100000000000000000000"));
}
