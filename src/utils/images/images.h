/** @file images.h
 * @brief Image operations and deterministic file selection, independent of model policy.
 */
#pragma once
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ptafdeploy::utils::images
{
    /** @brief Compare UTF-8 filenames numerically at digit runs, lexically for ties.
     * @param a First valid UTF-8 filename. @param b Second valid UTF-8 filename.
     * @return Whether a precedes b in the total ordering; digit runs cannot overflow.
     */
    [[nodiscard]] bool NaturalLess(const std::string& a, const std::string& b);
    /** @brief Select one regular file or supported images nonrecursively in natural order.
     * @param input Source file or directory.
     * @return Ordered paths without image data; PNG, JPEG, BMP, TIFF extensions are supported.
     * @throws std::exception For missing paths, empty image directories, or filesystem errors.
     */
    [[nodiscard]] std::vector<std::filesystem::path> SelectFrames(
        const std::filesystem::path& input);
    /** @brief Decode using explicit OpenCV flags.
     * @param source Image path. @param flags OpenCV imread flags selected by the caller.
     * @return Owned decoded image.
     * @throws std::runtime_error For empty decoding results. @throws cv::Exception On decoder
     * errors.
     */
    [[nodiscard]] cv::Mat Decode(const std::filesystem::path& source, int flags);
    /** @brief Decode original samples for annotation; honor JPEG EXIF orientation.
     * @param source Image path; JPEG is identified by its signature.
     * @return Owned image; non-JPEG inputs use IMREAD_UNCHANGED to preserve alpha/sample depth.
     * @throws std::exception On decoding or IO errors.
     */
    [[nodiscard]] cv::Mat DecodeForOverlay(const std::filesystem::path& source);
    /** @brief Convert colour representation into owned storage.
     * @param source Borrowed image. @param conversion Explicit OpenCV conversion code.
     * @return Converted image. @throws cv::Exception For invalid images or conversions.
     */
    [[nodiscard]] cv::Mat Convert(const cv::Mat& source, int conversion);
    /** @brief Resize using explicit interpolation.
     * @param source Borrowed image. @param size Positive destination extent.
     * @param interpolation OpenCV interpolation mode.
     * @return Resized image; a same-size result shares source storage.
     * @throws std::invalid_argument For empty input or invalid size/interpolation.
     * @throws cv::Exception On resize/allocation errors.
     */
    [[nodiscard]] cv::Mat Resize(const cv::Mat& source, cv::Size size, int interpolation);
    /** @brief Draw in-place with explicit style, clipping strokes without changing coordinates.
     * @param image Borrowed writable image.
     * @param position Finite pixel coordinate, possibly outside the image.
     * @param color Native sample-range/channel-order colour, including alpha when present.
     * @param size Positive marker span in pixels.
     * @param thickness Positive stroke thickness in pixels.
     * @throws std::invalid_argument For invalid extents, style, or non-finite coordinates.
     * @throws cv::Exception If the image or style cannot be drawn by OpenCV.
     */
    void DrawCrosshair(cv::Mat& image, cv::Point2d position, cv::Scalar color, int size,
                       int thickness);
    /** @brief Save uint8/uint16 samples as PNG without sample conversion.
     * @param destination Output path with .png extension; overwrite policy belongs to the caller.
     * @param image Borrowed nonempty image with one, three, or four channels.
     * @throws std::exception For unsupported sample/channel types, extension, or write errors.
     */
    void SavePng(const std::filesystem::path& destination, const cv::Mat& image);
} // namespace ptafdeploy::utils::images
