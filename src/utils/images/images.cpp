/** @file images.cpp
 * @brief OpenCV image utilities and natural filename ordering.
 */
#include "images.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace ptafdeploy::utils::images
{
    namespace fs = std::filesystem;
    bool NaturalLess(const std::string& a, const std::string& b)
    {
        size_t i = 0, j = 0;
        auto digit = [](char c) { return c >= '0' && c <= '9'; };
        while (i < a.size() && j < b.size())
        {
            if (digit(a[i]) && digit(b[j]))
            {
                size_t ae = i, be = j;
                while (ae < a.size() && digit(a[ae]))
                    ++ae;
                while (be < b.size() && digit(b[be]))
                    ++be;
                while (i < ae && a[i] == '0')
                    ++i;
                while (j < be && b[j] == '0')
                    ++j;
                if (ae - i != be - j)
                    return ae - i < be - j;
                const int comparison = a.compare(i, ae - i, b, j, be - j);
                if (comparison != 0)
                    return comparison < 0;
                i = ae;
                j = be;
            }
            else
            {
                if (a[i] != b[j])
                    return static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[j]);
                ++i;
                ++j;
            }
        }
        if (i != a.size() || j != b.size())
            return i == a.size();
        return a < b;
    }

    /**
     * @brief Select regular image files once; reject absent or empty inputs.
     * @param input One image or a non-recursive image directory.
     * @return Ordered source paths; no image pixels are retained.
     * @throws std::invalid_argument For absent inputs or empty selections.
     * @throws fs::filesystem_error If directory inspection fails.
     */
    std::vector<fs::path> SelectFrames(const fs::path& input)
    {
        if (fs::is_regular_file(input))
            return {input};
        if (!fs::is_directory(input))
            throw std::invalid_argument("Missing input: " + input.string());
        std::vector<fs::path> frames;
        for (const auto& entry : fs::directory_iterator(input))
        {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : char(c);
            });
            if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                                            ext == ".bmp" || ext == ".tif" || ext == ".tiff"))
                frames.push_back(entry.path());
        }
        std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
            return NaturalLess(a.filename().string(), b.filename().string());
        });
        if (frames.empty())
            throw std::invalid_argument("No supported images: " + input.string());
        return frames;
    }

    cv::Mat Decode(const fs::path& source, int flags)
    {
        auto image = cv::imread(source.string(), flags);
        if (image.empty())
            throw std::runtime_error("Cannot decode image: " + source.string());
        return image;
    }
    cv::Mat DecodeForOverlay(const fs::path& source)
    {
        std::ifstream header(source, std::ios::binary);
        unsigned char signature[2]{};
        header.read(reinterpret_cast<char*>(signature), sizeof(signature));
        const bool jpeg = signature[0] == 0xff && signature[1] == 0xd8;
        return Decode(source,
                      jpeg ? cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR : cv::IMREAD_UNCHANGED);
    }
    cv::Mat Convert(const cv::Mat& source, int conversion)
    {
        cv::Mat result;
        cv::cvtColor(source, result, conversion);
        return result;
    }
    cv::Mat Resize(const cv::Mat& source, cv::Size size, int interpolation)
    {
        if (source.empty() || size.width <= 0 || size.height <= 0)
            throw std::invalid_argument("Resize requires a nonempty image and positive extent");
        if (interpolation < cv::INTER_NEAREST || interpolation > cv::INTER_NEAREST_EXACT)
            throw std::invalid_argument("Unsupported resize interpolation");
        if (source.size() == size)
            return source;
        cv::Mat result;
        cv::resize(source, result, size, 0.0, 0.0, interpolation);
        return result;
    }
    void DrawCrosshair(cv::Mat& image, cv::Point2d position, cv::Scalar color, int size,
                       int thickness)
    {
        if (image.empty() || size <= 0 || thickness <= 0 || !std::isfinite(position.x) ||
            !std::isfinite(position.y))
            throw std::invalid_argument(
                "Crosshair requires an image, finite position, and positive style");
        // Draw clipped line segments in double precision before converting to integer pixels.
        // This also supports finite predictions far beyond the integer coordinate range.
        const double x = std::round(position.x), y = std::round(position.y);
        const double radius = size / 2;
        const double margin = thickness;
        if (image.cols + margin > std::numeric_limits<int>::max() ||
            image.rows + margin > std::numeric_limits<int>::max())
            throw std::invalid_argument(
                "Image extent plus stroke exceeds drawing coordinate range");
        const auto segment = [&](double x0, double y0, double x1, double y1) {
            if (x1 < -margin || y1 < -margin || x0 > image.cols - 1.0 + margin ||
                y0 > image.rows - 1.0 + margin)
                return;
            const cv::Point begin{
                static_cast<int>(std::clamp(x0, -margin, image.cols - 1.0 + margin)),
                static_cast<int>(std::clamp(y0, -margin, image.rows - 1.0 + margin))};
            const cv::Point end{
                static_cast<int>(std::clamp(x1, -margin, image.cols - 1.0 + margin)),
                static_cast<int>(std::clamp(y1, -margin, image.rows - 1.0 + margin))};
            cv::line(image, begin, end, color, thickness);
        };
        segment(x - radius, y, x + radius, y);
        segment(x, y - radius, x, y + radius);
    }
    void SavePng(const fs::path& destination, const cv::Mat& image)
    {
        if (image.empty() || (image.depth() != CV_8U && image.depth() != CV_16U) ||
            (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
            throw std::invalid_argument(
                "PNG requires uint8/uint16 images with 1, 3, or 4 channels");
        if (destination.extension() != ".png")
            throw std::invalid_argument("PNG output requires .png extension");
        if (!cv::imwrite(destination.string(), image))
            throw std::runtime_error("Cannot write PNG: " + destination.string());
    }
} // namespace ptafdeploy::utils::images
