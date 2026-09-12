/**
 * @file centroiding_io.h
 * @brief Demo-owned sequence selection, overlays, and bounded JSON publication.
 */
#pragma once
#include "centroiding_support.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <inference/model_facade.h>
#include <opencv2/imgcodecs.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <string>
#include <utility>

namespace ptafdeploy::examples::centroiding_models
{
    namespace fs = std::filesystem;
    using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

    /**
     * @brief Compare UTF-8 filenames naturally, then lexically for equal natural keys.
     * @param a First valid UTF-8 filename.
     * @param b Second valid UTF-8 filename.
     * @return Whether a precedes b in the documented total ordering.
     */
    inline bool NaturalLess(const std::string& a, const std::string& b)
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
    inline std::vector<fs::path> SelectFrames(const fs::path& input)
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

    /**
     * @brief Reject collisions and output ancestors before writing any result.
     * @param output New or empty result directory, created on success.
     * @param input Source location, resolved through symlinks for comparison.
     * @throws std::invalid_argument For collisions or an output ancestor of input.
     * @throws fs::filesystem_error If inspection or directory creation fails.
     */
    inline void PrepareOutput(const fs::path& output, const fs::path& input)
    {
        const auto source = fs::weakly_canonical(input);
        const auto destination = fs::weakly_canonical(output);
        auto relative = source.lexically_relative(destination);
        if (relative.empty() || *relative.begin() != "..")
            throw std::invalid_argument("Output must not equal or contain the input location");
        if (fs::exists(destination) &&
            (!fs::is_directory(destination) || !fs::is_empty(destination)))
            throw std::invalid_argument("Output must be a new or empty directory: " +
                                        output.string());
        fs::create_directories(destination);
    }

    /**
     * @brief Write a named UTF-8 JSON string.
     * @param w Active JSON object writer.
     * @param key Member name.
     * @param value Valid UTF-8 member value.
     */
    inline void String(JsonWriter& w, const char* key, const std::string& value)
    {
        w.Key(key);
        w.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
    }
    /**
     * @brief Write a named coordinate pair without rounding or clamping.
     * @param w Active JSON object writer.
     * @param key Member name.
     * @param p Finite coordinates from DecodeCentroid.
     */
    inline void Point(JsonWriter& w, const char* key, const SCoordinate& p)
    {
        w.Key(key);
        w.StartObject();
        w.Key("x");
        w.Double(p.x);
        w.Key("y");
        w.Double(p.y);
        w.EndObject();
    }
    /**
     * @brief Return whether a coordinate belongs to the half-open image bounds.
     * @param p Decoded centroid.
     * @param size Original positive image extent.
     * @return True for 0 <= x < width and 0 <= y < height.
     */
    inline bool Inside(const SCentroidResult& p, cv::Size size)
    {
        return p.original_image_pixels.x >= 0 && p.original_image_pixels.y >= 0 &&
               p.original_image_pixels.x < size.width && p.original_image_pixels.y < size.height;
    }
    /**
     * @brief Serialize one frame using the common schema.
     * @param index Zero-based sequence index.
     * @param source Relative source filename.
     * @param size Original image extent.
     * @param output Validated finite [1,2] output tensor.
     * @param result Coordinates decoded from output.
     * @param duration Non-negative inference duration in milliseconds.
     * @param overlay Relative overlay path, or empty for JSON null.
     * @return One complete JSON frame object.
     */
    inline std::string FrameJson(size_t index, const fs::path& source, cv::Size size,
                                 const infer::SFloatTensor& output, const SCentroidResult& result,
                                 double duration, const std::string& overlay)
    {
        rapidjson::StringBuffer buffer;
        JsonWriter w(buffer);
        w.StartObject();
        w.Key("index");
        w.Uint64(index);
        String(w, "source", source.generic_string());
        w.Key("image_size");
        w.StartObject();
        w.Key("width");
        w.Int(size.width);
        w.Key("height");
        w.Int(size.height);
        w.EndObject();
        w.Key("raw_output");
        w.StartObject();
        String(w, "name", output.name);
        w.Key("shape");
        w.StartArray();
        for (auto v : output.shape)
            w.Int64(v);
        w.EndArray();
        w.Key("values");
        w.StartArray();
        for (auto v : output.values)
            w.Double(v);
        w.EndArray();
        w.EndObject();
        w.Key("centroid");
        w.StartObject();
        Point(w, "normalized", result.normalized);
        Point(w, "model_pixels", result.model_input_pixels);
        Point(w, "image_pixels", result.original_image_pixels);
        w.Key("inside_image");
        w.Bool(Inside(result, size));
        w.EndObject();
        w.Key("inference_ms");
        w.Double(duration);
        w.Key("overlay");
        if (overlay.empty())
            w.Null();
        else
            w.String(overlay.c_str());
        w.EndObject();
        return buffer.GetString();
    }

    /**
     * @brief Serialize tensor metadata without copying the contract's tensor arrays.
     * @param w Active JSON object writer.
     * @param key Array member name.
     * @param tensors Ordered tensor metadata owned by the contract.
     */
    inline void TensorMetadata(JsonWriter& w, const char* key,
                               const std::vector<infer::STensorInfo>& tensors)
    {
        w.Key(key);
        w.StartArray();
        for (const auto& tensor : tensors)
        {
            w.StartObject();
            String(w, "name", tensor.name);
            String(w, "dtype", tensor.dtype);
            w.Key("shape");
            w.StartArray();
            for (auto dimension : tensor.shape)
                w.Int64(dimension);
            w.EndArray();
            w.EndObject();
        }
        w.EndArray();
    }

    /**
     * @brief Serialize run metadata; model remains null until loading succeeds.
     * @param input Supplied source location.
     * @param count Number of selected frames.
     * @param model Borrowed loaded facade, or null before loading.
     * @param requested_model Supplied manifest or artifact path.
     * @return JSON metadata without status or frame records.
     * @throws std::exception If metadata or source inspection fails.
     */
    inline std::string MetadataJson(const fs::path& input, size_t count,
                                    const infer::CModelFacade* model = nullptr,
                                    const fs::path& requested_model = {})
    {
        rapidjson::StringBuffer buffer;
        JsonWriter w(buffer);
        w.StartObject();
        w.Key("schema_version");
        w.Int(1);
        String(w, "requested_model_path", requested_model.string());
        w.Key("input");
        w.StartObject();
        String(w, "path", input.string());
        String(w, "kind", fs::is_directory(input) ? "directory" : "image");
        String(w, "ordering", "natural_filename");
        w.Key("selected_frame_count");
        w.Uint64(count);
        w.EndObject();
        w.Key("model");
        if (!model)
            w.Null();
        else
        {
            const auto c = model->GetContract();
            w.StartObject();
            String(w, "artifact_path", c.artifact_path);
            String(w, "config_path", c.config_path);
            String(w, "role", c.role);
            String(w, "backend_detail", c.backend_detail);
            String(w, "preprocessing", c.preprocessing);
            String(w, "postprocessing", c.postprocessing);
            w.Key("runtime");
            w.StartObject();
            w.Key("device_id");
            w.Int(c.runtime.device_id);
            w.Key("intra_op_num_threads");
            w.Int(c.runtime.intra_op_num_threads);
            w.Key("inter_op_num_threads");
            w.Int(c.runtime.inter_op_num_threads);
            w.Key("allow_fallback");
            w.Bool(c.runtime.allow_fallback);
            w.Key("backend");
            w.Int(static_cast<int>(c.runtime.backend));
            w.Key("artifact");
            w.Int(static_cast<int>(c.runtime.artifact));
            w.Key("execution_target_priority");
            w.StartArray();
            for (auto t : c.runtime.execution_target_priority)
                w.Int(static_cast<int>(t));
            w.EndArray();
            w.Key("enable_profiling");
            w.Bool(c.runtime.enable_profiling);
            String(w, "log_id", c.runtime.log_id);
            w.Key("tensorrt_optimization_profile_index");
            w.Int(c.runtime.tensorrt_optimization_profile_index);
            w.EndObject();
            TensorMetadata(w, "inputs", c.inputs);
            TensorMetadata(w, "outputs", c.outputs);
            w.EndObject();
        }
        w.Key("preprocessing");
        w.StartObject();
        String(w, "library", "OpenCV " CV_VERSION);
        String(w, "grayscale", "IMREAD_GRAYSCALE uint8");
        String(w, "resize", "bilinear");
        w.Key("scale");
        w.Double(1.0 / 255.0);
        w.EndObject();
        w.EndObject();
        return buffer.GetString();
    }

    /**
     * @brief Own an on-disk frame spool and atomically publish complete or incomplete reports.
     * @details Records remain on disk if publication fails. Abrupt termination recovery is
     * excluded.
     */
    class CReport
    {
        fs::path root_;
        size_t committed_frames_{};

      public:
        /** @brief Serialized run metadata, replaced after successful model loading. */
        std::string metadata;
        /**
         * @brief Open a report in a previously reserved empty output directory.
         * @param root Directory owned exclusively by this run.
         * @param initial JSON object returned by MetadataJson.
         * @throws std::exception If the initial spool/report cannot be published.
         */
        CReport(const fs::path& root, std::string initial)
            : root_(root), metadata(std::move(initial))
        {
            std::ofstream spool;
            spool.exceptions(std::ios::failbit | std::ios::badbit);
            spool.open(root_ / "frames.jsonl", std::ios::binary);
            spool.close();
            Publish(false);
        }
        /** @brief Keep exactly one owner for the run's output paths and record count. */
        CReport(const CReport&) = delete;
        /** @brief Prevent two report instances from sharing a run's output paths. */
        CReport& operator=(const CReport&) = delete;

        /**
         * @brief Flush a completed frame; a failed write does not commit its record.
         * @param record Complete object returned by FrameJson.
         * @throws std::ios_base::failure If opening, writing, or closing fails.
         * @note Flushing is not a power-loss durability guarantee.
         */
        void Append(const std::string& record)
        {
            std::ofstream spool;
            spool.exceptions(std::ios::failbit | std::ios::badbit);
            spool.open(root_ / "frames.jsonl", std::ios::binary | std::ios::app);
            spool << record << '\n';
            spool.close();
            ++committed_frames_;
        }
        /**
         * @brief Assemble and replace predictions.json, retaining records on failure.
         * @param complete Whether every selected frame has finished successfully.
         * @param error Complete JSON error object, or empty when absent.
         * @throws std::exception If reading, writing, or atomic replacement fails.
         */
        void Publish(bool complete, const std::string& error = {})
        {
            std::ifstream records(root_ / "frames.jsonl", std::ios::binary);
            if (!records)
                throw std::runtime_error("Cannot read frame spool: " + root_.string());
            std::ofstream report;
            report.exceptions(std::ios::failbit | std::ios::badbit);
            report.open(root_ / "predictions.json.tmp", std::ios::binary);
            report << metadata.substr(0, metadata.size() - 1) << ",\"status\":\""
                   << (complete ? "complete" : "incomplete") << "\",\"frames\":[";
            // Ignore a partial trailing append: only successfully closed writes are committed.
            for (size_t index = 0; index < committed_frames_; ++index)
            {
                std::string record;
                if (!std::getline(records, record) || records.eof())
                    throw std::runtime_error("Incomplete frame spool: " + root_.string());
                if (index != 0)
                    report << ',';
                report << record;
            }
            report << ']';
            if (!error.empty())
                report << ",\"error\":" << error;
            report << "}\n";
            report.close();
            fs::rename(root_ / "predictions.json.tmp", root_ / "predictions.json");
            if (complete)
            {
                // Cleanup failure does not invalidate a successfully published report.
                std::error_code ignored;
                fs::remove(root_ / "frames.jsonl", ignored);
            }
        }
    };

    /**
     * @brief Draw clipped crosshair strokes, preserving original pixels elsewhere.
     * @param source Current image path; source bytes remain unchanged.
     * @param destination Reserved PNG output path.
     * @param result Decoded original-image coordinates.
     * @param expected_size Inference image extent, or zero to omit the extent check.
     * @throws std::runtime_error For unsupported samples, dimensions, or IO failures.
     * @throws cv::Exception If image processing fails.
     */
    inline void SaveOverlay(const fs::path& source, const fs::path& destination,
                            const SCentroidResult& result, cv::Size expected_size = {})
    {
        // Detect JPEG by its signature, including explicit files with nonstandard extensions.
        std::ifstream header(source, std::ios::binary);
        unsigned char signature[2]{};
        header.read(reinterpret_cast<char*>(signature), sizeof(signature));
        const bool jpeg = signature[0] == 0xff && signature[1] == 0xd8;
        // JPEG has no alpha channel; these flags match inference EXIF orientation handling.
        const int flags = jpeg ? cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR : cv::IMREAD_UNCHANGED;
        cv::Mat overlay = cv::imread(source.string(), flags);
        if (overlay.empty())
            throw std::runtime_error("Cannot decode overlay: " + source.string());
        if (expected_size.width > 0 && expected_size.height > 0 && overlay.size() != expected_size)
            throw std::runtime_error("Overlay and inference image extents differ: " +
                                     source.string());
        if (overlay.depth() != CV_8U && overlay.depth() != CV_16U)
            throw std::runtime_error("PNG overlays require uint8 or uint16 images: " +
                                     source.string());
        if (overlay.channels() == 1)
            cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);
        const double x = result.original_image_pixels.x, y = result.original_image_pixels.y;
        // Reject only drawing outside the visible margin; the recorded coordinate is untouched.
        if (x >= -10 && y >= -10 && x < overlay.cols + 10.0 && y < overlay.rows + 10.0)
        {
            const cv::Point center{static_cast<int>(std::round(x)),
                                   static_cast<int>(std::round(y))};
            const double white = overlay.depth() == CV_16U ? 65535.0 : 255.0;
            cv::drawMarker(overlay, center, cv::Scalar(0, 0, 0, white), cv::MARKER_CROSS, 19, 3);
            cv::drawMarker(overlay, center, cv::Scalar(white, white, white, white),
                           cv::MARKER_CROSS, 17, 1);
        }
        if (!cv::imwrite(destination.string(), overlay))
            throw std::runtime_error("Cannot write overlay: " + destination.string());
    }
} // namespace ptafdeploy::examples::centroiding_models
