/** @file inference_output.cpp
 * @brief RapidJSON implementation and bounded disk publication; no image dependency.
 */
#include "inference_output.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <stdexcept>

namespace ptafdeploy::utils::inference_output
{
    namespace fs = std::filesystem;
    namespace
    {
        using Writer =
            rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>,
                              rapidjson::CrtAllocator, rapidjson::kWriteValidateEncodingFlag>;
        void Write(Writer& writer, const SJsonValue& json)
        {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, std::nullptr_t>)
                        writer.Null();
                    else if constexpr (std::is_same_v<T, bool>)
                        writer.Bool(value);
                    else if constexpr (std::is_same_v<T, int64_t>)
                        writer.Int64(value);
                    else if constexpr (std::is_same_v<T, uint64_t>)
                        writer.Uint64(value);
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        if (!std::isfinite(value))
                            throw std::invalid_argument("JSON requires finite numbers");
                        writer.Double(value);
                    }
                    else if constexpr (std::is_same_v<T, std::string>)
                    {
                        if (value.size() > std::numeric_limits<rapidjson::SizeType>::max())
                            throw std::invalid_argument("JSON string exceeds writer capacity");
                        if (!writer.String(value.data(),
                                           static_cast<rapidjson::SizeType>(value.size())))
                            throw std::invalid_argument("JSON strings require valid UTF-8");
                    }
                    else if constexpr (std::is_same_v<T, SJsonValue::Array>)
                    {
                        writer.StartArray();
                        for (const auto& item : value)
                            Write(writer, item);
                        writer.EndArray();
                    }
                    else
                    {
                        writer.StartObject();
                        for (const auto& [key, item] : value)
                        {
                            Write(writer, SJsonValue(key));
                            Write(writer, item);
                        }
                        writer.EndObject();
                    }
                },
                json.value);
        }
        template <typename T>
        SJsonValue::Array Values(const inference::STensorView& tensor, size_t count)
        {
            SJsonValue::Array values;
            values.reserve(count);
            const auto* bytes = static_cast<const std::byte*>(tensor.data);
            for (size_t i = 0; i < count; ++i)
            {
                // Host views need not be aligned for their declared element type.
                T value;
                std::memcpy(&value, bytes + i * sizeof(T), sizeof(T));
                if constexpr (std::is_floating_point_v<T>)
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Tensor contains non-finite values");
                values.emplace_back(value);
            }
            return values;
        }
    } // namespace
    std::string Serialize(const SJsonValue& value)
    {
        rapidjson::StringBuffer buffer;
        Writer writer(buffer);
        Write(writer, value);
        return {buffer.GetString(), buffer.GetSize()};
    }
    SJsonValue TensorValue(const inference::STensorView& tensor)
    {
        inference::ValidateHostTensorView(tensor.descriptor, tensor, "JSON output");
        const auto count = inference::ComputeElementCount(tensor.descriptor.shape);
        SJsonValue::Array values;
        using enum inference::ETensorElementType;
        switch (tensor.descriptor.dtype)
        {
        case uint8:
            values = Values<uint8_t>(tensor, count);
            break;
        case int8:
            values = Values<int8_t>(tensor, count);
            break;
        case uint16:
            values = Values<uint16_t>(tensor, count);
            break;
        case int16:
            values = Values<int16_t>(tensor, count);
            break;
        case uint32:
            values = Values<uint32_t>(tensor, count);
            break;
        case int32:
            values = Values<int32_t>(tensor, count);
            break;
        case uint64:
            values = Values<uint64_t>(tensor, count);
            break;
        case int64:
            values = Values<int64_t>(tensor, count);
            break;
        case float32:
            values = Values<float>(tensor, count);
            break;
        case float64:
            values = Values<double>(tensor, count);
            break;
        case boolean: {
            const auto* bytes = static_cast<const unsigned char*>(tensor.data);
            values.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                if (bytes[i] > 1)
                    throw std::invalid_argument("Boolean tensor values must be 0 or 1");
                values.emplace_back(bytes[i] != 0);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported JSON tensor dtype: " +
                                        inference::ToString(tensor.descriptor.dtype));
        }
        SJsonValue::Array shape;
        for (auto extent : tensor.descriptor.shape)
            shape.emplace_back(extent);
        return SJsonValue::Object{{"name", tensor.descriptor.name},
                                  {"dtype", inference::ToString(tensor.descriptor.dtype)},
                                  {"shape", std::move(shape)},
                                  {"values", std::move(values)}};
    }
    SJsonValue TensorValue(const inference::SFloatTensor& tensor)
    {
        inference::STensorDescriptor descriptor;
        descriptor.name = tensor.name;
        descriptor.dtype = inference::ETensorElementType::float32;
        descriptor.shape = tensor.shape;
        return TensorValue(inference::STensorView{descriptor, tensor.values.data(),
                                                  tensor.values.size() * sizeof(float)});
    }
    SJsonValue FrameValue(const SFrameRecord& frame)
    {
        if (!std::isfinite(frame.inference_ms) || frame.inference_ms < 0)
            throw std::invalid_argument("Inference duration must be finite and nonnegative");
        auto fields = frame.fields;
        for (const char* key : {"index", "source", "inference_ms"})
            if (fields.contains(key))
                throw std::invalid_argument("Reserved frame field: " + std::string(key));
        fields.emplace("index", frame.index);
        fields.emplace("source", frame.source);
        fields.emplace("inference_ms", frame.inference_ms);
        return fields;
    }
    void PrepareOutput(const fs::path& output, const fs::path& input)
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

    CReport::CReport(const fs::path& root, SRunMetadata initial)
        : root_(root), metadata(std::move(initial))
    {
        if (!fs::is_directory(root_) || !fs::is_empty(root_))
            throw std::invalid_argument("Report requires an empty reserved directory");
        std::ofstream spool;
        spool.exceptions(std::ios::failbit | std::ios::badbit);
        spool.open(root_ / "frames.jsonl", std::ios::binary);
        spool.close();
        Publish(false);
    }
    void CReport::Append(const SFrameRecord& frame)
    {
        if (state_ != EState::active)
            throw std::logic_error("Cannot append after a failed write or completed report");
        const auto record = Serialize(FrameValue(frame));
        try
        {
            std::ofstream spool;
            spool.exceptions(std::ios::failbit | std::ios::badbit);
            spool.open(root_ / "frames.jsonl", std::ios::binary | std::ios::app);
            spool << record << '\n';
            spool.close();
            ++committed_frames_;
        }
        catch (...)
        {
            // A later append must not join a partial tail to a supposedly committed record.
            state_ = EState::append_failed;
            throw;
        }
    }
    void CReport::Publish(bool complete, const SJsonValue& error)
    {
        if (state_ == EState::complete || (complete && state_ == EState::append_failed))
            throw std::logic_error("Report is complete or contains a failed append");
        auto fields = metadata.fields;
        for (const char* key : {"schema_version", "status", "frames", "error"})
            if (fields.contains(key))
                throw std::invalid_argument("Reserved run field: " + std::string(key));
        fields.emplace("schema_version", metadata.schema_version);
        fields.emplace("status", complete ? "complete" : "incomplete");
        if (!std::holds_alternative<std::nullptr_t>(error.value))
            fields.emplace("error", error);
        auto prefix = Serialize(fields);
        prefix.pop_back();
        std::ifstream records(root_ / "frames.jsonl", std::ios::binary);
        if (!records)
            throw std::runtime_error("Cannot read frame spool: " + root_.string());
        std::ofstream report;
        report.exceptions(std::ios::failbit | std::ios::badbit);
        report.open(root_ / "predictions.json.tmp", std::ios::binary);
        report << prefix << ",\"frames\":[";
        // Ignore partial append tails; only successfully closed records are committed.
        for (size_t i = 0; i < committed_frames_; ++i)
        {
            std::string record;
            if (!std::getline(records, record) || records.eof())
                throw std::runtime_error("Incomplete frame spool: " + root_.string());
            if (i)
                report << ',';
            report << record;
        }
        report << "]}\n";
        report.close();
        fs::rename(root_ / "predictions.json.tmp", root_ / "predictions.json");
        if (complete)
        {
            state_ = EState::complete;
            std::error_code ignored;
            fs::remove(root_ / "frames.jsonl", ignored);
        }
    }
} // namespace ptafdeploy::utils::inference_output
