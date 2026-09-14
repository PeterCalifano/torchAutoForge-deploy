/** @file report_reader.cpp
 * @brief Strict JSON/report input with private RapidJSON parsing.
 */

#include "inference_output.h"
#include <cmath>
#include <fstream>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/document.h>
#include <stdexcept>

namespace ptafdeploy::utils::inference_output
{
    namespace
    {
        SJsonValue ConvertParsedJsonValue(const rapidjson::Value& value)
        {
            if (value.IsNull())
                return {};
            if (value.IsBool())
                return value.GetBool();

            // Preserve integer values exactly instead of converting them through double.
            if (value.IsInt64())
                return value.GetInt64();
            if (value.IsUint64())
                return value.GetUint64();
            if (value.IsDouble())
            {
                if (!std::isfinite(value.GetDouble()))
                    throw std::invalid_argument("Non-finite JSON number");
                return value.GetDouble();
            }

            if (value.IsString())
                return std::string(value.GetString(), value.GetStringLength());

            // Recursively copy containers into values independent of the parser lifetime.
            if (value.IsArray())
            {
                SJsonValue::Array array_values;
                array_values.reserve(value.Size());
                for (const auto& array_element : value.GetArray())
                    array_values.push_back(ConvertParsedJsonValue(array_element));
                return array_values;
            }

            // Reject duplicate keys rather than silently retaining one value.
            SJsonValue::Object object_fields;
            for (auto member = value.MemberBegin(); member != value.MemberEnd(); ++member)
            {
                std::string member_name(member->name.GetString(), member->name.GetStringLength());
                if (!object_fields.emplace(member_name, ConvertParsedJsonValue(member->value))
                         .second)
                    throw std::invalid_argument("Duplicate JSON key: " + member_name);
            }
            return object_fields;
        }

        uint64_t ReadNonnegativeInteger(const SJsonValue& value)
        {
            if (const auto* numeric_value = std::get_if<uint64_t>(&value.value))
                return *numeric_value;
            if (const auto* numeric_value = std::get_if<int64_t>(&value.value);
                numeric_value && *numeric_value >= 0)
                return static_cast<uint64_t>(*numeric_value);
            throw std::invalid_argument("Expected nonnegative integer");
        }

        double ReadNumericDuration(const SJsonValue& value)
        {
            if (const auto* numeric_value = std::get_if<double>(&value.value))
                return *numeric_value;
            if (const auto* numeric_value = std::get_if<int64_t>(&value.value))
                return static_cast<double>(*numeric_value);
            if (const auto* numeric_value = std::get_if<uint64_t>(&value.value))
                return static_cast<double>(*numeric_value);
            throw std::invalid_argument("Expected numeric duration");
        }
    } // namespace

    SJsonValue ReadJsonFile(const std::filesystem::path& json_path)
    {
        std::ifstream input_stream(json_path, std::ios::binary);
        if (!input_stream)
            throw std::runtime_error("Cannot read JSON: " + json_path.string());

        // Validate encoding and parse precision before exposing any report fields.
        rapidjson::IStreamWrapper json_stream(input_stream);
        rapidjson::Document parsed_document;
        parsed_document.ParseStream<rapidjson::kParseValidateEncodingFlag |
                                    rapidjson::kParseFullPrecisionFlag>(json_stream);
        if (input_stream.bad())
            throw std::runtime_error("JSON read failed: " + json_path.string());
        if (parsed_document.HasParseError())
            throw std::invalid_argument(
                "Invalid JSON " + json_path.string() + ": " +
                rapidjson::GetParseError_En(parsed_document.GetParseError()) + " at byte " +
                std::to_string(parsed_document.GetErrorOffset()));

        return ConvertParsedJsonValue(parsed_document);
    }

    SInferenceReport ReadInferenceReport(const std::filesystem::path& json_path)
    {
        try
        {
            // Check the version and completion marker before decoding frame records.
            auto report_json = ReadJsonFile(json_path);
            auto report_fields = std::get<SJsonValue::Object>(std::move(report_json.value));
            if (ReadNonnegativeInteger(report_fields.at("schema_version")) != 1)
                throw std::invalid_argument("Unsupported report schema_version");
            const auto completion_status = std::get<std::string>(report_fields.at("status").value);
            if (completion_status != "complete" && completion_status != "incomplete")
                throw std::invalid_argument("Invalid report status");

            // Retain application-specific fields while validating the shared frame envelope.
            SInferenceReport report;
            report.complete = completion_status == "complete";
            auto frame_values =
                std::get<SJsonValue::Array>(std::move(report_fields.at("frames").value));
            for (auto& frame : frame_values)
            {
                auto frame_fields = std::get<SJsonValue::Object>(std::move(frame.value));
                const auto frame_index = ReadNonnegativeInteger(frame_fields.at("index"));
                if (frame_index != report.frames.size())
                    throw std::invalid_argument("Frame indices must be consecutive from zero");

                SFrameRecord frame_record{static_cast<size_t>(frame_index),
                                          std::get<std::string>(frame_fields.at("source").value),
                                          ReadNumericDuration(frame_fields.at("inference_ms")),
                                          {}};
                if (frame_record.source.empty() || !std::isfinite(frame_record.inference_ms) ||
                    frame_record.inference_ms < 0)
                    throw std::invalid_argument("Invalid source or duration in frame " +
                                                std::to_string(frame_index));

                // Move remaining fields without imposing a model-specific schema.
                frame_fields.erase("index");
                frame_fields.erase("source");
                frame_fields.erase("inference_ms");
                frame_record.fields = std::move(frame_fields);
                report.frames.push_back(std::move(frame_record));
            }

            // Preserve error context for callers evaluating an incomplete run.
            if (auto member = report_fields.find("error"); member != report_fields.end())
                report.error = std::move(member->second);
            for (const char* member_name : {"schema_version", "status", "frames", "error"})
                report_fields.erase(member_name);

            report.metadata.fields = std::move(report_fields);
            return report;
        }
        catch (const std::exception& error)
        {
            throw std::invalid_argument("Cannot decode report " + json_path.string() + ": " +
                                        error.what());
        }
    }
} // namespace ptafdeploy::utils::inference_output
