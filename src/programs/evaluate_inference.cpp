/** @file evaluate_inference.cpp
 * @brief Summarize versioned inference reports and optional source-matched point references.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tclap/CmdLine.h>
#include <tclap/UnlabeledMultiArg.h>
#include <utils/inference_output/inference_output.h>
#include <utils/metrics/metrics.h>

namespace
{
    namespace inference_output = ptafdeploy::utils::inference_output;
    namespace metrics = ptafdeploy::utils::metrics;
    using inference_output::SJsonValue;

    SJsonValue MakeStatisticsJson(const metrics::SStatistics& statistics)
    {
        return SJsonValue::Object{{"count", statistics.count},     {"mean", statistics.mean},
                                  {"median", statistics.median},   {"p95", statistics.p95},
                                  {"minimum", statistics.minimum}, {"maximum", statistics.maximum}};
    }

    double ReadJsonNumber(const SJsonValue& json_value)
    {
        if (const auto* numeric_value = std::get_if<double>(&json_value.value))
            return *numeric_value;
        if (const auto* numeric_value = std::get_if<int64_t>(&json_value.value))
            return static_cast<double>(*numeric_value);
        if (const auto* numeric_value = std::get_if<uint64_t>(&json_value.value))
            return static_cast<double>(*numeric_value);
        throw std::invalid_argument("Point coordinate must be numeric");
    }

    float ReadPointCoordinateAsFloat(const SJsonValue& value)
    {
        // Check range before narrowing JSON coordinates to the shared float point type.
        const double number = ReadJsonNumber(value);
        if (std::abs(number) > std::numeric_limits<float>::max())
            throw std::invalid_argument("Point coordinate exceeds SPoint2D float range");
        return static_cast<float>(number);
    }

    const SJsonValue::Object& FindPointFieldsByPath(const SJsonValue::Object& fields,
                                                    const std::string& field_path)
    {
        // Follow the configured object path without assuming a particular model output.
        const auto* current_object = &fields;
        size_t component_begin = 0;

        while (component_begin < field_path.size())
        {
            const auto component_end = field_path.find('.', component_begin);
            const auto component_name =
                field_path.substr(component_begin, component_end == std::string::npos
                                                       ? component_end
                                                       : component_end - component_begin);
            if (component_name.empty())
                throw std::invalid_argument("Empty point-field component");
            current_object =
                &std::get<SJsonValue::Object>(current_object->at(component_name).value);
            if (component_end == std::string::npos)
                return *current_object;
            component_begin = component_end + 1;
        }
        throw std::invalid_argument("Point-field must identify an object with x and y");
    }

    SJsonValue MakeStringArrayJson(const std::vector<std::string>& strings)
    {
        SJsonValue::Array result;
        for (const auto& value : strings)
            result.emplace_back(value);
        return result;
    }
} // namespace

/** @brief Write summaries without overwriting results; return 2 for incomplete input reports. */
int main(int argc, char** argv)
{
    try
    {
        TCLAP::CmdLine command_line("Evaluate inference reports independently of model execution",
                                    ' ', "1");
        TCLAP::ValueArg<std::string> output_path_option("", "output", "New summary JSON path", true,
                                                        "", "path", command_line);
        TCLAP::ValueArg<int> discard_count_option("", "discard",
                                                  "Initial frames excluded from latency statistics",
                                                  false, 5, "count", command_line);
        TCLAP::ValueArg<std::string> reference_path_option("", "references",
                                                           "Version-one source/x/y reference JSON",
                                                           false, "", "path", command_line);
        TCLAP::ValueArg<std::string> point_field_option("", "point-field",
                                                        "Dot-separated prediction object with x/y",
                                                        false, "", "field", command_line);
        TCLAP::ValueArg<std::string> units_option("", "units",
                                                  "Verified common prediction/reference units",
                                                  false, "", "units", command_line);
        TCLAP::ValueArg<std::string> coordinate_convention_option(
            "", "convention", "Verified common coordinate convention", false, "", "convention",
            command_line);
        TCLAP::UnlabeledMultiArg<std::string> report_paths_option(
            "reports", "Version-one inference reports", true, "paths", command_line);

        // Reject invalid options and collisions before reading reports or creating files.
        command_line.parse(argc, argv);
        if (discard_count_option.getValue() < 0)
            throw std::invalid_argument("--discard must be nonnegative");

        namespace filesystem = std::filesystem;
        const filesystem::path summary_path = output_path_option.getValue();
        if (filesystem::exists(summary_path) || filesystem::is_symlink(summary_path))
            throw std::invalid_argument("Summary destination exists");

        // Coordinate semantics are declared explicitly; matching never assumes a model role.
        std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>> references;
        if (!reference_path_option.getValue().empty())
        {
            if (point_field_option.getValue().empty() || units_option.getValue().empty() ||
                coordinate_convention_option.getValue().empty())
                throw std::invalid_argument(
                    "References require --point-field, --units and --convention");

            // Check the declared reference convention before accepting its coordinates.
            const auto reference_json =
                inference_output::ReadJsonFile(reference_path_option.getValue());
            const auto& reference_fields = std::get<SJsonValue::Object>(reference_json.value);
            if (ReadJsonNumber(reference_fields.at("schema_version")) != 1 ||
                std::get<std::string>(reference_fields.at("units").value) !=
                    units_option.getValue() ||
                std::get<std::string>(reference_fields.at("coordinate_convention").value) !=
                    coordinate_convention_option.getValue())
                throw std::invalid_argument("Reference version, units, or convention mismatch");

            for (const auto& entry :
                 std::get<SJsonValue::Array>(reference_fields.at("points").value))
            {
                const auto& point_fields = std::get<SJsonValue::Object>(entry.value);
                references.push_back({std::get<std::string>(point_fields.at("source").value),
                                      ptafdeploy::inference::SPoint2D{
                                          ReadPointCoordinateAsFloat(point_fields.at("x")),
                                          ReadPointCoordinateAsFloat(point_fields.at("y"))}});
            }

            // Validate even when reports contain no predictions.
            (void)metrics::ComputeMatchedPointErrors({}, references);
        }
        else if (!point_field_option.getValue().empty() || !units_option.getValue().empty() ||
                 !coordinate_convention_option.getValue().empty())
            throw std::invalid_argument("Point metric options require --references");

        SJsonValue::Array run_summaries;
        bool complete = true;
        for (const auto& report_path : report_paths_option.getValue())
        {
            const auto report = inference_output::ReadInferenceReport(report_path);
            complete = complete && report.complete;

            std::vector<double> latency_samples;
            for (size_t frame_index = discard_count_option.getValue();
                 frame_index < report.frames.size(); ++frame_index)
                latency_samples.push_back(report.frames[frame_index].inference_ms);

            // Keep each run separate so first-frame effects and failures remain visible.
            SJsonValue::Object run_summary{
                {"report", filesystem::absolute(report_path).string()},
                {"status", report.complete ? "complete" : "incomplete"},
                {"frame_count", report.frames.size()},
                {"excluded_initial_frames",
                 std::min<size_t>(discard_count_option.getValue(), report.frames.size())},
                {"first_frame_ms", report.frames.empty()
                                       ? SJsonValue{}
                                       : SJsonValue{report.frames.front().inference_ms}},
                {"latency_ms",
                 latency_samples.empty()
                     ? SJsonValue{}
                     : MakeStatisticsJson(metrics::ComputeSampleStatistics(latency_samples))},
                {"eligible_for_comparison", report.complete && !latency_samples.empty()},
                {"metadata", report.metadata.fields},
                {"error", report.error},
                {"point_errors", SJsonValue{}}};

            // Coordinate evaluation uses all frames, including discarded timing samples.
            if (!reference_path_option.getValue().empty())
            {
                std::vector<std::pair<std::string, ptafdeploy::inference::SPoint2D>> predictions;
                for (const auto& frame : report.frames)
                {
                    try
                    {
                        const auto& point_fields =
                            FindPointFieldsByPath(frame.fields, point_field_option.getValue());
                        predictions.push_back(
                            {frame.source, ptafdeploy::inference::SPoint2D{
                                               ReadPointCoordinateAsFloat(point_fields.at("x")),
                                               ReadPointCoordinateAsFloat(point_fields.at("y"))}});
                    }
                    catch (const std::exception& error)
                    {
                        throw std::invalid_argument("Invalid prediction " + frame.source + ": " +
                                                    error.what());
                    }
                }

                const auto errors = metrics::ComputeMatchedPointErrors(predictions, references);
                run_summary["point_errors"] = SJsonValue::Object{
                    {"units", units_option.getValue()},
                    {"coordinate_convention", coordinate_convention_option.getValue()},
                    {"matched", errors.matched},
                    {"unmatched_predictions", MakeStringArrayJson(errors.unmatched_predictions)},
                    {"unmatched_references", MakeStringArrayJson(errors.unmatched_references)},
                    {"euclidean",
                     errors.euclidean ? MakeStatisticsJson(*errors.euclidean) : SJsonValue{}},
                    {"bias_x", errors.matched ? SJsonValue{errors.bias_x} : SJsonValue{}},
                    {"bias_y", errors.matched ? SJsonValue{errors.bias_y} : SJsonValue{}},
                    {"rmse", errors.matched ? SJsonValue{errors.rmse} : SJsonValue{}}};
            }

            run_summaries.emplace_back(std::move(run_summary));
        }

        // Validate every report before creating the destination.
        const auto serialized_summary = inference_output::Serialize(SJsonValue::Object{
            {"schema_version", 1}, {"complete", complete}, {"reports", std::move(run_summaries)}});
        if (!summary_path.parent_path().empty())
            filesystem::create_directories(summary_path.parent_path());

        std::ofstream summary_stream;
        summary_stream.exceptions(std::ios::badbit | std::ios::failbit);
        summary_stream.open(summary_path, std::ios::binary);
        summary_stream << serialized_summary << '\n';
        summary_stream.close();
        return complete ? 0 : 2;
    }
    catch (const TCLAP::ExitException& error)
    {
        return error.getExitStatus();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
