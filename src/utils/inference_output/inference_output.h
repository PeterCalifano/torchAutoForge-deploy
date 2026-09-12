/**
 * @file inference_output.h
 * @brief Owned JSON values, tensor serialization, and incremental inference reports.
 */
#pragma once
#include <filesystem>
#include <inference/inference_common.h>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ptafdeploy::utils::inference_output
{
    /** @brief Owned JSON tree; signed/unsigned integers retain their full precision. */
    struct SJsonValue
    {
        using Array = std::vector<SJsonValue>;
        using Object = std::map<std::string, SJsonValue>;
        using Value = std::variant<std::nullptr_t, bool, int64_t, uint64_t, double, std::string,
                                   Array, Object>;
        /** @brief Owned payload; strings and object keys must be valid UTF-8. */
        Value value{nullptr};
        /** @name Value construction
         * @brief Construct an owned JSON value; arrays/objects are moved from by-value inputs.
         * @details Numeric values use int64, uint64, or double storage. Finiteness and
         * UTF-8 validity are checked on serialization; C strings must be non-null.
         * @{ */
        SJsonValue() = default;
        SJsonValue(std::nullptr_t) : value(nullptr) {}
        SJsonValue(bool v) : value(v) {}
        template <std::signed_integral T> SJsonValue(T v) : value(int64_t(v)) {}
        template <std::unsigned_integral T> SJsonValue(T v) : value(uint64_t(v)) {}
        template <std::floating_point T> SJsonValue(T v) : value(double(v)) {}
        SJsonValue(const char* v) : value(std::string(v)) {}
        SJsonValue(std::string v) : value(std::move(v)) {}
        SJsonValue(Array v) : value(std::move(v)) {}
        SJsonValue(Object v) : value(std::move(v)) {}
        /** @} */
    };
    /** @brief Run metadata; fields cannot replace schema_version, status, frames, or error. */
    struct SRunMetadata
    {
        /** @brief Application schema version written into each published report. */
        uint32_t schema_version{1};
        /** @brief Owned application fields; reserved names are rejected. */
        SJsonValue::Object fields;
    };
    /** @brief One completed frame; additional fields cannot replace its identity or timing. */
    struct SFrameRecord
    {
        /** @brief Application-supplied zero-based sequence position. */
        size_t index{};
        /** @brief Source identity, normally relative to the run input directory. */
        std::string source;
        /** @brief Finite nonnegative model execution duration, excluding output IO. */
        double inference_ms{};
        /** @brief Owned application fields; reserved names are rejected. */
        SJsonValue::Object fields;
    };
    /** @brief Serialize a borrowed value.
     * @param value Owned JSON tree borrowed for this call.
     * @return Compact UTF-8 JSON text.
     * @throws std::invalid_argument For non-finite numbers or invalid UTF-8.
     */
    [[nodiscard]] std::string Serialize(const SJsonValue& value);
    /** @brief Copy a validated host tensor into JSON values; no model execution occurs.
     * @param tensor Borrowed host storage, valid for this call.
     * @return Name, dtype, shape, and values. float16/bfloat16 are unsupported.
     * @throws std::invalid_argument For malformed storage, unsupported types or non-finite values.
     */
    [[nodiscard]] SJsonValue TensorValue(const inference::STensorView& tensor);
    /** @brief Convert an owned float tensor through the same validated host-tensor conversion.
     * @param tensor Borrowed float tensor. @return Owned name/dtype/shape/values object.
     * @throws std::invalid_argument For invalid shape/cardinality or non-finite values.
     */
    [[nodiscard]] SJsonValue TensorValue(const inference::SFloatTensor& tensor);
    /** @brief Convert one completed frame into a JSON object.
     * @param frame Borrowed frame metadata and application fields. @return Owned JSON object.
     * @throws std::invalid_argument For invalid duration or reserved-field collisions.
     */
    [[nodiscard]] SJsonValue FrameValue(const SFrameRecord& frame);
    /** @brief Create a new/empty output directory, rejecting input ancestors and collisions.
     * @param output Directory to reserve for exclusive use by this run.
     * @param input Input location; an output descendant is allowed for fixed nonrecursive input.
     * @throws std::exception For invalid paths, collisions, or filesystem failures.
     * @note Callers must exclude concurrent writers; this is not an interprocess lock.
     */
    void PrepareOutput(const std::filesystem::path& output, const std::filesystem::path& input);
    /** @brief Sole owner of a disk spool and atomically replaced predictions.json.
     * @details Only successfully closed records are committed. Memory scales with one record.
     * Abrupt termination and power-loss recovery are outside the contract.
     */
    class CReport
    {
        std::filesystem::path root_;
        size_t committed_frames_{};
        enum class EState
        {
            active,
            append_failed,
            complete
        };
        EState state_{EState::active};

      public:
        /** @brief Metadata may be replaced after model loading; validated on publication. */
        SRunMetadata metadata;
        /** @brief Initialize a report in a reserved empty directory and publish incomplete status.
         * @param root Empty output directory reserved exclusively for this run.
         * @param initial Initial metadata; may be replaced after loading the model.
         * @throws std::exception If metadata or output publication fails.
         */
        CReport(const std::filesystem::path& root, SRunMetadata initial);
        /** @brief Reports have exclusive ownership of their spool and destination. */
        CReport(const CReport&) = delete;
        /** @brief Report ownership cannot be duplicated by assignment. */
        CReport& operator=(const CReport&) = delete;
        /** @brief Flush one valid frame; failed writes do not commit it.
         * @param frame Completed record, borrowed until its serialized bytes have been closed.
         * @throws std::exception If serialization or disk IO fails, or appending is no longer
         * allowed.
         */
        void Append(const SFrameRecord& frame);
        /** @brief Publish all committed records with status and optional structured error.
         * @param complete Whether every selected frame completed successfully.
         * @param error Structured application error, or null when absent.
         * @throws std::exception If publication fails or completion is invalid after a failed
         * append.
         */
        void Publish(bool complete, const SJsonValue& error = {});
    };
} // namespace ptafdeploy::utils::inference_output
