/**
 * @file value_parsing.h
 * @brief Strict, framework-independent parsing for reusable textual values.
 *
 * This installed API owns syntax that is useful outside any one executable or
 * command-line framework. Parsing is locale-independent, inputs are borrowed
 * for the duration of each call, and returned values own their storage.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ptafdeploy::parsing
{
    /**
     * @brief A textual value with an optional caller-supplied name.
     *
     * An empty name represents an unqualified value. The owning strings make
     * the result independent of the parsed input's lifetime. This type remains
     * an aggregate so callers can compose values without factory-only policy.
     */
    struct SNamedValue
    {
        /** @brief Optional qualifier; empty denotes an unqualified value. */
        std::string name{};

        /** @brief Required textual payload. */
        std::string value{};

        /**
         * @brief Compare both the optional name and textual payload.
         * @param other Value to compare.
         * @return True when both fields are equal.
         */
        [[nodiscard]] bool operator==(const SNamedValue& other) const = default;
    };

    /**
     * @brief Parse a value expressed as `[name=]value`.
     *
     * The first equals sign is the only separator. Further equals signs remain
     * part of the value, and an absent separator produces an empty name.
     * Whitespace is preserved because values may represent paths or other
     * whitespace-sensitive integration data.
     *
     * @param specification Text to parse; only the first equals sign separates
     * name from value.
     * @param context User-facing description included in validation errors.
     * @return An owning optional-name value.
     * @throws std::invalid_argument If the specification or either explicit
     * side of the separator is empty.
     */
    [[nodiscard]] SNamedValue ParseNamedValue(std::string_view specification,
                                              std::string_view context);

    /**
     * @brief Parse a strict delimiter-separated list of signed 64-bit integers.
     *
     * Each token accepts an optional sign and surrounding ASCII whitespace.
     * Conversion rejects partial parses and range overflow instead of applying
     * locale rules or silently discarding trailing characters.
     *
     * @param value Delimited text whose individual tokens may contain surrounding
     * ASCII whitespace.
     * @param delimiter Token delimiter, which must not be the null character.
     * @param context User-facing description included in validation errors.
     * @return Parsed integers in their source order.
     * @throws std::invalid_argument If the list or a token is empty, malformed,
     * contains trailing characters, overflows `int64_t`, or uses a null delimiter.
     */
    [[nodiscard]] std::vector<int64_t> ParseIntegerList(std::string_view value, char delimiter,
                                                        std::string_view context);

    /**
     * @brief Parse one strict finite single-precision value.
     *
     * Decimal, scientific, and hexadecimal forms are converted without locale
     * state. NaN and infinity are rejected even when the standard conversion
     * primitive recognizes their spelling.
     *
     * @param value Decimal, scientific, or `0x`-prefixed hexadecimal text whose
     * surrounding ASCII whitespace is ignored.
     * @param context User-facing description included in validation errors.
     * @return The parsed finite float.
     * @throws std::invalid_argument If the value is empty, malformed, has trailing
     * characters, overflows, or represents a non-finite value.
     */
    [[nodiscard]] float ParseFiniteFloat(std::string_view value, std::string_view context);
} // namespace ptafdeploy::parsing
