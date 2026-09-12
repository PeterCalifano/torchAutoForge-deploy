/**
 * @file value_parsing.cpp
 * @brief Implementation of strict reusable textual-value parsing.
 *
 * Successful numeric parsing performs no temporary string allocation. Owning
 * strings are created only for public results and actionable error messages.
 */

#include <utils/parsing/value_parsing.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <system_error>

namespace
{
    /** @brief Recognize the fixed ASCII whitespace accepted at numeric boundaries. */
    [[nodiscard]] constexpr bool IsAsciiWhitespace(const char character) noexcept
    {
        return character == ' ' || character == '\t' || character == '\n' || character == '\r' ||
               character == '\f' || character == '\v';
    }

    /** @brief Borrow the subview remaining after trimming ASCII boundary whitespace. */
    [[nodiscard]] constexpr std::string_view TrimAsciiWhitespace(std::string_view value) noexcept
    {
        while (!value.empty() && IsAsciiWhitespace(value.front()))
        {
            value.remove_prefix(1U);
        }
        while (!value.empty() && IsAsciiWhitespace(value.back()))
        {
            value.remove_suffix(1U);
        }
        return value;
    }

    /**
     * @brief Adapt one valid leading plus for `std::from_chars`.
     *
     * Repeated or conflicting signs remain unchanged so conversion rejects
     * them rather than normalizing malformed input.
     */
    [[nodiscard]] constexpr std::string_view AcceptLeadingPlus(std::string_view value) noexcept
    {
        if (value.size() > 1U && value.front() == '+' && value[1U] != '+' && value[1U] != '-')
        {
            value.remove_prefix(1U);
        }
        return value;
    }

    /** @brief Build an error message only after parsing has failed. */
    [[nodiscard]] std::string MakeError(const std::string_view context,
                                        const std::string_view detail)
    {
        std::string message;
        message.reserve(context.size() + detail.size() + 2U);
        message.append(context);
        message.append(": ");
        message.append(detail);
        return message;
    }
} // namespace

namespace ptafdeploy::utils::parsing
{
    SNamedValue ParseNamedValue(const std::string_view specification,
                                const std::string_view context)
    {
        if (specification.empty())
        {
            throw std::invalid_argument(MakeError(context, "expected a non-empty value"));
        }

        const size_t separator = specification.find('=');

        // A missing separator is the positional shorthand. Explicit named
        // syntax requires non-empty fields on both sides of the separator.
        if (separator == std::string_view::npos)
        {
            return SNamedValue{"", std::string(specification)};
        }
        if (separator == 0U || separator + 1U == specification.size())
        {
            throw std::invalid_argument(
                MakeError(context, "expected [name=]value with both explicit sides non-empty"));
        }

        return SNamedValue{std::string(specification.substr(0U, separator)),
                           std::string(specification.substr(separator + 1U))};
    }

    std::vector<int64_t> ParseIntegerList(const std::string_view value, const char delimiter,
                                          const std::string_view context)
    {
        if (delimiter == '\0')
        {
            throw std::invalid_argument(MakeError(context, "null is not a valid delimiter"));
        }
        if (value.empty())
        {
            throw std::invalid_argument(MakeError(context, "expected at least one integer"));
        }

        std::vector<int64_t> parsed_values;

        // Reserve exactly the maximum result count implied by the delimiters.
        parsed_values.reserve(
            1U + static_cast<size_t>(std::count(value.begin(), value.end(), delimiter)));

        size_t token_begin = 0U;
        while (token_begin <= value.size())
        {
            // Borrow and trim one token without materializing a temporary string.
            const size_t separator = value.find(delimiter, token_begin);
            const size_t token_end = separator == std::string_view::npos ? value.size() : separator;
            const std::string_view token =
                TrimAsciiWhitespace(value.substr(token_begin, token_end - token_begin));
            if (token.empty())
            {
                throw std::invalid_argument(MakeError(context, "integer tokens must not be empty"));
            }

            const std::string_view numeric_token = AcceptLeadingPlus(token);
            int64_t parsed_value = 0;

            // A successful conversion must consume the complete normalized token.
            const auto [parsed_end, error] = std::from_chars(
                numeric_token.data(), numeric_token.data() + numeric_token.size(), parsed_value);
            if (error != std::errc{} || parsed_end != numeric_token.data() + numeric_token.size())
            {
                throw std::invalid_argument(
                    MakeError(context, "invalid integer '" + std::string(token) + "'"));
            }
            parsed_values.push_back(parsed_value);

            if (separator == std::string_view::npos)
            {
                break;
            }
            token_begin = separator + 1U;
        }
        return parsed_values;
    }

    float ParseFiniteFloat(const std::string_view value, const std::string_view context)
    {
        const std::string_view token = TrimAsciiWhitespace(value);
        if (token.empty())
        {
            throw std::invalid_argument(MakeError(context, "expected one finite float"));
        }

        const std::string_view numeric_token = AcceptLeadingPlus(token);
        std::string_view conversion_token = numeric_token;
        std::chars_format format = std::chars_format::general;
        bool negate_hexadecimal = false;

        // Preserve the hexadecimal grammar accepted by the former std::stof
        // implementation. from_chars hex mode expects neither a sign nor the
        // 0x prefix, so apply those pieces explicitly.
        std::string_view unsigned_token = numeric_token;
        if (unsigned_token.front() == '-')
        {
            negate_hexadecimal = true;
            unsigned_token.remove_prefix(1U);
        }
        if (unsigned_token.starts_with("0x") || unsigned_token.starts_with("0X"))
        {
            conversion_token = unsigned_token.substr(2U);
            format = std::chars_format::hex;
        }
        else
        {
            negate_hexadecimal = false;
        }

        float parsed_value = 0.0F;

        // Reject conversion errors, partial consumption, and non-finite results
        // through one boundary so every failure carries the same context.
        const auto [parsed_end, error] = std::from_chars(
            conversion_token.data(), conversion_token.data() + conversion_token.size(),
            parsed_value, format);
        if (negate_hexadecimal)
        {
            parsed_value = -parsed_value;
        }
        if (error != std::errc{} ||
            parsed_end != conversion_token.data() + conversion_token.size() ||
            !std::isfinite(parsed_value))
        {
            throw std::invalid_argument(
                MakeError(context, "invalid finite float '" + std::string(token) + "'"));
        }
        return parsed_value;
    }
} // namespace ptafdeploy::utils::parsing
