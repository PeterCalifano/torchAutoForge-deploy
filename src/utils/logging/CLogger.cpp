/**
 * @file CLogger.cpp
 * @brief Dependency-free project logging implementation.
 */

#include "CLogger.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>

namespace ptafdeploy::logging
{
    namespace
    {
        constexpr std::string_view color_reset = "\033[0m";
        constexpr std::string_view critical_color = "\033[1;31m";
        constexpr std::string_view error_color = "\033[31m";
        constexpr std::string_view warning_color = "\033[33m";
        constexpr std::string_view info_color = "\033[34m";
        constexpr std::string_view debug_color = "\033[36m";
        constexpr std::string_view trace_color = "\033[2m";

        std::mutex &GetOutputMutex()
        {
            static std::mutex output_mutex;
            return output_mutex;
        }

        std::string_view TrimAsciiWhitespace(std::string_view text)
        {
            while (!text.empty() &&
                   std::isspace(
                       static_cast<unsigned char>(text.front())) != 0)
            {
                text.remove_prefix(1);
            }
            while (!text.empty() &&
                   std::isspace(
                       static_cast<unsigned char>(text.back())) != 0)
            {
                text.remove_suffix(1);
            }
            return text;
        }
    } // namespace

    CLogger::CLogger(
        std::string component_name,
        const ELogLevel level,
        const ELogColorMode color_mode,
        std::ostream &output_stream,
        std::ostream &diagnostic_stream)
        : component_name_(
              component_name.empty() ? "ptafdeploy" : std::move(component_name)),
          level_(level),
          color_mode_(color_mode),
          output_stream_(output_stream),
          diagnostic_stream_(diagnostic_stream)
    {
    }

    void CLogger::setLevel(const ELogLevel level) noexcept
    {
        level_.store(level, std::memory_order_relaxed);
    }

    ELogLevel CLogger::getLevel() const noexcept
    {
        return level_.load(std::memory_order_relaxed);
    }

    bool CLogger::shouldLog(const ELogLevel severity) const noexcept
    {
        const ELogLevel configured_level = getLevel();
        const auto configured_value =
            static_cast<std::uint8_t>(configured_level);
        const auto severity_value = static_cast<std::uint8_t>(severity);

        if (configured_level == ELogLevel::Quiet ||
            severity == ELogLevel::Quiet)
        {
            return false;
        }
        if (configured_value >
                static_cast<std::uint8_t>(ELogLevel::Trace) ||
            severity_value > static_cast<std::uint8_t>(ELogLevel::Trace))
        {
            return false;
        }
        return severity_value <= configured_value;
    }

    std::optional<ELogLevel>
    CLogger::tryParseLevel(std::string_view level_text)
    {
        level_text = TrimAsciiWhitespace(level_text);
        if (level_text.size() == 1 && level_text.front() >= '0' &&
            level_text.front() <= '6')
        {
            return static_cast<ELogLevel>(level_text.front() - '0');
        }

        std::string normalized_level(level_text);
        std::transform(
            normalized_level.begin(),
            normalized_level.end(),
            normalized_level.begin(),
            [](const unsigned char value)
            {
                return static_cast<char>(std::tolower(value));
            });

        if (normalized_level == "quiet" || normalized_level == "off")
        {
            return ELogLevel::Quiet;
        }
        if (normalized_level == "critical" || normalized_level == "fatal")
        {
            return ELogLevel::Critical;
        }
        if (normalized_level == "error")
        {
            return ELogLevel::Error;
        }
        if (normalized_level == "warning" || normalized_level == "warn")
        {
            return ELogLevel::Warning;
        }
        if (normalized_level == "info")
        {
            return ELogLevel::Info;
        }
        if (normalized_level == "debug")
        {
            return ELogLevel::Debug;
        }
        if (normalized_level == "trace")
        {
            return ELogLevel::Trace;
        }
        return std::nullopt;
    }

    bool CLogger::setLevelFromEnvironment(
        const std::string_view variable_name)
    {
        if (variable_name.empty())
        {
            return false;
        }

        const std::string variable_name_copy(variable_name);
        const char *environment_value =
            std::getenv(variable_name_copy.c_str());
        if (environment_value == nullptr)
        {
            return false;
        }

        const std::optional<ELogLevel> parsed_level =
            tryParseLevel(environment_value);
        if (!parsed_level.has_value())
        {
            return false;
        }
        setLevel(*parsed_level);
        return true;
    }

    void CLogger::writeMessage_(
        const ELogLevel severity,
        const std::string_view message)
    {
        std::ostream &selected_stream = selectStream_(severity);
        const bool use_color = color_mode_ == ELogColorMode::Enabled;

        std::ostringstream formatted_line_stream;
        if (use_color)
        {
            formatted_line_stream << getColorCode_(severity);
        }
        formatted_line_stream << '[' << component_name_ << "]["
                              << getLevelLabel_(severity) << "] " << message;
        if (use_color)
        {
            formatted_line_stream << color_reset;
        }
        formatted_line_stream << '\n';

        const std::string formatted_line = formatted_line_stream.str();
        const std::scoped_lock output_lock(GetOutputMutex());
        selected_stream << formatted_line;
    }

    std::ostream &
    CLogger::selectStream_(const ELogLevel severity) const noexcept
    {
        if (severity == ELogLevel::Critical ||
            severity == ELogLevel::Error ||
            severity == ELogLevel::Warning)
        {
            return diagnostic_stream_;
        }
        return output_stream_;
    }

    std::string_view
    CLogger::getLevelLabel_(const ELogLevel severity) noexcept
    {
        switch (severity)
        {
        case ELogLevel::Critical:
            return "CRITICAL";
        case ELogLevel::Error:
            return "ERROR";
        case ELogLevel::Warning:
            return "WARNING";
        case ELogLevel::Info:
            return "INFO";
        case ELogLevel::Debug:
            return "DEBUG";
        case ELogLevel::Trace:
            return "TRACE";
        case ELogLevel::Quiet:
        default:
            return "QUIET";
        }
    }

    std::string_view
    CLogger::getColorCode_(const ELogLevel severity) noexcept
    {
        switch (severity)
        {
        case ELogLevel::Critical:
            return critical_color;
        case ELogLevel::Error:
            return error_color;
        case ELogLevel::Warning:
            return warning_color;
        case ELogLevel::Info:
            return info_color;
        case ELogLevel::Debug:
            return debug_color;
        case ELogLevel::Trace:
            return trace_color;
        case ELogLevel::Quiet:
        default:
            return {};
        }
    }
} // namespace ptafdeploy::logging
