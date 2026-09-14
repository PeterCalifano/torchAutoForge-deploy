/**
 * @file CLogger.h
 * @brief Dependency-free project logging API.
 */

#pragma once

#include <atomic>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace ptafdeploy::logging
{
    /** @brief Ordered verbosity threshold used by CLogger. */
    enum class ELogLevel : std::uint8_t
    {
        Quiet = 0,
        Critical = 1,
        Error = 2,
        Warning = 3,
        Info = 4,
        Debug = 5,
        Trace = 6
    };

    /** @brief Select whether CLogger emits ANSI terminal color sequences. */
    enum class ELogColorMode : std::uint8_t
    {
        Disabled = 0,
        Enabled = 1
    };

    /** @brief A value that can be appended to a standard output stream. */
    template <typename TValue>
    concept StreamInsertable =
        requires(std::ostream &stream, TValue &&value)
        {
            stream << std::forward<TValue>(value);
        };

    /**
     * @brief Small dependency-free, component-scoped logger.
     *
     * Each message is formatted before taking a process-wide output lock, so
     * independent logger instances cannot interleave complete lines. Critical,
     * error, and warning messages use the diagnostic stream; less severe
     * messages use the ordinary output stream.
     *
     * Custom streams remain owned by the caller and must outlive the logger.
     */
    class CLogger final
    {
      public:
        /**
         * @brief Construct a logger for one component.
         *
         * @param component_name Name printed in each message. Empty values use
         * `ptafdeploy`.
         * @param level Initial verbosity threshold.
         * @param color_mode ANSI color policy.
         * @param output_stream Stream used for info/debug/trace messages.
         * @param diagnostic_stream Stream used for warnings and errors.
         */
        explicit CLogger(
            std::string component_name,
            ELogLevel level = ELogLevel::Error,
            ELogColorMode color_mode = ELogColorMode::Disabled,
            std::ostream &output_stream = std::cout,
            std::ostream &diagnostic_stream = std::clog);

        CLogger(const CLogger &) = delete;
        CLogger &operator=(const CLogger &) = delete;
        CLogger(CLogger &&) = delete;
        CLogger &operator=(CLogger &&) = delete;
        ~CLogger() = default;

        /** @brief Change the active verbosity threshold. */
        void setLevel(ELogLevel level) noexcept;

        /** @brief Return the active verbosity threshold. */
        [[nodiscard]] ELogLevel getLevel() const noexcept;

        /** @brief Return whether the supplied severity is enabled. */
        [[nodiscard]] bool shouldLog(ELogLevel severity) const noexcept;

        /**
         * @brief Parse a case-insensitive level name or a number from 0 to 6.
         *
         * Accepted names are quiet/off, critical/fatal, error, warning/warn,
         * info, debug, and trace.
         */
        [[nodiscard]] static std::optional<ELogLevel>
        tryParseLevel(std::string_view level_text);

        /**
         * @brief Apply a valid level read from an environment variable.
         *
         * @return True only when a valid value was found and applied.
         */
        bool setLevelFromEnvironment(
            std::string_view variable_name = "PTAFDEPLOY_LOG_LEVEL");

        template <StreamInsertable... TArgs>
        void critical(TArgs &&...args)
        {
            write_(ELogLevel::Critical, std::forward<TArgs>(args)...);
        }

        template <StreamInsertable... TArgs>
        void error(TArgs &&...args)
        {
            write_(ELogLevel::Error, std::forward<TArgs>(args)...);
        }

        template <StreamInsertable... TArgs>
        void warning(TArgs &&...args)
        {
            write_(ELogLevel::Warning, std::forward<TArgs>(args)...);
        }

        template <StreamInsertable... TArgs>
        void info(TArgs &&...args)
        {
            write_(ELogLevel::Info, std::forward<TArgs>(args)...);
        }

        template <StreamInsertable... TArgs>
        void debug(TArgs &&...args)
        {
            write_(ELogLevel::Debug, std::forward<TArgs>(args)...);
        }

        template <StreamInsertable... TArgs>
        void trace(TArgs &&...args)
        {
            write_(ELogLevel::Trace, std::forward<TArgs>(args)...);
        }

      private:
        template <StreamInsertable... TArgs>
        void write_(ELogLevel severity, TArgs &&...args)
        {
            if (!shouldLog(severity))
            {
                return;
            }

            std::ostringstream message_stream;
            (message_stream << ... << std::forward<TArgs>(args));
            writeMessage_(severity, message_stream.str());
        }

        void writeMessage_(ELogLevel severity, std::string_view message);
        [[nodiscard]] std::ostream &
        selectStream_(ELogLevel severity) const noexcept;
        [[nodiscard]] static std::string_view
        getLevelLabel_(ELogLevel severity) noexcept;
        [[nodiscard]] static std::string_view
        getColorCode_(ELogLevel severity) noexcept;

        std::string component_name_;
        std::atomic<ELogLevel> level_;
        ELogColorMode color_mode_;
        std::ostream &output_stream_;
        std::ostream &diagnostic_stream_;
    };
} // namespace ptafdeploy::logging
