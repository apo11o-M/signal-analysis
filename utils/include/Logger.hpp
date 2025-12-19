#pragma once

#include <iostream>
#include <fstream>

// Simple logger class for debugging and info logging to human readable formats
class Logger {
public:
    enum class Level { INFO, WARNING, ERROR };

    explicit Logger(const std::string& filename)
        : log_file_(filename, std::ios::out) {}

    bool ok() const noexcept { return log_file_.is_open(); }

    std::ostream& stream(Level level) {
        log_file_ << "[" << level_str(level) << "] ";
        return log_file_;
    }

    void log(Level level, const std::string& msg) {
        stream(level) << msg << '\n';
    }

private:
    static const char* level_str(Level level) {
        switch (level) {
            case Level::INFO: return "INFO";
            case Level::WARNING: return "WARNING";
            case Level::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }

    std::ofstream log_file_;
};
