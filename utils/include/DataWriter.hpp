#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "Frame.hpp"

// Simple data writer class for outputting data in binary for visualization and
// analysis
class DataWriter {
public:
    // base_dir: e.g. "dumps"
    // label:    e.g. "test1"
    explicit DataWriter(const std::string& base_dir, const std::string& label = "run")
    {
        run_dir_ = std::filesystem::path(base_dir) / make_run_name_(label);
        std::filesystem::create_directories(run_dir_);

        // stage directories you care about
        std::filesystem::create_directories(run_dir_ / "tx");
        std::filesystem::create_directories(run_dir_ / "imp");
        std::filesystem::create_directories(run_dir_ / "rx");

        // simple index
        index_.open(run_dir_ / "index.csv", std::ios::out | std::ios::trunc);
        if (!index_) throw std::runtime_error("DataWriter: failed to open index.csv");
        index_ << "stage,frame_id,rel_path,elem_count\n";
    }

    const std::filesystem::path& run_dir() const noexcept { return run_dir_; }

    // stage must be "tx", "imp", or "rx" for v0
    void write_frame(const std::string& stage, const Frame& f) {
        const auto rel = std::filesystem::path(stage) / frame_filename_(f.frame_id);
        const auto full = run_dir_ / rel;

        std::ofstream out(full, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("DataWriter: failed to open " + full.string());

        const auto bytes = static_cast<std::streamsize>(f.data_.size() * sizeof(std::complex<float>));
        out.write(reinterpret_cast<const char*>(f.data_.h_data()), bytes);
        if (!out) throw std::runtime_error("DataWriter: failed to write " + full.string());

        // record in index
        index_ << stage << ","
               << f.frame_id << ","
               << rel.generic_string() << ","
               << f.data_.size() << "\n";
    }

private:
    static std::string frame_filename_(uint64_t frame_id) {
        std::ostringstream oss;
        oss << "frame_" << std::setw(12) << std::setfill('0') << frame_id << ".c64";
        return oss.str();
    }

    static std::string make_run_name_(const std::string& label) {
        using namespace std::chrono;
        const auto t = system_clock::to_time_t(system_clock::now());
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << "run_" << label << "_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }

private:
    std::filesystem::path run_dir_;
    std::ofstream index_;
};
