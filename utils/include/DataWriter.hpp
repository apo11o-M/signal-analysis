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
#include <complex>

#include "Frame.hpp"
#include "Receiver.hpp"

// DataWriter owns a "run directory" and writes:
//  1) IQ frames as binary .c64 files under tx/ imp/ rx/
//  2) a manifest CSV (frame_manifest.csv) that lists what files were written
//  3) receiver results CSV (rx_results.csv) for plotting
class DataWriter {
public:
    // base_dir: directory containing multiple runs, e.g. "dumps"
    // label: run label, e.g. "tone_test"
    explicit DataWriter(const std::string& base_dir, const std::string& label = "run") {
        run_output_dir_ = std::filesystem::path(base_dir) / make_run_dir_name_(label);

        std::filesystem::create_directories(run_output_dir_);
        std::filesystem::create_directories(run_output_dir_ / "tx");
        std::filesystem::create_directories(run_output_dir_ / "imp");
        std::filesystem::create_directories(run_output_dir_ / "rx");

        open_frame_manifest_csv_();
        open_rx_results_csv_();
    }

    // Path to this run directory (e.g., dumps/run_tone_test_20251222_203500)
    const std::filesystem::path& run_dir() const noexcept { return run_output_dir_; }

    // Writes the complex<float> IQ samples of a frame to:
    //   <run_dir>/<stage>/frame_XXXXXXXXXXXX.c64
    //
    // Also appends a row to frame_manifest.csv so the Python scripts can find files.
    //
    // stage: "tx", "imp", or "rx" (for now)
    void write_iq_frame_binary(const std::string& stage, const Frame& frame) {
        const auto rel_path = std::filesystem::path(stage) / make_frame_filename_(frame.frame_id);
        const auto abs_path = run_output_dir_ / rel_path;

        std::ofstream frame_bin(abs_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!frame_bin) {
            throw std::runtime_error("DataWriter: failed to open " + abs_path.string());
        }

        const auto bytes = static_cast<std::streamsize>(
            frame.data_.size() * sizeof(std::complex<float>)
        );

        frame_bin.write(reinterpret_cast<const char*>(frame.data_.h_data()), bytes);
        if (!frame_bin) {
            throw std::runtime_error("DataWriter: failed to write " + abs_path.string());
        }

        // Record in manifest CSV
        frame_manifest_csv_
            << stage << ","
            << frame.frame_id << ","
            << rel_path.generic_string() << ","
            << frame.data_.size() << ","
            << frame.timestamp_
            << "\n";
    }

    // Appends one row to rx/rx_results.csv using the schema provided by RxResults.
    // RxResults itself owns the CSV schema via:
    //   - RxResults::csv_header()
    //   - RxResults::csv_row()
    void write_rx_results(const RxResults& results) {
        // Ensure header is written once
        if (!rx_results_header_written_) {
            rx_results_csv_ << RxResults::csv_header();
            rx_results_header_written_ = true;
        }
        rx_results_csv_ << results.csv_row();
    }

private:
    // helper to open CSV files
    void open_frame_manifest_csv_() {
        const auto manifest_path = run_output_dir_ / "frame_manifest.csv";
        frame_manifest_csv_.open(manifest_path, std::ios::out | std::ios::trunc);
        if (!frame_manifest_csv_) {
            throw std::runtime_error("DataWriter: failed to open " + manifest_path.string());
        }

        // Manifest describes *IQ frame files* written by write_iq_frame_binary()
        frame_manifest_csv_ << "stage,frame_id,rel_path,elem_count,timestamp\n";
    }

    // helper to open rx_results.csv
    void open_rx_results_csv_() {
        const auto rx_path = run_output_dir_ / "rx" / "rx_results.csv";
        rx_results_csv_.open(rx_path, std::ios::out | std::ios::trunc);
        if (!rx_results_csv_) {
            throw std::runtime_error("DataWriter: failed to open " + rx_path.string());
        }
        // header is written lazily on first write_rx_results() call
        rx_results_header_written_ = false;
    }

    // helper to create the frame filename
    static std::string make_frame_filename_(uint64_t frame_id) {
        std::ostringstream oss;
        oss << "frame_" << std::setw(12) << std::setfill('0') << frame_id << ".c64";
        return oss.str();
    }

    // helper to create a run directory name with timestamp
    static std::string make_run_dir_name_(const std::string& label) {
        using namespace std::chrono;
        const auto t = system_clock::to_time_t(system_clock::now());
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << "run_" << label << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }

private:
    // Root directory for this run
    std::filesystem::path run_output_dir_;

    // CSV manifest listing where each IQ frame file was written
    std::ofstream frame_manifest_csv_;

    // CSV log of receiver estimates/results
    std::ofstream rx_results_csv_;
    bool rx_results_header_written_ = false;
};
