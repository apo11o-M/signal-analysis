#pragma once

#include "Transmitter.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>
#include <algorithm>

struct TxConfigChirp {
    TxConfigCommon common;

    double f0 = 1.0e3;              // starting frequency in Hz
    double chirp_rate_hz = 5.0e6;     // chirp rate (slope) in Hz/s
    std::size_t duration_sample = 2048;   // duration in samples
    int8_t sweep_direction = 1;     // 1 for up-chirp, -1 for down-chirp

    float amplitude = 0.5f;
};

class TransmitterChirp : public Transmitter {
public:
    using cfloat = std::complex<float>;

    TransmitterChirp(const TxConfigChirp& config)
        : Transmitter(config.common), config_(config) {}

    ~TransmitterChirp() override = default;

    // Generate a new frame
    Frame next_frame() override {
        const std::size_t frame_len = config_.common.frame_len;

        const double chirp_rate = config_.chirp_rate_hz * (config_.sweep_direction > 0 ? 1.0 : -1.0);
        const double fs = config_.common.sample_rate_hz;
        const double f0 = config_.f0;

        Frame f(frame_len);
        f.frame_id = frame_index_;
        f.timestamp_ = Timestamp();
        for (std::size_t i = 0; i < frame_len; i++) {
            float re = static_cast<float>(std::cos(phase_));
            float im = static_cast<float>(std::sin(phase_));
            f.data_[i] = config_.amplitude * cfloat(re, im);

            // update instantaneous frequency
            double fn = f0 + chirp_rate * (static_cast<double>(chirp_pos_)) / fs;

            phase_ += 2.0 * M_PI * (fn / fs);
            phase_ = std::fmod(phase_, 2.0 * M_PI);
            if (phase_ < 0) phase_ += 2.0 * M_PI;

            chirp_pos_++;
            // reset chirp once it reached the chirp duration
            if (chirp_pos_ >= config_.duration_sample) { chirp_pos_ = 0; }
        }

        frame_index_++;
        return f;
    }

protected:
    TxConfigChirp config_;
    double phase_ = 0.0;

    // count samples within the current chip in case the chirp goes over 
    // the frame boundary
    std::size_t chirp_pos_ = 0;

};
