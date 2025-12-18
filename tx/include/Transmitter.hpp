#pragma once

#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>

struct TxConfig {
    // fs
    double sample_rate_hz = 1.0e6;

    // f0, baseband
    double tone_freq_hz = 10.0e3;
    float amplitude = 0.5f;

    // samples per frame
    std::size_t frame_len = 4096;
};

class Transmitter {
public:
    using cfloat = std::complex<float>;
    Transmitter(const TxConfig& config) : config_(config) {
        // phase increment per sample: dphi = 2 * pi * f0 / fs;
        dphi_ = 2.0 * M_PI * (config_.tone_freq_hz / config_.sample_rate_hz);
    };

    // Generate a new frame
    Frame next_frame() {
        Frame f;
        f.frame_id = frame_index_;
        f.timestamp_ = Timestamp();
        
        for (std::size_t i = 0; i < config_.frame_len; i++) {
            float re = static_cast<float>(std::cos(phase_));
            float im = static_cast<float>(std::sin(phase_));
            f.data_[i] = config_.amplitude * cfloat(re, im);
            
            phase_ += dphi_;
            if (phase_ > 2.0 * M_PI) phase_ -= (2.0 * M_PI);
        }
        frame_index_++;
        return f;
    };

private:
    TxConfig config_;
    double dphi_ = 0.0;
    double phase_ = 0.0;
    std::uint64_t frame_index_ = 0;

};
