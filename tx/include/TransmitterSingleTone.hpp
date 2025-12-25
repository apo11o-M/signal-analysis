#pragma once

#include "Transmitter.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>

struct TxConfigSingleTone {
    TxConfigCommon common;
    // f0, baseband
    double tone_freq_hz = 10.0e3;
    float amplitude = 0.5f;
};

class TransmitterSingleTone : public Transmitter {
public:
    using cfloat = std::complex<float>;

    TransmitterSingleTone(const TxConfigSingleTone& config) 
        : Transmitter(config.common), config_(config) {
        // phase increment per sample: dphi = 2 * pi * f0 / fs;
        dphi_ = 2.0 * M_PI * (config_.tone_freq_hz / config_.common.sample_rate_hz);
    };

    ~TransmitterSingleTone() override = default;

    // Generate a new frame
    Frame next_frame() override {
        Frame f(config_.common.frame_len);
        f.frame_id = frame_index_;
        f.timestamp_ = Timestamp();
        
        for (std::size_t i = 0; i < config_.common.frame_len; i++) {
            float re = static_cast<float>(std::cos(phase_));
            float im = static_cast<float>(std::sin(phase_));
            f.data_[i] = config_.amplitude * cfloat(re, im);
            
            phase_ += dphi_;
            if (phase_ > 2.0 * M_PI) phase_ -= (2.0 * M_PI);
        }
        frame_index_++;
        return f;
    };

protected:
    TxConfigSingleTone config_;
    double dphi_ = 0.0;
    double phase_ = 0.0;

};
