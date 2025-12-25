#pragma once

#include "Transmitter.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>

struct TxConfigChirp {
    TxConfigCommon common;

    double chirp_start_hz = 1.0e3;
    double chirp_end_hz = 100.0e3;
    float amplitude = 0.5f;
};

class TransmitterChirp : public Transmitter {
public:
    using cfloat = std::complex<float>;

    TransmitterChirp(const TxConfigChirp& config, std::size_t total_frames) 
        : Transmitter(config.common), config_(config) {

        // calculate the linear ramp factor, representing frequency increment per sample
        const std::size_t total_N = config_.common.frame_len * total_frames;
        linear_ramp_ = (config_.chirp_end_hz - config_.chirp_start_hz) / static_cast<double>(total_N - 1);
    };

    ~TransmitterChirp() override = default;

    // Generate a new frame
    Frame next_frame() override {
        const std::size_t frame_len = config_.common.frame_len;

        Frame f(frame_len);
        f.frame_id = frame_index_;
        f.timestamp_ = Timestamp();
        
        for (std::size_t i = 0; i < frame_len; i++) {
            float re = static_cast<float>(std::cos(phase_));
            float im = static_cast<float>(std::sin(phase_));
            f.data_[i] = config_.amplitude * cfloat(re, im);
            
            // we don't want the chip to reset its frequency to f0 every frame,
            // so we keep track of the global sample index so the frequency keeps
            // ramping up continuously.
            const std::size_t global_i = frame_index_ * frame_len + i;
            const double fn = config_.chirp_start_hz + linear_ramp_ * static_cast<double>(global_i);
            phase_ += 2.0 * M_PI * (fn / config_.common.sample_rate_hz);

            phase_ = std::fmod(phase_, 2.0 * M_PI);
            if (phase_ < 0) phase_ += 2.0 * M_PI;
        }
        frame_index_++;
        return f;
    };

protected:
    TxConfigChirp config_;
    double phase_ = 0.0;

    // frequency increment per sample
    double linear_ramp_ = 0.0;
};
