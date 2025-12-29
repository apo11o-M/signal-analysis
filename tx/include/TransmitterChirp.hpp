#pragma once

#include "Transmitter.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>
#include <algorithm>

struct TxConfigChirp {
    TxConfigCommon common;

    // These define the chirp span *within a single frame*.
    double chirp_start_hz = 1.0e3;
    double chirp_end_hz   = 5.0e3;

    float amplitude = 0.5f;

    // Optional: absolute max instantaneous frequency before wrap/reset.
    // If <= 0, we default to ~Nyquist.
    double max_freq_hz = 100e3;
};

class TransmitterChirp : public Transmitter {
public:
    using cfloat = std::complex<float>;

    TransmitterChirp(const TxConfigChirp& config)
        : Transmitter(config.common), config_(config) {

        chirp_bw_hz_ = config_.chirp_end_hz - config_.chirp_start_hz;

        // Default reset limit near nyquist, also leaves a tiny margin to avoid
        // edge weirdness
        const double nyquist = 0.5 * config_.common.sample_rate_hz;
        const double nyquist_margin = 0.999 * nyquist;

        if (config_.max_freq_hz > 0.0) {
            reset_limit_hz_ = std::min(config_.max_freq_hz, nyquist_margin);
        }
        else {
            reset_limit_hz_ = nyquist_margin;
        }
    }

    ~TransmitterChirp() override = default;

    // Generate a new frame
    Frame next_frame() override {
        const std::size_t frame_len = config_.common.frame_len;

        // If the next chirp band would exceed our limit, wrap back.
        // We check the end-of-chirp since that's the peak instantaneous frequency.
        const double next_frame_end_hz = (config_.chirp_end_hz + band_offset_hz_);
        if (next_frame_end_hz >= reset_limit_hz_) {
            band_offset_hz_ = 0.0;
        }

        const double f0 = config_.chirp_start_hz + band_offset_hz_;
        const double f1 = config_.chirp_end_hz   + band_offset_hz_;

        // Linear ramp within this frame
        const double ramp_per_sample =
            (frame_len > 1) ? ((f1 - f0) / static_cast<double>(frame_len - 1)) : 0.0;

        Frame f(frame_len);
        f.frame_id = frame_index_;
        f.timestamp_ = Timestamp();
        
        for (std::size_t i = 0; i < frame_len; i++) {
            float re = static_cast<float>(std::cos(phase_));
            float im = static_cast<float>(std::sin(phase_));
            f.data_[i] = config_.amplitude * cfloat(re, im);

            const double fn = f0 + ramp_per_sample * static_cast<double>(i);
            phase_ += 2.0 * M_PI * (fn / config_.common.sample_rate_hz);

            phase_ = std::fmod(phase_, 2.0 * M_PI);
            if (phase_ < 0) phase_ += 2.0 * M_PI;
        }

        // Move the chirp band up for the next frame (so frames "stair-step" upward).
        // Using chirp bandwidth makes the next frame start at the previous frame's end.
        band_offset_hz_ += chirp_bw_hz_;

        frame_index_++;
        return f;
    }

protected:
    TxConfigChirp config_;
    double phase_ = 0.0;

    // How much the chirp band has been shifted up from the config's per-frame [start,end]
    double band_offset_hz_ = 0.0;

    double chirp_bw_hz_ = 0.0;
    double reset_limit_hz_ = 0.0;
};
