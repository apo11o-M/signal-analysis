#pragma once

#include "Frame.hpp"
#include "Util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
#include <random>

#define M_PI 3.14159265358979323846

// Impairment base class, being used in the Channel class in Channel.hpp
class Impairment {
public:
    virtual ~Impairment() = default;
    virtual const char* name() const = 0;
    virtual void apply(Frame& f) = 0;
};


// Apply a constant gain to the frame data
class GainImpairment : public Impairment {
public:
    GainImpairment(double fs, float gain) : gain_(gain) {};

    const char* name() const override { return "Gain Impairment"; }

    void apply(Frame& f) override {
        auto* x = f.data_.h_data();
        const std::size_t n = f.data_.size();
        for (std::size_t i = 0; i < n; i++) {
            x[i] *= gain_;
        }
    }

private:
    float gain_ = 1.0f;
};


// Carrier frequency offset
// Tx and Rx oscillators are never perfectly aligned, even a 1ppm error at 1 GHz
// gives a 1 kHz offset. 
// y[n] = x[n] * exp(j * (phase0 + n * w)), w = 2 * pi * df / fs
// phase accumulator is kept across frames (stateful)
class CFOImpairment : public Impairment {
public:
    CFOImpairment(double fs, double cfo_hz, double initial_phase_rad = 0.0) 
        : fs_(fs), cfo_hz_(cfo_hz), phase_(initial_phase_rad) {}

    const char* name() const override { return "Carrier Frequency Offset"; }

    void set_cfo_hz(double hz) { cfo_hz_ = hz; }

    void apply(Frame& f) override {
        auto* x = f.data_.h_data();
        const std::size_t n = f.data_.size();
        if (n == 0) return;

        const double w = 2.0 * M_PI * cfo_hz_ / fs_;
        double phase = phase_;
        for (std::size_t i = 0; i < n; i++) {
            const float c = static_cast<float>(std::cos(phase));
            const float s = static_cast<float>(std::sin(phase));
            x[i] *= std::complex<float>(c, s);
            phase += w;
        }
        phase_ = std::remainder(phase, 2.0 * M_PI);
    }

private:
    double fs_;
    double cfo_hz_;
    double phase_;
};


// Additive White Gaussian Noise (AWGN)
// Random thermal noise from electronics, atmosphere, cosmic background, etc
// Achieved by adding random complex noise to each sample
class AWGNImpairment : public Impairment {
public:
    AWGNImpairment(double fs, double snr_db, uint64_t seed =1234)
        : snr_db_(snr_db), rng_(seed), norm_(0.0, 1.0) {}
    
    const char* name() const override { return "Additive White Gaussian Noise"; }

    void set_snr_db(double snr_db) { snr_db_ = snr_db; }

    void apply(Frame& f) override {
        const std::size_t n = f.data_.size();
        if (n == 0) return;
        
        const double p_sig = frame_avg_power(f);
        if (p_sig <= 0.0) return;

        const double snr_lin = std::pow(10.0, snr_db_ / 10.0);
        const double p_noise = p_sig / snr_lin;

        // complex noise power
        const double sigma = std::sqrt(p_noise / 2.0);

        auto* x = f.data_.h_data();
        for (std::size_t i = 0; i < n; i++) {
            const float nreal = static_cast<float>(sigma * norm_(rng_));
            const float nimag = static_cast<float>(sigma * norm_(rng_));
            x[i] += std::complex<float>(nreal, nimag);
        }
    }

private:
    double snr_db_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> norm_;
};
