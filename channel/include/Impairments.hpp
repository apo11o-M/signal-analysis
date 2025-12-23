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
// We should see a constant phase rotation and the phase difference estimator
// will see an offset of the estimated vs actual frequency
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


// Constant phase offset impairment
// This happens when the receiver doesn't know the absolute RF phase when it starts sampling
// Frequency estimate should still work, but coherent demodulation will be affected
class PhaseOffsetImpairment : public Impairment {
public:
    PhaseOffsetImpairment(double fs, double phase_rad) : phase_rad_(phase_rad) {}

    const char* name() const override { return "Phase Offset"; }

    void apply(Frame& f) override {
        const float c = static_cast<float>(std::cos(phase_rad_));
        const float s = static_cast<float>(std::sin(phase_rad_));
        const std::complex<float> rot(c, s);

        auto* x = f.data_.h_data();
        const std::size_t n = f.data_.size();
        for (std::size_t i = 0; i < n; i++) {
            x[i] *= rot;
        }
    }

private:
    double phase_rad_;
};


// Integer timing offset (either circular or zero pad)
// This happens when the receiver doesn't start sampling exactly at the frame boundary
// The phase estimator should still work, but symbol aligned systems will break
// and preambles become necessary
class TimingOffsetImpairment : public Impairment {
public:
    enum class Mode { Circular, ZeroPad };

    TimingOffsetImpairment(double fs, int sample_offset, Mode mode = Mode::Circular)
        : offset_(sample_offset), mode_(mode) {}

    const char* name() const override { return "Timing Offset Impairment"; }

    void apply(Frame& f) override {
        const std::size_t n = f.data_.size();
        if (n == 0 || offset_ == 0) return;

        std::vector<std::complex<float>> temp(n);
        const auto* x = f.data_.h_data();

        if (mode_ == Mode::Circular) {
            // y[i] = x[i - offset]
            for (std::size_t i = 0; i < n; i++) {
                const long long src = static_cast<long long>(i) - static_cast<long long>(offset_);
                temp[i] = x[wrap_index(src, n)];
            }
        }
        else {
            for (std::size_t i = 0; i < n; i++) {
                const long long src = static_cast<long long>(i) - static_cast<long long>(offset_);
                if (src < 0 || src >= static_cast<long long>(n)) temp[i] = { 0.0f, 0.0f };
                else temp[i] = x[static_cast<long long>(src)];
            }
        }

        auto* y = f.data_.h_data();
        std::copy(temp.begin(), temp.end(), y);
    }

private:
    int offset_;
    Mode mode_;
};
