#pragma once

#include "Receiver.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cstdint>

struct RxConfigSingleTone {
    RxConfigCommon common;
};

// results generated from the receiver, this heavily depends on what waveform
// is being processed and what the receiver is designed to do.
struct RxResultsSingleTone : public RxResults {
    bool detected = false;

    // estimated frequency of the received tone
    double est_freq_hz = 0.0;

    // estimated phase increment per sample
    double est_dphi_rad = 0.0;

    float avg_power = 0.0f;
    float mean_re = 0.0f;
    float mean_im = 0.0f;
    float var_re = 0.0f;
    float var_im = 0.0f;

    float noise_power = 0.0f;

    // estimated SNR in dB
    float snr_db = -std::numeric_limits<float>::infinity();

    ~RxResultsSingleTone() override = default;

    std::string csv_header() const override {
        return "frame_id,timestamp,est_freq_hz,est_dphi_rad,avg_power,mean_re,mean_im,var_re,var_im,noise_power,snr_db,detected\n";
    }

    std::string csv_row() const override {
        std::ostringstream oss;
        oss << frame_id << "," 
            << timestamp << ","
            << est_freq_hz << ","
            << est_dphi_rad << ","
            << avg_power << ","
            << mean_re << ","
            << mean_im << ","
            << var_re << ","
            << var_im << ","
            << noise_power << ","
            << snr_db << ","
            << (detected ? "1" : "0") << "\n";
        return oss.str();
    }
};

class ReceiverSingleTone : public Receiver {
public:
    using cfloat = std::complex<float>;

    ReceiverSingleTone(const RxConfigSingleTone& config) 
        : Receiver(config.common), config_(config) {};

    // Use a simple phase difference frequency estimator, since our 
    // transmitter generates a pure tone at just one frequency. No need to
    // use FFTs or other complex methods here for now
    // 
    // A phase difference estimator could be understood as "calculating the 
    // average phase difference between consecutive samples, and average them
    // to get an accurate estimate". It works on an assumption that the signal
    // is single tone only. 
    std::unique_ptr<RxResults> process_frame(const Frame& frame) {
        std::unique_ptr<RxResultsSingleTone> res = std::make_unique<RxResultsSingleTone>();
        res->frame_id = frame.frame_id;
        res->timestamp = frame.timestamp_;

        const std::size_t N = frame.data_.size();
        if (N < 2) {
            res->detected = false;
            return res;
        }

        // frequency estimate via phase difference
        // compute sum_{n=1..N-1} x[n] * conj(x[n-1])
        // angle(sum) estimates average phase advance per sample (dphi)
        cfloat acc_corr(0.0, 0.0);
        double sum_phase = 0.0;
        double sum_re = 0.0, sum_im = 0.0;

        for (std::size_t i = 0; i < N; i++) {
            const cfloat x = frame.data_[i];
            const double re = static_cast<double>(x.real());
            const double im = static_cast<double>(x.imag());
            sum_re += re;
            sum_im += im;
            sum_phase += re * re + im * im;

            if (i > 0) {
                const cfloat x_prev = frame.data_[i - 1];
                const cfloat product(
                    static_cast<float>(x.real()) * static_cast<float>(x_prev.real()) +
                    static_cast<float>(x.imag()) * static_cast<float>(x_prev.imag()),
                    static_cast<float>(x.imag()) * static_cast<float>(x_prev.real()) -
                    static_cast<float>(x.real()) * static_cast<float>(x_prev.imag())
                );
                acc_corr += product;
            }
        }

        // dphi in [-pi, pi]
        res->est_dphi_rad = std::atan2(acc_corr.imag(), acc_corr.real());
        res->est_freq_hz = (res->est_dphi_rad * config_.common.sample_rate_hz) / (2.0 * M_PI);


        // noise & SNR estimate
        // it's more of a heuristic and not very accurate. should be okay for now
        const double invN = 1.0 / static_cast<double>(N);
        res->avg_power = static_cast<float>(sum_phase * invN);
        res->mean_re = static_cast<float>(sum_re * invN);
        res->mean_im = static_cast<float>(sum_im * invN);

        double acc_var_re = 0.0, acc_var_im = 0.0;
        const double mu_re = static_cast<double>(res->mean_re);
        const double mu_im = static_cast<double>(res->mean_im);

        for (std::size_t i = 0; i < N; i++) {
            const cfloat x = frame.data_[i];
            const double diff_re = static_cast<double>(x.real()) - mu_re;
            const double diff_im = static_cast<double>(x.imag()) - mu_im;
            acc_var_re += diff_re * diff_re;
            acc_var_im += diff_im * diff_im;
        }

        const double invNm1 = 1.0 / static_cast<double>(N - 1);
        res->var_re = static_cast<float>(acc_var_re * invNm1);
        res->var_im = static_cast<float>(acc_var_im * invNm1);

        const float noise_proxy = (res->var_re < res->var_im) ? res->var_re : res->var_im;
        res->noise_power = noise_proxy;
        // signal power proxy
        float sig_power = res->avg_power - res->noise_power;
        if (sig_power < 1e-12f) sig_power = 1e-12f;
        float n_power = res->noise_power;
        if (n_power < 1e-12f) n_power = 1e-12f;
        
        res->snr_db = 10.0f * std::log10(sig_power / n_power);
        res->detected = (res->snr_db >= config_.common.snr_threshold_db);

        return res;
    }

private:
    RxConfigSingleTone config_;

};
