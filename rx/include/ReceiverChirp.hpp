#pragma once

#include "Receiver.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cstdint>
#include <vector>
using std::cout;
using std::endl;

struct RxConfigChirp {
    RxConfigCommon common;

    std::size_t frame_len = 4096;

    // chirp definition
    double f0 = 1.0e3;              // starting frequency in Hz
    double chirp_rate_hz = 5.0e6;     // chirp rate (slope) in Hz/s
    std::size_t duration_sample = 2048;   // duration in samples
    int8_t sweep_direction = 1;     // 1 for up-chirp, -1 for down-chirp

    // 0 = search for chip in entire frame, otherwise limit search from sample 0
    // to search_span samples
    std::size_t search_span = 1000;

    // detection thresholding
    // peak magnitude / (mean magnitude + eps)
    double detection_threshold = 3.0;

    // exclude +/- this many samples around the correlation peak when estimating
    // the mean noise floor, this protects the mean from getting inflated by the
    // peak and throw off the detection threshold
    std::size_t mean_guard_samples = 64;

    // TODO: CFO estimation parameters
};

// results generated from the receiver, this heavily depends on what waveform
// is being processed and what the receiver is designed to do.
struct RxResultsChirp : public RxResults {
    bool detected = false;

    // estimated timing offset, representing where the chirp starts/best aligned
    // in samples from the start of the frame
    std::size_t est_tau_samples = 0;
    // correlation metrics
    double corr_peak_mag = 0.0;
    double corr_mean_mag = 0.0;

    // estimated SNR in dB
    float snr_db = -std::numeric_limits<float>::infinity();

    ~RxResultsChirp() override = default;

    std::string csv_header() const override {
        return "frame_id,timestamp,est_tau_samples,corr_peak_mag,corr_mean_mag,snr_db,detected\n";
    }

    std::string csv_row() const override {
        std::ostringstream oss;
        oss << frame_id << "," 
            << timestamp << ","
            << est_tau_samples << ","
            << corr_peak_mag << ","
            << corr_mean_mag << ","
            << snr_db << ","
            << (detected ? "1" : "0") << "\n";
        return oss.str();
    }
};

class ReceiverChirp : public Receiver {
public:
    using cfloat = std::complex<float>;

    ReceiverChirp(const RxConfigChirp& config) 
        : Receiver(config.common), config_(config),
          replica_chirp_(config.duration_sample, cfloat(0.0f, 0.0f)),
          prev_tail_(config.duration_sample - 1, cfloat(0.0f, 0.0f)),
          have_prev_tail_(false) {

        // generate the replica chirp for match filtering
        const std::size_t N = config_.duration_sample;
        const double chirp_rate = config_.chirp_rate_hz * (config_.sweep_direction > 0 ? 1.0 : -1.0);
        const double fs = config_.common.sample_rate_hz;
        double phase = 0.0;

        for (std::size_t i = 0; i < N; i++) {
            float re = static_cast<float>(std::cos(phase));
            float im = static_cast<float>(std::sin(phase));
            replica_chirp_[i] = cfloat(re, im);

            // update instantaneous frequency
            double fn = config_.f0 + chirp_rate * (static_cast<double>(i)) / fs;
            phase += 2.0 * M_PI * (fn / fs);
            phase = std::fmod(phase, 2.0 * M_PI);
            if (phase < 0) phase += 2.0 * M_PI;
        }
    };

    std::unique_ptr<RxResults> process_frame(const Frame& frame) {
        std::unique_ptr<RxResultsChirp> res = std::make_unique<RxResultsChirp>();
        res->frame_id = frame.frame_id;
        res->timestamp = frame.timestamp_;

        const std::size_t N = frame.data_.size();
        const std::size_t L = config_.duration_sample;
        if (N < 2) {
            res->detected = false;
            return res;
        }

        // set up search range
        std::size_t max_tau = N - L;
        if (config_.search_span > 0) {
            max_tau = std::min(config_.search_span, max_tau);
        }

        const std::size_t tau_count = max_tau + 1;
        std::vector<double> mags(tau_count, 0.0);

        // stage 1: match filter to find the timing offset of the chirp
        double best_mag = -1.0;
        std::size_t best_tau = 0;

        for (std::size_t tau = 0; tau <= max_tau; tau++) {
            cfloat acc_corr(0.0f, 0.0f);

            // calculate correlation for this tau offset
            for (std::size_t n = 0; n < L; n++) {
                const cfloat x = frame.data_[tau + n];
                const cfloat s = replica_chirp_[n];
                acc_corr += x * std::conj(s);
            }

            // keep track of the best correlation magnitude
            const double mag = static_cast<double>(std::abs(acc_corr));
            mags[tau] = mag;

            if (mag > best_mag) {
                best_mag = mag;
                best_tau = tau;
            }
        }

        // calculate mean magnitude
        const std::size_t G = config_.mean_guard_samples;
        const std::size_t exclude_low = (best_tau > G) ? (best_tau - G) : 0;
        const std::size_t exclude_high = std::min(max_tau, best_tau + G);
        double sum_mag = 0.0;
        std::size_t count = 0;

        for (std::size_t tau = 0; tau <= max_tau; tau++) {
            if (tau >= exclude_low && tau <= exclude_high) continue;
            sum_mag += mags[tau];
            count++;
        }

        // fallback, if guard window excluded everything (which could happen if
        // max_tau is too small), revert to mean over all taus
        if (count == 0) {
            sum_mag = 0.0;
            count = mags.size();
            for (double m : mags) sum_mag += m;
        }

        double mean_mag = (count > 0) ? sum_mag / static_cast<double>(count) : 0.0;


        // update result struct
        res->est_tau_samples = best_tau;
        res->corr_peak_mag = best_mag;
        res->corr_mean_mag = mean_mag;

        const double ratio = best_mag / (mean_mag + 1e-12);
        res->detected = (ratio >= config_.detection_threshold);
        res->snr_db = static_cast<float>(10.0 * std::log10((best_mag) / (mean_mag + 1e-12)));

        if (!res->detected) {
            cout << "Chirp not detected: ratio = " << ratio << ", threshold = " << config_.detection_threshold << endl;
            res->snr_db = -std::numeric_limits<float>::infinity();
            return res;
        } else {
            cout << "Chirp detected: tau: " << best_tau 
                 << ", ratio = " << ratio 
                 << ", threshold = " << config_.detection_threshold << endl;
        }
        return res;
    }


private:
    RxConfigChirp config_;

    std::vector<cfloat> replica_chirp_;

    // to handle chirps that span frame boundaries. Handle just one frame 
    // spanning for now
    std::vector<cfloat> prev_tail_;
    bool have_prev_tail_ = false;
};
