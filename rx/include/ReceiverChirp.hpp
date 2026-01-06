#pragma once

#include "Receiver.hpp"
#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cstdint>
#include <vector>
using std::cout;
using std::endl;

#define EPSILON 1e-12

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
    int64_t est_tau_samples = 0;
    // correlation metrics
    double corr_peak_mag = 0.0;
    double corr_mean_mag = 0.0;

    double est_cfo_hz = 0.0;

    // estimated SNR in dB
    float snr_db = -std::numeric_limits<float>::infinity();

    ~RxResultsChirp() override = default;

    std::string csv_header() const override {
        return "frame_id,timestamp,est_tau_samples,corr_peak_mag,corr_mean_mag,est_cfo_hz,snr_db,detected\n";
    }

    std::string csv_row() const override {
        std::ostringstream oss;
        oss << frame_id << "," 
            << timestamp << ","
            << est_tau_samples << ","
            << corr_peak_mag << ","
            << corr_mean_mag << ","
            << est_cfo_hz << ","
            << snr_db << ","
            << (detected ? "1" : "0") << "\n";
        return oss.str();
    }
};

class ReceiverChirp : public Receiver {
private:
    struct MatchFilterResult {
        std::size_t best_tau_ext = 0;
        int64_t tau_rel = 0;
        double best_mag = 0.0;
        double mean_mag = 0.0;
    };

public:
    using cfloat = std::complex<float>;
    using cdouble = std::complex<double>;

    ReceiverChirp(const RxConfigChirp& config) 
        : Receiver(config.common), config_(config),
          replica_chirp_(config.duration_sample, cfloat(0.0f, 0.0f)),
          prev_tail_((config.duration_sample > 0) ? (config.duration_sample - 1) : 0, cfloat(0.0f, 0.0f)),
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
        if (N < 2 || L == 0) {
            res->detected = false;
            return res;
        }

        // =====================================================================
        // set up extended frame parameters
        const std::size_t tail_length = (L > 0) ? (L - 1) : 0;
        // Extended length without materializing x_ext
        const std::size_t Next = tail_length + N;
        if (Next < L) {
            res->detected = false;
            update_prev_tail_(frame);
            return res;
        }

        // Search range in extended coordinates
        std::size_t max_tau_ext = Next - L;
        // Interpret search_span as limiting START positions inside the CURRENT frame:
        // tau_rel in [0, search_span] => tau_ext in [tail_length, tail_length + search_span]
        if (config_.search_span > 0) {
            const std::size_t max_tau_rel = std::min(config_.search_span, (N >= L) ? (N - L) : 0);
            max_tau_ext = std::min(max_tau_ext, tail_length + max_tau_rel);
        }

        // =====================================================================
        // Match filter processing & update results structf
        MatchFilterResult mf_res = match_filter_(frame, max_tau_ext);
        res->est_tau_samples = mf_res.tau_rel;
        res->corr_peak_mag = mf_res.best_mag;
        res->corr_mean_mag = mf_res.mean_mag;
        const double ratio = mf_res.best_mag / (mf_res.mean_mag + EPSILON);
        res->detected = (ratio >= config_.detection_threshold);
        res->snr_db = res->detected
            ? static_cast<float>(10.0 * std::log10((mf_res.best_mag + EPSILON) / (mf_res.mean_mag + EPSILON)))
            : -std::numeric_limits<float>::infinity();

        if (!res->detected) {
            cout << "Chirp not detected: "
                << "tau_ext=" << mf_res.best_tau_ext
                << ", tau_rel=" << mf_res.tau_rel
                << ", ratio=" << ratio
                << ", threshold=" << config_.detection_threshold
                << ", est_cfo_hz=" << res->est_cfo_hz 
                << endl;
        } else {
            res->est_cfo_hz = estimate_cfo_hz_(frame, mf_res.best_tau_ext);

            cout << "Chirp detected: "
                << "tau_ext=" << mf_res.best_tau_ext
                << ", tau_rel=" << mf_res.tau_rel
                << ", ratio=" << ratio
                << ", threshold=" << config_.detection_threshold
                << ", est_cfo_hz=" << res->est_cfo_hz 
                << endl;
        }

        update_prev_tail_(frame);
        return res;
    }

private:
    inline cfloat sample_ext_(const Frame& frame, std::size_t ext_idx) const {
        const std::size_t L = config_.duration_sample;
        const std::size_t tail_length = (L > 0) ? (L - 1) : 0;

        // If no previous tail is available yet, treat it as zero padding.
        if (tail_length > 0 && ext_idx < tail_length) {
            return have_prev_tail_ ? prev_tail_[ext_idx] : cfloat(0.0f, 0.0f);
        }

        const std::size_t i = ext_idx - tail_length;

        // Buffer bounds-checked via assert in operator[]
        return frame.data_[i];
    }

    void update_prev_tail_(const Frame& frame) {
        const std::size_t L = config_.duration_sample;
        const std::size_t N = frame.data_.size();
        const std::size_t tail_length = (L > 0) ? (L - 1) : 0;

        if (tail_length == 0) {
            prev_tail_.clear();
            have_prev_tail_ = true; // no tail is needed, but the state is valid
            return;
        }

        prev_tail_.resize(tail_length, cfloat(0.0));
        const cfloat* x_ptr = frame.data_.h_data();
        
        // copy the frame data to the prev_tail_ vector
        if (N >= tail_length) {
            std::copy_n(x_ptr + (N - tail_length), tail_length, prev_tail_.begin());
        }
        else {
            // right align the frame that's too short for the entire prev_tail_
            // vector, zero pad the front
            const std::size_t pad = tail_length - N;
            std::fill(prev_tail_.begin(), prev_tail_.begin() + pad, cfloat(0.0));
            std::copy_n(x_ptr, N, prev_tail_.begin() + pad);
        }
        have_prev_tail_ = true;
    }

    // Perform matched filtering on the given frame (and previous tail if any),
    // return the best correlation results.
    // Also keep track of the correlation magnitudes and exclude the peak region
    // to avoid biasing the mean noise floor estimate
    MatchFilterResult match_filter_(const Frame& frame, std::size_t max_tau_ext) {
        const std::size_t L = config_.duration_sample;

        // initialize mags_ storage
        const std::size_t tau_count = max_tau_ext + 1;
        if (mags_.size() < tau_count) mags_.resize(tau_count);
        std::fill_n(mags_.data(), tau_count, 0.0);

        // matched filter
        double best_mag = -1.0;
        std::size_t best_tau_ext = 0;

        for (std::size_t tau = 0; tau <= max_tau_ext; ++tau) {
            cfloat acc_corr(0.0f, 0.0f);

            for (std::size_t n = 0; n < L; ++n) {
                const cfloat x = sample_ext_(frame, tau + n);
                const cfloat s = replica_chirp_[n];
                acc_corr += x * std::conj(s);
            }

            const double mag = static_cast<double>(std::abs(acc_corr));
            mags_[tau] = mag;

            if (mag > best_mag) {
                best_mag = mag;
                best_tau_ext = tau;
            }
        }

        // Mean magnitude excluding guard window around peak
        const std::size_t G = config_.mean_guard_samples;
        const std::size_t exclude_low  = (best_tau_ext > G) ? (best_tau_ext - G) : 0;
        const std::size_t exclude_high = std::min(max_tau_ext, best_tau_ext + G);

        double sum_mag = 0.0;
        std::size_t count = 0;

        for (std::size_t tau = 0; tau <= max_tau_ext; ++tau) {
            if (tau >= exclude_low && tau <= exclude_high) continue;
            sum_mag += mags_[tau];
            count++;
        }

        if (count == 0) {
            sum_mag = 0.0;
            count = mags_.size();
            for (double m : mags_) sum_mag += m;
        }

        const double mean_mag = (count > 0) ? sum_mag / static_cast<double>(count) : 0.0;

        // set up result struct
        MatchFilterResult res;
        res.best_tau_ext = best_tau_ext;
        // Convert tau_ext (extended coords) to tau_rel (relative to current frame start)
        res.tau_rel = static_cast<int64_t>(best_tau_ext) - static_cast<int64_t>((L > 0) ? (L - 1) : 0);
        res.best_mag = best_mag;
        res.mean_mag = mean_mag;
        return res;
    }

    // estimate Carrier Frequency Offset in Hz based on the received frame
    // CFO happens when there is a frequency mismatch between the transmitter
    // and receiver's local oscillators. This also happens when there is a doppler
    // shift due to relative motion between the tx and rx which is highly relevant
    // to this project
    // TODO: We use phase estimator based for now, we'll implement fft based 
    // estimator later
    double estimate_cfo_hz_(const Frame& frame, std::size_t best_tau_ext) const {
        const std::size_t L = config_.duration_sample;
        if (L < 2) return 0.0;
        const double fs = config_.common.sample_rate_hz;

        // one lag accumulator: sum z*[n] z[n + 1] = |A|^2 e^{j 2π f_cfo / fs}
        cdouble acc(0.0, 0.0);

        // build z[n] on the fly from x[n + best_tau_ext] and s[n]
        cfloat z_prev = sample_ext_(frame, best_tau_ext + 0) * std::conj(replica_chirp_[0]);

        for (std::size_t n = 0; n + 1 < L; n++) {
            const cfloat z_next = sample_ext_(frame, best_tau_ext + n + 1) * std::conj(replica_chirp_[n + 1]);
            acc += std::conj(cdouble(z_prev.real(), z_prev.imag())) * 
                   cdouble(z_next.real(), z_next.imag());
            z_prev = z_next;
        }

        // if acc magnitude is near 0, return 0 to show the phase is unreliable
        const double acc_mag = std::abs(acc);
        if (acc_mag < EPSILON) return 0.0;

        const double acc_phase_rad = std::atan2(acc.imag(), acc.real());
        const double cfo_hz = (acc_phase_rad * fs) / (2.0 * M_PI);
        return cfo_hz;
    }


private:
    RxConfigChirp config_;

    std::vector<cfloat> replica_chirp_;

    // temporary storage for correlation magnitudes during processing
    std::vector<double> mags_;

    // to handle chirps that span frame boundaries. Handle just one frame 
    // spanning for now
    std::vector<cfloat> prev_tail_;
    bool have_prev_tail_ = false;
};
