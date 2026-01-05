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

    // expected chirp definition
    double chirp_start_hz = 1.0e3;
    double chirp_end_hz = 5.0e3;

    // acquisition search
    std::size_t search_max_offset_samples = 0;

    // detection thresholding
    // peak magnitude / (mean magnitude + eps)
    double detection_threshold = 3.0;

    // cfo estimation
    // after dechirp, we estimate cfo from phase differences
    std::size_t cfo_est_skip_samples = 0;
    std::size_t cfo_est_count_samples = 0;


    
};

// results generated from the receiver, this heavily depends on what waveform
// is being processed and what the receiver is designed to do.
struct RxResultsChirp : public RxResults {
    bool detected = false;

    // estimated timing offset, representing where the chirp starts/best aligned
    int64_t timing_offset_samples = 0;
    double corr_peak_mag = 0.0;
    double corr_mean_mag = 0.0;
    double detect_metric = 0.0; // corr_peak_mag / (corr_mean_mag + eps)

    // estimated constant frequency offset after dechirp
    double est_cfo_hz = 0.0;
    double est_cfo_dphi_rad = 0.0;

    // debug stats
    float avg_power = 0.0f;
    float noise_power = 0.0f;

    // estimated SNR in dB
    float snr_db = -std::numeric_limits<float>::infinity();

    ~RxResultsChirp() override = default;

    std::string csv_header() const override {
        return "frame_id,timestamp,timing_offset_samples,corr_peak_mag,corr_mean_mag,detect_metric,est_cfo_hz,est_cfo_dphi_rad,avg_power,noise_power,snr_db,detected\n";
    }

    std::string csv_row() const override {
        std::ostringstream oss;
        oss << frame_id << "," 
            << timestamp << ","
            << timing_offset_samples << ","
            << corr_peak_mag << ","
            << corr_mean_mag << ","
            << detect_metric << ","
            << est_cfo_hz << ","
            << est_cfo_dphi_rad << ","
            << avg_power << ","
            << noise_power << ","
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
          replica_chirp_(config.frame_len, cfloat(0.0f, 0.0f)) {

        // generate the replica chirp for match filtering
        const std::size_t N = config_.frame_len;
        const double f0 = config_.chirp_start_hz;
        const double f1 = config_.chirp_end_hz;
        const double fs = config_.common.sample_rate_hz;

        // linear ramp in Hz/sample across the reference length
        const double ramp = (N > 1) ? (f1 - f0) / static_cast<double>(N - 1) : 0.0;
        
        double phase = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            replica_chirp_[i] = cfloat(
                static_cast<float>(std::cos(phase)),
                static_cast<float>(std::sin(phase))
            );
            const double fn = f0 + ramp * static_cast<double>(i);
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
        if (N < 2) {
            res->detected = false;
            return res;
        }

        const double fs = config_.common.sample_rate_hz;
        constexpr double kTwoPi = 2.0 * M_PI;
        constexpr double eps = 1e-12;

        {
            double p = 0.0;
            for (std::size_t i = 0; i < N; i++) {
                const cfloat& v = frame.data_[i];
                const double re = static_cast<double>(v.real());
                const double im = static_cast<double>(v.imag());
                p += re * re + im * im;
            }
            res->avg_power = static_cast<float>(p / static_cast<double>(N));
        }

        // stage 1: match filter acquisition calculation
        // we search tau in [0, max_tau], and for each tau we correlate over L samples
        // y[tau] = sum_{n=0}^{L-1} r[tau + n] * conj(s[n])
        const std::size_t max_tau = std::min(
            config_.search_max_offset_samples,
            (N > 0) ? (N - 1) : 0
        );
        const std::size_t L = N - max_tau;
        if (L < 2) {
            res->detected = false;
            return res;
        }
        cout << "Acquisition search: max_tau = " << max_tau << ", L = " << L << endl;

        // correlate for each tau
        std::size_t best_tau = 0;
        double best_mag = -1.0;
        double sum_mag = 0.0;
        for (std::size_t tau = 0; tau <= max_tau; tau++) {
            std::complex<double> acc_corr(0.0, 0.0);
            // calculate correlation
            for (std::size_t n = 0; n < L; n++) {
                const cfloat x = frame.data_[tau + n];
                const cfloat s = replica_chirp_[n];

                acc_corr += std::complex<double>(x.real(), x.imag()) *
                            std::complex<double>(s.real(), -s.imag());
            }

            const double mag = std::abs(acc_corr);
            sum_mag += mag;
            if (mag > best_mag) {
                best_mag = mag;
                best_tau = tau;
            }
            cout << "tau: " << tau << ", mag: " << mag << endl;
        }
        const double mean_mag = sum_mag / static_cast<double>(max_tau + 1);
        const double metric = best_mag / (mean_mag + eps);

        res->timing_offset_samples = static_cast<int64_t>(best_tau);
        res->corr_peak_mag = best_mag;
        res->corr_mean_mag = mean_mag;
        res->detect_metric = metric;
        res->detected = (metric >= config_.detection_threshold);
        
        if (!res->detected) {
            // when not detected, treat the average power as noise
            res->noise_power = res->avg_power;
            res->snr_db = -std::numeric_limits<float>::infinity();
            return res;
        }

        
        // stage 2: align received chirp using best timing offset
        std::vector<cfloat> aligned_chirp(L, cfloat(0.0f, 0.0f));
        for (std::size_t i = 0; i < L; i++) {
            aligned_chirp[i] = frame.data_[best_tau + i];
        }


        // stage 3: dechirp
        std::vector<cfloat> dechirped_chirp(L, cfloat(0.0f, 0.0f));
        for (std::size_t i = 0; i < L; i++) {
            dechirped_chirp[i] = aligned_chirp[i] * std::conj(replica_chirp_[i]);
        }


        // stage 4: cfo estimate from dechirped signal (phase difference)
        const std::size_t skip = std::min(
            config_.cfo_est_skip_samples,
            (L > 1) ? (L - 1) : 0
        );
        std::size_t count = config_.cfo_est_count_samples;
        if (count == 0) {
            count = (L > 1 + skip) ? (L - 1 - skip) : 0;
        }
        else {
            count = std::min(count, (L > 1 + skip) ? (L - 1 - skip) : 0);
        }

        // not enough samples to estimate cfo, default to 0
        if (count == 0) {
            res->est_cfo_hz = 0.0;
            res->est_cfo_dphi_rad = 0.0;
        }
        else {
            double acc_dphi = 0;
            std::size_t M = 0;

            // phase diff between consecutive dechirped samples
            // dphi[i] = angle(dechirped_chirp[i] * conj(dechirped_chirp[i - 1]))
            for (std::size_t i = 1 + skip; i < 1 + skip + count; i++) {
                const cfloat w = dechirped_chirp[i] * std::conj(dechirped_chirp[i - 1]);
                const double dphi = std::atan2(
                    static_cast<double>(w.imag()),
                    static_cast<double>(w.real())
                );
                acc_dphi += dphi;
                M++;
            }
            const double est_dphi = acc_dphi / static_cast<double>(M);
            const double est_cfo = (est_dphi * fs) / kTwoPi;

            res->est_cfo_hz = est_cfo;
            res->est_cfo_dphi_rad = est_dphi;
        }


        // stage 5: estimate noise power and snr metric
        // TODO
        return res;
    }


private:
    RxConfigChirp config_;

    std::vector<cfloat> replica_chirp_;
};
