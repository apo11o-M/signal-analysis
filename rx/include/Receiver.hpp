#pragma once

#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cstdint>
#include <memory>

#define M_PI 3.14159265358979323846

struct RxConfigCommon {
    double sample_rate_hz = 1.0e6;
    float snr_threshold_db = 10.0f;
};

// results generated from the receiver, this heavily depends on what waveform
// is being processed and what the receiver is designed to do.
struct RxResults {
    uint64_t frame_id;
    Timestamp timestamp;

    virtual ~RxResults() = default;
    virtual std::string csv_header() const = 0;
    virtual std::string csv_row() const = 0;
};

class Receiver {
public:
    Receiver() = default;
    Receiver(const RxConfigCommon& config) : common_config_(config) {};

    virtual ~Receiver() = default;

    virtual std::unique_ptr<RxResults> process_frame(const Frame& frame) = 0;

protected:
    RxConfigCommon common_config_;

};
