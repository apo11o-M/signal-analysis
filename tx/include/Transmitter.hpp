#pragma once

#include "Frame.hpp"
#include "Timestamp.hpp"

#include <cmath>
#include <complex>

#define M_PI 3.14159265358979323846

struct TxConfigCommon {
    // fs
    double sample_rate_hz = 1.0e6;
    // samples per frame
    std::size_t frame_len = 4096;
};

class Transmitter {
public:
    Transmitter() = default;
    Transmitter(const TxConfigCommon& config) : common_config_(config) {};

    virtual ~Transmitter() = default;

    // Generate a new frame
    virtual Frame next_frame() = 0;

protected:
    TxConfigCommon common_config_;
    std::uint64_t frame_index_ = 0;

};
