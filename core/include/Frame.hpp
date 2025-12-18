#pragma once

#include <complex>

#include "Buffer.hpp"
#include "Timestamp.hpp"

struct Frame {
    uint64_t frame_id = 0;
    Timestamp timestamp_;

    Buffer<std::complex<float>> data_;
};