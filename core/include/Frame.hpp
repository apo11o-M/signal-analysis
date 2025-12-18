#pragma once

#include <complex>

#include "Buffer.hpp"
#include "Timestamp.hpp"

class Frame {
    uint64_t frame_id = 0;
    Timestamp timestamp_;

    Buffer<std::complex<float>> data_;

    void dump(std::ostream& stream = std::cout) {
        stream << "Frame ID: " << frame_id << "\n";
        data_.dump(stream, 5);
    }
};