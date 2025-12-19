#pragma once

#include <complex>

#include "Buffer.hpp"
#include "Timestamp.hpp"

class Frame {
public:
    Frame() = default;
    Frame(std::size_t size) : data_(size) {};

    uint64_t frame_id = 0;
    Timestamp timestamp_;

    Buffer<std::complex<float>> data_;

    void dump(std::ostream& stream = std::cout, std::size_t max_elems = 0) {
        stream << "Frame ID: " << frame_id << ", Timestamp: " << timestamp_ << "\n";
        data_.dump(stream, max_elems);
    }
};
