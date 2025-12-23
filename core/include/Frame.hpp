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

inline double frame_avg_power(const Frame& f) {
    const auto* x = f.data_.h_data();
    const std::size_t n = f.data_.size();
    if (n == 0) return 0.0;

    long double acc = 0.0L;
    for (std::size_t i = 0; i < n; i++) {
        const float re = x[i].real();
        const float im = x[i].imag();
        acc += static_cast<long double>(re * re) + static_cast<long double>(im * im);
    }
    return static_cast<double>(acc / static_cast<long double>(n));
}
