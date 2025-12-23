#pragma once

#include <cstdint>

inline std::size_t wrap_index(long long index, std::size_t n) {
    if (n == 0) return 0;
    long long m = index % static_cast<long long>(n);
    if (m < 0) m += static_cast<long long>(n);
    return static_cast<std::size_t>(m);
}
