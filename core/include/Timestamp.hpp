#pragma once

#include <chrono>
#include <iostream>

struct Timestamp {
    std::chrono::steady_clock::time_point curr_time_;

    Timestamp() : curr_time_(std::chrono::steady_clock::now()) {}

    friend std::ostream& operator<<(std::ostream& os, const Timestamp& ts) {
        auto duration = ts.curr_time_.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        os << "Timestamp(" << millis << " ms since epoch)";
        return os;
    }
};
