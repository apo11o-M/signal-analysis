#pragma once

#include "Frame.hpp"
#include "Impairments.hpp"
#include "Logger.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class Channel {
public:
    explicit Channel(double sample_rate_hz, Logger& logger) 
        : fs_(sample_rate_hz), logger_(logger) {}

    double sample_rate_hz() const { return fs_; }

    template <typename T, typename... Args>
    T& add(Args&&... args) {
        static_assert(std::is_base_of_v<Impairment, T>);
        auto ptr = std::make_unique<T>(fs_, std::forward<Args>(args)...);
        T& ref = *ptr;
        stages_.push_back(std::move(ptr));
        return ref;
    }

    void apply(Frame& f) {
        for (auto& s : stages_) {
            logger_.log(Logger::Level::INFO, std::string("Impairment entry: ") + s->name());
            s->apply(f);
        }
    }

private:
    double fs_;
    std::vector<std::unique_ptr<Impairment>> stages_;

    Logger& logger_;
};
