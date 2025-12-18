#pragma once

#include <cstddef>
#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>

// A buffer class that holds data for processing, could be both on host and device
// TODO: add cuda implementation later
template <typename T>
class Buffer {
public:
    Buffer() noexcept = default;

    explicit Buffer(std::size_t size)
        : size_(size),
          h_ptr_(size_ ? new T[size_] : nullptr) {}

    ~Buffer() { release(); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : name(std::move(other.name)),
          size_(other.size_),
          h_ptr_(other.h_ptr_) {
        other.size_ = 0;
        other.h_ptr_ = nullptr;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        release();
        name = std::move(other.name);
        size_ = other.size_;
        h_ptr_ = other.h_ptr_;
        other.size_ = 0;
        other.h_ptr_ = nullptr;
        return *this;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    void resize(std::size_t new_size) {
        if (new_size == size_) return;
        release();
        size_ = new_size;
        h_ptr_ = size_ ? new T[size_] : nullptr;
    }

    // host memory access
    T* h_data() noexcept { return h_ptr_; }
    const T* h_data() const noexcept { return h_ptr_; }

    T& operator[](std::size_t index) noexcept {
        assert(index < size_);
        return h_ptr_[index];
    }
    const T& operator[](std::size_t index) const noexcept {
        assert(index < size_);
        return h_ptr_[index];
    }

    void fill(const T& value) {
        if (!h_ptr_) return;
        std::fill_n(h_ptr_, size_, value);
    }

    void zeroize() {
        fill(T{});
    }

    void set_name(std::string n) { name = std::move(n); }
    const std::string& get_name() const noexcept { return name; }

    void dump(std::ostream& stream = std::cout, std::size_t max_elems = 0) const {
        stream << "Buffer: " << name << ", size: " << size_ << "\n";
        stream << "[";

        if (size_ == 0) {
            stream << "]\n";
            return;
        }

        std::size_t n = size_;
        if (max_elems != 0 && max_elems < n) n = max_elems;

        for (std::size_t i = 0; i < n; ++i) {
            stream << h_ptr_[i];
            if (i + 1 < n) stream << " ";
        }

        if (n < size_) stream << " ...";
        stream << "]\n";
    }

private:
    void release() noexcept {
        delete[] h_ptr_;
        h_ptr_ = nullptr;
        size_ = 0;
    }

private:
    std::string name;
    std::size_t size_ = 0;

    // host array pointer
    T* h_ptr_ = nullptr;

    // device array pointer
    // T* d_ptr_ = nullptr;
};
