#pragma once

#include <cstring>
#include <cstddef>
#include <utility>
#include <iostream>

// A buffer class that holds data for processing, could be both on host and device
// TODO: add cuda implementation later
template <typename T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t size) {
        size_ = size;
        if (size_ > 0) h_ptr_ = new T[size_];
    }

    ~Buffer() { release(); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) {
        size_ = other.size_;
        h_ptr_ = other.h_ptr_;
        other.size_ = 0;
        other.h_ptr_ = nullptr;
    }

    Buffer& operator=(Buffer&& other) {
        if (this == &other) return *this;
        release();
        size_ = other.size_;
        h_ptr_ = other.h_ptr_;
        other.size_ = 0;
        other.h_ptr_ = nullptr;
        return *this;
    }

    size_t size() const noexcept { return size_; }
    bool empty() const noexcept{ return size_ == 0; }
    
    void resize(size_t new_size) {
        if (new_size == size_ || new_size == 0) return; 
        release();
        size_ = new_size;
        h_ptr_ = new T[size_];
    }

    // host memory access
    T* h_data() noexcept { return h_ptr_; }
    const T* h_data() const noexcept { return h_ptr_; }

    T& operator[](size_t index) noexcept {
        return h_ptr_[index];
    }

    const T& operator[](size_t index) const noexcept {
        return h_ptr_[index];
    }

    void fill(const T& value) {
        std::memset(h_ptr_, value, size_ * sizeof(T));
    }

    void zeroize() {
        std::memset(h_ptr_, 0, size_ * sizeof(T));
    }

    void dump(std::ostream& stream = std::cout) const {
        stream << "Buffer: " << name << ", size: " << size_ << "\n";

        stream << "[";
        for (size_t i = 0; i < size_ - 1; i++) {
            stream << h_ptr_[i] << " "
        }
        stream << h_ptr_[size_ - 1] << "]\n";
    }

private:
    void release() {
        delete[] h_ptr_;
        h_ptr_ = nullptr;
        size_ = 0;
    }

private:
    std::string name;

    size_t size_ = 0;

    // host array pointer
    T* h_ptr_ = nullptr;

    // device array pointer
    // T* d_ptr_ = nullptr;

};

