#ifndef RING_BUFFER_HPP_
#define RING_BUFFER_HPP_

#include <atomic>
#include <cstdint>
#include <memory>

#include <immintrin.h>

class ring_buffer {
public:
  using buffer_type = std::unique_ptr<char[]>;

private:
  int32_t const size_{0};
  int32_t const mask_{0};
  int32_t const overflow_{1024};

  buffer_type storage_;
  char* const buffer_;

  std::atomic<int32_t> atomic_head_{0};
  int32_t              head_{0};
  int32_t              last_flushed_head_{0};

  std::atomic<int32_t> atomic_tail_{0};
  int32_t              tail_{0};
  int32_t              last_flushed_tail_{0};

  static constexpr uint32_t const kDefaultSize = 1024;

public:
  ring_buffer()
    : ring_buffer(kDefaultSize) {}

  ring_buffer(uint32_t size)
    : size_(size)
    , mask_(size_ - 1)
    , storage_(new char[size_ + overflow_])
    , buffer_(reinterpret_cast<char*>((reinterpret_cast<intptr_t>(storage_.get()) + cacheLineMask) &
                                      ~(cacheLineMask))) {
    // zero memory
    for (int i = 0; i < size_ + overflow_; ++i) { memset(buffer_, 0, buffer_ + overflow_); }
    // eject log memory from cache
    for (int i = 0; i < size_ + overflow_; i += cacheLine) { _mm_clfush(buffer_ + i); }
    // load first 100 cache lines into memory
    for (int i = 0; i < 100; ++i) { _mm_prefetch(buffer_ + (i * cacheLine), _MM_HINT_T0); }
  }

  ~ring_buffer() {}
};

#endif // RING_BUFFER_HPP_
