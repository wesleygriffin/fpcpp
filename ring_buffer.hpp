#ifndef RING_BUFFER_HPP_
#define RING_BUFFER_HPP_

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include <emmintrin.h>
#include <immintrin.h>
#if defined(__GNUC__)
#include <x86intrin.h>
#endif

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

  static constexpr uint32_t const kDefaultSize   = 1024;
  static constexpr uint64_t const kCacheLine     = 64;
  static constexpr uint64_t const kCacheLineMask = 63;

public:
  ring_buffer()
    : ring_buffer(kDefaultSize) {}

  ring_buffer(uint32_t size)
    : size_(size)
    , mask_(size_ - 1)
    , storage_(new char[size_ + overflow_])
    , buffer_(reinterpret_cast<char*>(
        (reinterpret_cast<intptr_t>(storage_.get()) + kCacheLineMask) & ~(kCacheLineMask))) {
    // zero memory
    for (int i = 0; i < size_ + overflow_; ++i) { memset(buffer_, 0, size_ + overflow_); }
    // eject log memory from cache
    for (int i = 0; i < size_ + overflow_; i += kCacheLine) { _mm_clflush(buffer_ + i); }
    // load first 100 cache lines into memory
    for (int i = 0; i < 100; ++i) { _mm_prefetch(buffer_ + (i * kCacheLine), _MM_HINT_T0); }
  }

  ~ring_buffer() {}

  int32_t head(int32_t diff = 0) noexcept { return (head_ + diff) & mask_; }
  int32_t tail(int32_t diff = 0) noexcept { return (tail_ + diff) & mask_; }

  char* pick_produce(int32_t size = 0) {
    auto ft = atomic_tail_.load(std::memory_order_acquire);
    return (head_ - ft > size_ - (128 + size)) ? nullptr : buffer_ + head();
  }

  char* pick_consume(int32_t size = 0) {
    auto fh = atomic_head_.load(std::memory_order_acquire);
    return fh - (tail_ + size) < 1 ? nullptr : buffer_ + tail();
  }

  void produce(uint32_t size) noexcept { head_ += size; }
  void consume(int32_t size) noexcept { tail_ += size; }

  uint32_t clfu_count{0};
  void cleanup(int32_t& last, int32_t offset) {
    int32_t l_diff = last - (last & kCacheLineMask);
    int32_t c_diff = offset - (offset & kCacheLineMask);
    while (c_diff > l_diff) {
#if defined(_MSC_VER)
      _mm_clflush(buffer_ + (l_diff & mask_));
#else
      _mm_clflushopt(buffer_ + (l_diff & mask_));
#endif
      l_diff += kCacheLine;
      last = l_diff;
      ++clfu_count;
    }
  }

  void cleanup_consume() {
    cleanup(last_flushed_tail_, tail_);
    atomic_tail_.store(tail_, std::memory_order_release);
  }

  void cleanup_produce() {
    cleanup(last_flushed_head_, head_);
    _mm_prefetch(buffer_ + head(kCacheLine * 12), _MM_HINT_T0);
    atomic_head_.store(head_, std::memory_order_release);
  }

  char* get() { return buffer_; }
};

#endif // RING_BUFFER_HPP_
