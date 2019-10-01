#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
class error;

template <class It>
void mergesort(It first, It last) {
  auto merge = [](It lo, It md, It hi) {
    using std::swap;
    std::vector<typename std::iterator_traits<It>::value_type> scratch(std::distance(lo, hi));
    auto&& end = scratch.end();
    for (auto lit = lo, rit = md, it = scratch.begin(); it != end; ++it) {
      if (lit == md) {
        swap(*rit, *it);
        ++rit;
      } else if (rit == hi) {
        swap(*lit, *it);
        ++lit;
      } else if (*lit <= *rit) {
        swap(*lit, *it);
        ++lit;
      } else {
        swap(*rit, *it);
        ++rit;
      }
    }
  };

  auto const dist = std::distance(first, last);
  if (dist > 1) {
    auto middle = first + (dist / 2);
    mergesort(first, middle);
    mergesort(middle, last);
    merge(first, middle, last);
  }
}

template <class It>
void quicksort(It first, It last) {
  // lo and hi are both deferenced and thus hi must not be one-past-the-end
  auto partition = [](It lo, It hi) {
    using std::swap;
    // swap the middle value of the range into hi, results in a better pivot
    auto md = lo + (std::distance(lo, hi) / 2);
    if (*md < *lo) swap(*lo, *md);
    if (*hi < *lo) swap(*lo, *hi);
    if (*md < *hi) swap(*md, *hi);

    auto pivot = *hi;
    auto cut = lo;

    for (auto it = lo; it < hi; ++it) {
      if (*it < pivot) {
        swap(*cut, *it);
        ++cut;
      }
    }

    swap(*cut, *hi);
    return cut;
  };

  if (first < last) {
    auto&& cut = partition(first, last - 1);
    quicksort(first, cut);
    quicksort(cut + 1, last);
  }
}

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

TEST(sorting, quick) {
	std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);

  std::vector<int> is(1000);
  std::generate(is.begin(), is.end(), [&] { return dis(gen); });
  ASSERT_FALSE(std::is_sorted(is.begin(), is.end()));

  quicksort(is.begin(), is.end());
  ASSERT_TRUE(std::is_sorted(is.begin(), is.end()));
}

TEST(sorting, merge) {
	std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);

  std::vector<int> is(1000);
  std::generate(is.begin(), is.end(), [&] { return dis(gen); });
  ASSERT_FALSE(std::is_sorted(is.begin(), is.end()));

  mergesort(is.begin(), is.end());
  ASSERT_TRUE(std::is_sorted(is.begin(), is.end()));
}
