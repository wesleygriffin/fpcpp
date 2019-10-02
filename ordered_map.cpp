#include <cstddef>
#include <utility>

template <typename Key, typename Value, typename Compare = std::less<Key>>
class ordered_map {
public:
  using key_type        = Key;
  using mapped_type     = Value;
  using value_type      = std::pair<key_type const, mapped_type>;
  using key_compare     = Compare;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = value_type const&;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;

private:
  struct node;
  void clear(node* node);

  node*     root_{nullptr};
  size_type size_{0};

public:
  constexpr size_type size() const noexcept { return size_; }
  constexpr bool      empty() const noexcept { return root_ == nullptr; }

  void clear() { clear(root_); }

  friend void swap(ordered_map& a, ordered_map& b) noexcept {
    std::swap(a.root_, b.root_);
    std::swap(a.size_, b.size_);
  }

  constexpr ordered_map() noexcept = default;
  ordered_map(ordered_map const& other);
  ordered_map(ordered_map&& other) { swap(*this, other); }
  ordered_map& operator=(ordered_map other) {
    swap(*this, other);
    return *this;
  }
  ~ordered_map() { clear(); }

private:
  enum class colors { red, black };

  struct node {
    Key    key;
    Value  value;
    node*  left{nullptr};
    node*  right{nullptr};
    colors color{colors::black};

    node(Key k, Value v)
      : key{std::move(k)}
      , value{std::move(v)} {}
  }; // struct node
};   // class ordered_map

template <typename Key, typename Value, typename Compare>
void ordered_map<Key, Value, Compare>::clear(node* node) {
  if (!node) return;
  clear(node->left);
  clear(node->right);
  delete node;
}

#include "gtest/gtest.h"

TEST(ordered_map, construction) {
  ordered_map<int, std::string> m;
}
