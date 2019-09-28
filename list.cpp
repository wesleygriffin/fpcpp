#include <iterator>
#include <utility>

template <typename T>
class linked_list {
public:
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = T&;
  using const_reference = T const&;
  using pointer         = T*;
  using const_pointer   = T const*;

  class iterator;
  class const_iterator;

private:
  struct node;
  void push_front(node* new_head);

  node*     head_{nullptr};
  size_type size_{0};

public:
  constexpr size_type size() const noexcept { return size_; }
  constexpr bool      empty() const noexcept { return head_ == nullptr; }

  reference       front() { return head_->value; }
  const_reference front() const { return head_->value; }

  iterator       begin() noexcept { return {head_}; }
  const_iterator begin() const noexcept { return {head_}; }
  const_iterator cbegin() const noexcept { return {head_}; }

  iterator       end() noexcept { return {nullptr}; }
  const_iterator end() const noexcept { return {nullptr}; }
  const_iterator cend() const noexcept { return {nullptr}; }

  void clear() noexcept(noexcept(~T()));

  void push_front(T const& value) { push_front(new node{value}); }
  void push_front(T&& value) { push_front(new node{std::forward<T>(value)}); }

  friend void swap(linked_list& a, linked_list& b) noexcept {
    std::swap(a.head_, b.head_);
    std::swap(a.size_, b.size_);
  }

  constexpr linked_list() noexcept = default;
  linked_list(linked_list const& other);
  linked_list(linked_list&& other);
  linked_list& operator=(linked_list other);
  ~linked_list() noexcept(noexcept(clear())) { clear(); }

  class iterator {
  public:
    using value_type        = linked_list::value_type;
    using difference_type   = linked_list::difference_type;
    using reference         = linked_list::reference;
    using pointer           = linked_list::pointer;
    using iterator_category = std::forward_iterator_tag;

    bool operator==(iterator const& other) const noexcept { return node_ == other.node_; }
    bool operator!=(iterator const& other) const noexcept { return node_ != other.node_; }

    reference operator*() { return node_->value; }
    pointer   operator->() { return &node_->value; }

    iterator& operator++() noexcept {
      node_ = node_->next;
      return *this;
    }

    iterator operator++(int) {
      iterator prev{*this};
      node_ = node_->next;
      return prev;
    }

    iterator(node* node)
      : node_{node} {}

  private:
    node* node_;
  }; // class iterator

  class const_iterator {
  public:
    using value_type        = linked_list::value_type;
    using difference_type   = linked_list::difference_type;
    using reference         = linked_list::const_reference;
    using pointer           = linked_list::const_pointer;
    using iterator_category = std::forward_iterator_tag;

    bool operator==(iterator const& other) const noexcept { return node_ == other.node_; }
    bool operator!=(iterator const& other) const noexcept { return node_ != other.node_; }

    reference operator*() { return node_->value; }
    pointer   operator->() { return &node_->value; }

    const_iterator& operator++() noexcept {
      node_ = node_->next;
      return *this;
    }

    const_iterator operator++(int) {
      iterator prev{*this};
      node_ = node_->next;
      return prev;
    }

    const_iterator(node* node)
      : node_{node} {}

  private:
    node* node_;
  }; // class const_iterator

private:
  struct node {
    T     value;
    node* next{nullptr};

    node(T const& value)
      : value{value} {}
    node(T&& value)
      : value{std::forward<T>(value)} {}
  }; // struct node
}; // class linked_list

template <typename T>
void linked_list<T>::clear() noexcept(noexcept(~T())) {
  auto node = head_;
  while (node) {
    auto next = node->next;
    delete node;
    node = next;
  }
} // linked_list<T>::clear

template <typename T>
void linked_list<T>::push_front(node* new_head) {
  new_head->next = head_;
  head_          = new_head;
  size_ += 1;
} // linked_list<T>::push_front

template <typename T>
linked_list<T>::linked_list(linked_list&& other)
  : linked_list{} {
  swap(*this, other);
} // linked_list<T>::linked_list

template <typename T>
linked_list<T>::linked_list(linked_list const& other) {
  if (!other.head_) return;

  head_           = new node{other.head_->value};
  auto this_node  = head_;
  auto other_node = other.head_->next;

  while (other_node) {
    auto new_node   = new node{other_node->value};
    this_node->next = new_node;
    this_node       = new_node;
    other_node      = other_node->next;
  }

  size_ = other.size_;
} // linked_list<T>::linked_list

template <typename T>
linked_list<T>& linked_list<T>::operator=(linked_list other) {
  swap(*this, other);
  return *this;
} // linked_list<T>::operator=

#include "gtest/gtest.h"

TEST(list, construction) {
  linked_list<int> ints;
  ASSERT_TRUE(ints.empty());
  ASSERT_EQ(ints.size(), 0);
}

TEST(list, push_front) {
  linked_list<int> ints;
  ints.push_front(32);
  ints.push_front(48);
  ASSERT_EQ(ints.size(), 2);
  ASSERT_EQ(ints.front(), 48);

  auto it = ints.begin();
  ASSERT_EQ(*it, 48);
  ++it;
  ASSERT_EQ(*it, 32);
  ++it;
  ASSERT_EQ(it, ints.end());
}

TEST(list, copy_construction) {
  linked_list<int> ints;
  ints.push_front(32);
  ints.push_front(48);

  linked_list<int> ints2{ints};

  ASSERT_FALSE(ints.empty());
  ASSERT_EQ(ints.size(), 2);
  ASSERT_EQ(ints.front(), 48);

  ASSERT_FALSE(ints2.empty());
  ASSERT_EQ(ints2.size(), 2);
  ASSERT_EQ(ints2.front(), 48);

  auto it = ints2.begin();
  ASSERT_EQ(*it, 48);
  ++it;
  ASSERT_EQ(*it, 32);
  ++it;
  ASSERT_EQ(it, ints2.end());
}

TEST(list, move_construction) {
  linked_list<int> ints;
  ints.push_front(32);
  ints.push_front(48);

  linked_list<int> ints2{std::move(ints)};

  ASSERT_TRUE(ints.empty());

  ASSERT_FALSE(ints2.empty());
  ASSERT_EQ(ints2.size(), 2);
  ASSERT_EQ(ints2.front(), 48);

  auto it = ints2.begin();
  ASSERT_EQ(*it, 48);
  ++it;
  ASSERT_EQ(*it, 32);
  ++it;
  ASSERT_EQ(it, ints2.end());
}

TEST(list, iterators) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_front(i);
  int j = 9;
  for (auto&& i : ints) ASSERT_EQ(i, j--);
}

TEST(list, assignment) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_front(i);

  linked_list<int> ints2;
  ints2 = ints;
  int j = 9;
  for (auto&& i : ints2) ASSERT_EQ(i, j--);

  linked_list<int> ints3;
  ints3 = std::move(ints);
  ASSERT_TRUE(ints.empty());
  j = 9;
  for (auto&& i : ints3) ASSERT_EQ(i, j--);
}
