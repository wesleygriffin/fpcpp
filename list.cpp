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
  void     push_front(node* new_head);
  void     push_back(node* new_tail);
  iterator insert(const_iterator pos, node* new_node);

  node*     head_{nullptr};
  node*     tail_{nullptr};
  size_type size_{0};

public:
  constexpr size_type size() const noexcept { return size_; }
  constexpr bool      empty() const noexcept { return head_ == nullptr; }

  reference       front() { return head_->value; }
  const_reference front() const { return head_->value; }

  reference       back() { return tail_->value; }
  const_reference back() const { return tail_->value; }

  iterator       begin() noexcept { return {head_}; }
  const_iterator begin() const noexcept { return {head_}; }
  const_iterator cbegin() const noexcept { return {head_}; }

  iterator       end() noexcept { return {nullptr}; }
  const_iterator end() const noexcept { return {nullptr}; }
  const_iterator cend() const noexcept { return {nullptr}; }

  void clear() noexcept(noexcept(~T()));

  void push_front(T const& value) { push_front(new node{value}); }
  void push_front(T&& value) { push_front(new node{std::forward<T>(value)}); }

  void push_back(T const& value) { push_back(new node{value}); }
  void push_back(T&& value) { push_back(new node{std::forward<T>(value)}); }

  void pop_front();
  void pop_back();

  iterator insert(const_iterator pos, T const& value) { return insert(pos, new node{value}); }
  iterator insert(const_iterator pos, T&& value) {
    return insert(pos, new node{std::forward<T>(value)});
  }

  iterator erase(const_iterator pos);

  friend void swap(linked_list& a, linked_list& b) noexcept {
    using std::swap;
    swap(a.head_, b.head_);
    swap(a.tail_, b.tail_);
    swap(a.size_, b.size_);
  }

  constexpr linked_list() noexcept = default;
  linked_list(linked_list const& other);
  linked_list(linked_list&& other);
  linked_list& operator=(linked_list other);
#if __GNUC__ < 9
  ~linked_list() noexcept(noexcept(~T())) { clear(); }
#else
  ~linked_list() noexcept(noexcept(clear())) { clear(); }
#endif

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

    iterator& operator--() noexcept {
      node_ = node_->prev;
      return *this;
    }

    iterator operator++(int) {
      iterator prev{*this};
      node_ = node_->next;
      return prev;
    }

    iterator operator--(int) {
      iterator prev{*this};
      node_ = node_->prev;
      return prev;
    }

    iterator(node* node)
      : node_{node} {}

  private:
    node* node_;
    template <class U>
    friend class linked_list;
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

    const_iterator operator--() {
      node_ = node_->prev;
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator prev{*this};
      node_ = node_->next;
      return prev;
    }

    const_iterator operator--(int) {
      const_iterator prev{*this};
      node_ = node_->prev;
      return prev;
    }

    const_iterator(node* node)
      : node_{node} {}

  private:
    node* node_;
    template <class U>
    friend class linked_list;
  }; // class const_iterator

private:
  struct node {
    T     value;
    node* next{nullptr};
    node* prev{nullptr};

    node(T const& value)
      : value{value} {}
    node(T&& value)
      : value{std::forward<T>(value)} {}
  }; // struct node
};   // class linked_list

template <typename T>
void linked_list<T>::clear() noexcept(noexcept(~T())) {
  auto node = head_;
  while (node) {
    auto next = node->next;
    delete node;
    node = next;
  }
  head_ = nullptr;
  tail_ = nullptr;
  size_ = 0;
} // linked_list<T>::clear

template <typename T>
void linked_list<T>::push_front(node* new_head) {
  if (!head_ && !tail_) { // empty list
    head_ = tail_ = new_head;
  } else {
    new_head->next = head_;
    head_->prev    = new_head;
    head_          = new_head;
  }

  size_ += 1;
} // linked_list<T>::push_front

template <typename T>
void linked_list<T>::push_back(node* new_tail) {
  if (!tail_ && !head_) {
    tail_ = head_ = new_tail;
  } else {
    new_tail->prev = tail_;
    tail_->next    = new_tail;
    tail_          = new_tail;
  }

  size_ += 1;
}

template <typename T>
typename linked_list<T>::iterator linked_list<T>::insert(const_iterator pos, node* new_node) {
  if (pos == end()) {
    push_back(new_node);
  } else if (pos == begin()) {
    push_front(new_node);
  } else {
    new_node->next = pos.node_;
    new_node->prev = pos.node_->prev;
    if (pos.node_->prev) pos.node_->prev->next = new_node;
    pos.node_->prev = new_node;
    if (pos.node_ == head_) head_ = new_node;
    size_ += 1;
  }

  return {new_node};
} // linked_list<T>::insert

template <typename T>
typename linked_list<T>::iterator linked_list<T>::erase(const_iterator pos) {
  auto next = pos.node_->next;

  if (pos.node_->prev) {
    pos.node_->prev->next = pos.node_->next;
  } else {
    head_ = next;
  }

  if (pos.node_->next) pos.node_->next->prev = pos.node_->prev;
  delete pos.node_;
  size_ -= 1;
  return next;
} // linked_list<T>::erase

template <typename T>
void linked_list<T>::pop_front() {
  auto node   = head_;
  head_       = head_->next;
  head_->prev = nullptr;
  size_ -= 1;
  delete node;
}

template <typename T>
void linked_list<T>::pop_back() {
  auto node   = tail_;
  tail_       = tail_->prev;
  tail_->next = nullptr;
  size_ -= 1;
  delete node;
}

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
    if (!other_node->next) tail_ = nullptr;
    other_node = other_node->next;
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
  ASSERT_EQ(ints.back(), 32);

  auto it = ints.begin();
  ASSERT_EQ(*it, 48);
  ++it;
  ASSERT_EQ(*it, 32);
  ++it;
  ASSERT_EQ(it, ints.end());
}

TEST(list, push_back) {
  linked_list<int> ints;
  ints.push_back(32);
  ints.push_back(48);
  ASSERT_EQ(ints.size(), 2);
  ASSERT_EQ(ints.front(), 32);
  ASSERT_EQ(ints.back(), 48);

  auto it = ints.begin();
  ASSERT_EQ(*it, 32);
  ++it;
  ASSERT_EQ(*it, 48);
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

TEST(list, clear) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_front(i);
  ASSERT_FALSE(ints.empty());
  ASSERT_EQ(ints.size(), 10);

  ints.clear();
  ASSERT_TRUE(ints.empty());
  ASSERT_EQ(ints.size(), 0);
}

TEST(list, pop_front_back) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_back(i);
  ASSERT_FALSE(ints.empty());
  ASSERT_EQ(ints.size(), 10);

  ints.pop_front();
  int j = 1;
  for (auto&& i : ints) ASSERT_EQ(i, j++);

  ints.pop_back();
  j = 1;
  for (auto&& i : ints) ASSERT_EQ(i, j++);
}

TEST(list, insert) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_back(i);
  ASSERT_EQ(ints.size(), 10);

  auto p = ints.insert(ints.cbegin(), -1);
  ASSERT_EQ(ints.size(), 11);
  ASSERT_EQ(p, ints.begin());

  p = ints.insert(ints.cend(), 10);
  ASSERT_EQ(ints.size(), 12);
  ASSERT_EQ(*p, 10);

  int j = -1;
  for (auto&& i : ints) ASSERT_EQ(i, j++);

  auto cp = ints.cbegin();
  for (int i = 0; i < 6; ++i) ++cp;
  p = ints.insert(cp, 42);
  ASSERT_EQ(ints.size(), 13);
  ASSERT_EQ(*p, 42);

  auto it = ints.begin();
  for (int k = 0; k < ints.size(); ++k, ++it) {
    if (k < 6) {
      ASSERT_EQ(*it, k - 1);
    } else if (k == 6) {
      ASSERT_EQ(*it, 42);
    } else {
      ASSERT_EQ(*it, k - 2);
    }
  }
}

TEST(list, erase) {
  linked_list<int> ints;
  for (int i = 0; i < 10; ++i) ints.push_back(i);
  ASSERT_EQ(ints.size(), 10);

  auto p = ints.insert(ints.cbegin(), -1);
  ASSERT_EQ(ints.size(), 11);
  ASSERT_EQ(p, ints.begin());

  p = ints.insert(ints.cend(), 10);
  ASSERT_EQ(ints.size(), 12);
  ASSERT_EQ(*p, 10);

  int j = -1;
  for (auto&& i : ints) ASSERT_EQ(i, j++);

  auto cp = ints.cbegin();
  for (int i = 0; i < 6; ++i) ++cp;
  p = ints.insert(cp, 42);
  ASSERT_EQ(ints.size(), 13);
  ASSERT_EQ(*p, 42);

  auto it = ints.begin();
  for (int k = 0; k < ints.size(); ++k, ++it) {
    if (k < 6) {
      ASSERT_EQ(*it, k - 1);
    } else if (k == 6) {
      ASSERT_EQ(*it, 42);
    } else {
      ASSERT_EQ(*it, k - 2);
    }
  }

  p = ints.erase(--cp);
  ASSERT_EQ(ints.size(), 12);
  ASSERT_EQ(*p, 5);

  j = -1;
  for (auto&& i : ints) ASSERT_EQ(i, j++);

  p = ints.erase(ints.cbegin());
  ASSERT_EQ(ints.size(), 11);
  ASSERT_EQ(*p, 0);

  j = 0;
  for (auto&& i : ints) ASSERT_EQ(i, j++);
}
