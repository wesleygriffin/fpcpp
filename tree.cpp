#include <iterator>
#include <utility>

template <typename K, typename V>
class binary_tree {
public:
  using key_type        = K;
  using value_type      = V;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = V&;
  using const_reference = V const&;
  using pointer         = V*;
  using const_pointer   = V const*;

  enum class traversals { pre_order, in_order, post_order };

private:
  struct node;
  void clear(node* node);

  template <typename F>
  void traverse(node* node, traversals traversal, F callback);

  node*     root_{nullptr};
  size_type size_{0};

public:
  constexpr size_type size() const noexcept { return size_; }
  constexpr bool      empty() const noexcept { return root_ == nullptr; }

  void clear();

  void insert(K const& key, V value);

  template <typename F>
  void traverse(traversals traversal, F callback) {
    traverse(root_, traversal, callback);
  }

  friend void swap(binary_tree& a, binary_tree& b) noexcept {
    using std::swap;
    swap(a.root_, b.root_);
    swap(a.size_, b.size_);
  }

  constexpr binary_tree() noexcept = default;
  binary_tree(binary_tree const& other);
  binary_tree(binary_tree&& other);
  binary_tree& operator=(binary_tree other);
  ~binary_tree() { clear(); }

private:
  struct node {
    K     key;
    V     value;
    node* left{nullptr};
    node* right{nullptr};

    node(K k, V v)
      : key{std::move(k)}
      , value{std::move(v)} {}
  }; // struct node
};   // class binary_tree

template <typename K, typename V>
void binary_tree<K, V>::clear(node* node) {
  if (!node) return;
  clear(node->left);
  clear(node->right);
  delete node;
} // binary_tree<K, V>::clear

template <typename K, typename V>
template <typename F>
void binary_tree<K, V>::traverse(node* node, traversals traversal, F callback) {
  if (!node) return;
  if (traversal == traversals::pre_order) callback(node->key, node->value);
  traverse(node->left, traversal, callback);
  if (traversal == traversals::in_order) callback(node->key, node->value);
  traverse(node->right, traversal, callback);
  if (traversal == traversals::post_order) callback(node->key, node->value);
} // binary_tree<K, V>::traverse

template <typename K, typename V>
void binary_tree<K, V>::insert(K const& key, V value) {
  auto curr = &root_;
  while (*curr) {
    auto curr_key = (*curr)->key;
    if (curr_key == key) {
      (*curr)->value = std::move(value);
      return;
    } else if (curr_key > key) {
      curr = &((*curr)->left);
    } else {
      curr = &((*curr)->right);
    }
  }

  *curr = new node{key, std::move(value)};
  size_ += 1;
} // binary_tree<K, V>::insert

template <typename K, typename V>
void binary_tree<K, V>::clear() {
  clear(root_);
  root_ = nullptr;
  size_ = 0;
} // binary_tree<K, V>::clear

#include "gtest/gtest.h"
#include <string>

TEST(tree, construction) {
  binary_tree<int, std::string> tree;
  ASSERT_TRUE(tree.empty());
  ASSERT_EQ(tree.size(), 0);
}

TEST(tree, insertion) {
  binary_tree<int, std::string> tree;
  ASSERT_TRUE(tree.empty());
  ASSERT_EQ(tree.size(), 0);

  tree.insert(0, "zero");
  ASSERT_FALSE(tree.empty());
  ASSERT_EQ(tree.size(), 1);

  tree.insert(1, "one");
  ASSERT_EQ(tree.size(), 2);
}

TEST(tree, traversal) {
  binary_tree<int, std::string> tree;
  tree.insert(1, "one");
  tree.insert(0, "zero");
  tree.insert(2, "two");

  int i = 0;
  tree.traverse(decltype(tree)::traversals::pre_order, [&i](auto&& key, auto&& val) {
    switch (i) {
    case 0:
      ASSERT_EQ(key, 1);
      ASSERT_EQ(val, "one");
      break;
    case 1:
      ASSERT_EQ(key, 0);
      ASSERT_EQ(val, "zero");
      break;
    case 2:
      ASSERT_EQ(key, 2);
      ASSERT_EQ(val, "two");
      break;
    }
    i += 1;
  });

  i = 0;
  tree.traverse(decltype(tree)::traversals::in_order, [&i](auto&& key, auto&& val) {
    switch (i) {
    case 0:
      ASSERT_EQ(key, 0);
      ASSERT_EQ(val, "zero");
      break;
    case 1:
      ASSERT_EQ(key, 1);
      ASSERT_EQ(val, "one");
      break;
    case 2:
      ASSERT_EQ(key, 2);
      ASSERT_EQ(val, "two");
      break;
    }
    i += 1;
  });

  i = 0;
  tree.traverse(decltype(tree)::traversals::post_order, [&i](auto&& key, auto&& val) {
    switch (i) {
    case 0:
      ASSERT_EQ(key, 0);
      ASSERT_EQ(val, "zero");
      break;
    case 1:
      ASSERT_EQ(key, 2);
      ASSERT_EQ(val, "two");
      break;
    case 2:
      ASSERT_EQ(key, 1);
      ASSERT_EQ(val, "one");
      break;
    }
    i += 1;
  });
}
