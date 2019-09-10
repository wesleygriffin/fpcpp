#include <memory>

template <typename T>
class immutable_linked_list {
public:

private:
  struct node {
    T data;
    std::shared_ptr<node> tail{};

    ~node() {
      // not thread-safe
      // iterate down tail and delete if unique instead of recursively
      // calling destructors of tail
      auto next = std::move(tail);
      while (next) {
        if (!next.unique()) break;
        decltype(next.tail) tail; // tail.tail == nullptr
        swap(tail, next->tail);
        next.reset(); // next.tail == nullptr, so no recursion
        next = std::move(tail);
      }
    }
  };

  std::shared_ptr<node> head_;
};

int main() {
}