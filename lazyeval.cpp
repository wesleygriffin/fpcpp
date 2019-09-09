#include <mutex>
#include <optional>

template <typename F>
class lazy_val {
  F                                               computation_;
  mutable std::optional<decltype(computation_())> cache_;
  mutable std::once_flag                          once_;

public:
  lazy_val(F computation)
    : computation_(computation) {}

  operator decltype(computation_()) const&() const {
    std::call_once(once_, [this] { cache_ = computation_(); });
    return *cache_;
  }
};

template <typename F>
lazy_val(F)->lazy_val<F>;

#include <iostream>

int main() {
  auto add24 = lazy_val([] {
    std::cout << "add\n";
    return 2 + 4;
  });

  int sum = add24;
  std::cout << "sum: " << sum << "\n";
  std::cout << "2nd: " << add24 << "\n";
}