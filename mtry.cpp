#include "helpers.hpp"

#include <iostream>

bool fail() { return true; }

int main() {
  auto result = monad::mtry([=] {
    if (fail()) {
      throw std::runtime_error("fail");
    }
    return 1;
  });

  if (!result) {
    try {
      std::rethrow_exception(result.error());
    } catch (std::exception const& e) {
      std::cout << "error: " << e.what() << "\n";
    }
  } else {
    std::cout << "suscess: " << result.value() << "\n";
  }
}