/////
//
// Sum of a list of Ts is the first element plus the sum of the rest of the list
// Sum of an empty list is 0
//
/////

template <class T>
T sum() {
  return T(0);
}

template <class T, class... Ts>
T sum(T t, Ts... ts) {
  return t + sum<T>(ts...);
}

#include <iostream>

int main() {
  std::cout << "sum: " << sum(1, 2, 3, 4) << "\n";
  return EXIT_SUCCESS;
}