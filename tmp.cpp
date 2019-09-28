#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <vector>

template <typename T>
class error;

template <typename T1, typename T2>
struct my_is_same : std::false_type {};

template <typename T>
struct my_is_same<T, T> : std::true_type {};

template <typename T1, typename T2>
inline constexpr bool my_is_same_v = my_is_same<T1, T2>::value;

template <typename T>
struct my_remove_reference {
  using type = T;
};

template <typename T>
struct my_remove_reference<T&> {
  using type = T;
};

template <typename T>
struct my_remove_reference<T&&> {
  using type = T;
};

template <typename T>
using my_remove_reference_t = typename my_remove_reference<T>::type;

template <typename T>
struct my_remove_cv {
  using type = T;
};

template <typename T>
struct my_remove_cv<T const> {
  using type = T;
};

template <typename T>
struct my_remove_cv<T volatile> {
  using type = T;
};

template <typename T>
struct my_remove_cv<T const volatile> {
  using type = T;
};

template <typename T>
using my_remove_cv_t = typename my_remove_cv<T>::type;

template <typename T>
using contained_type_t =
  std::remove_cv_t<std::remove_reference_t<decltype(*begin(std::declval<T>()))>>;

template <typename C, typename R = typename C::value_type>
R sum_collection(C const& c) {
  return std::accumulate(begin(c), end(c), R{});
}

template <typename C, typename R = contained_type_t<C>>
R sum_iterable(C const& c) {
  return std::accumulate(begin(c), end(c), R{});
}

template <typename C, typename = std::void_t<>>
struct has_value_type : std::false_type {};

template <typename C>
struct has_value_type<C, std::void_t<typename C::value_type>> : std::true_type {};

template <typename C>
inline constexpr bool has_value_type_v = has_value_type<C>::value;

template <typename C, typename = std::void_t<>>
struct is_iterable : std::false_type {};

template <typename C>
struct is_iterable<
  C, std::void_t<decltype(*begin(std::declval<C>())), decltype(end(std::declval<C>()))>>
  : std::true_type {};

template <typename C>
inline constexpr bool is_iterable_v = is_iterable<C>::value;

template <typename C>
auto sum(C const& c) {
  if constexpr (has_value_type<C>()) {
    using R = typename C::value_type;
    return std::accumulate(begin(c), end(c), R{});
  } else if constexpr (is_iterable<C>()) {
    using R = std::remove_cv_t<std::remove_reference_t<decltype(*begin(c))>>;
    return std::accumulate(begin(c), end(c), R{});
  } else {
  }
}

int main() {
  std::vector<int> v{1, 2, 3, 4, 5};
  auto s = sum(v);
  static_assert(my_is_same_v<decltype(s), int>);
  assert(s == 15);
}
