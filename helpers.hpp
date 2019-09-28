#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "tl/expected.hpp"
#include "tl/optional.hpp"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace variant {

template <typename T, typename Variant>
std::optional<T> get_if(Variant const& variant) {
  if (T* ptr = std::get_if<T>(&variant)) {
    return *ptr;
  } else {
    return std::optional<T>();
  }
}

template <typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts...)->overloaded<Ts...>;

template <typename Variant, typename... Matchers>
auto match(Variant&& variant, Matchers&&... matchers) {
  return std::visit(overloaded{std::forward<Matchers>(matchers)...},
                    std::forward<Variant>(variant));
}

} // namespace variant

namespace algorithm {

// C++20 accumulate with std::move
template <typename InputIt, typename T, typename BinaryOperation = std::plus<T>>
T accumulate(InputIt first, InputIt last, T init, BinaryOperation op = std::plus<T>()) {
  for (; first != last; ++first) init = op(std::move(init), *first);
  return init;
}

template <typename InputIt, typename UnaryPredicate>
constexpr bool all_of(InputIt first, InputIt last, UnaryPredicate p) {
  return algorithm::accumulate(first, last, true, [&p](bool r, auto v) { return r & p(v); });
}

template <typename InputIt, typename UnaryPredicate>
constexpr bool any_of(InputIt first, InputIt last, UnaryPredicate p) {
  return algorithm::accumulate(first, last, false, [&p](bool r, auto v) { return r | p(v); });
}

template <typename T, typename F>
auto transform(std::optional<T> const& o, F f) -> decltype(std::optional{f(o.value())}) {
  if (o) {
    return {f(o.value())};
  } else {
    return {};
  }
}

} // namespace algorithm

namespace monad {

template <typename T, typename F>
auto mbind(std::optional<T> const& o, F f) -> decltype(f(o.value())) {
  if (o) {
    return f(o.value());
  } else {
    return {};
  }
}

template <typename F, typename R = std::result_of_t<F()>,
          typename E = tl::expected<R, std::exception_ptr>>
E mtry(F f) {
  try {
    return {f()};
  } catch (...) { return tl::unexpected{std::current_exception()}; }
}

template <typename F, typename G>
auto mcompose(F f, G g) {
  return [=](auto v) { return mbind(f(v), g); };
}

} // namespace monad

namespace iter {
template <typename C, typename Iter = decltype(std::begin(std::declval<C>())),
          typename = decltype(std::end(std::declval<C>()))>
constexpr auto enumerate(C&& c) {
  struct iterator {
    using iterator_category = std::input_iterator_tag;
    using value_type        = typename std::iterator_traits<Iter>::value_type;
    using difference_type   = typename std::iterator_traits<Iter>::difference_type;
    using pointer           = typename std::iterator_traits<Iter>::pointer;
    using reference         = typename std::iterator_traits<Iter>::reference;

    bool operator==(iterator const& o) const { return i_ == o.i_; }
    bool operator!=(iterator const& o) const { return !operator==(o); }

    auto operator*() const { return std::tie(c_, *i_); }

    // clang-format off
    auto operator++() { ++c_; ++i_; return *this; }
    // clang-format on

    std::size_t c_;
    Iter        i_;
  }; // struct iterator

  struct wrapper {
    auto begin() const { return iterator{0, std::begin(c_)}; }
    auto end() const { return iterator{0, std::end(c_)}; }

    C c_;
  }; // struct wrapper

  return wrapper{std::forward<C>(c)};
} // enumerate

namespace zip_detail {

struct deref {
  template <std::size_t I, typename T>
  decltype(auto) constexpr get(T& t) const {
    return *std::get<I>(t);
  } // get

  template <std::size_t... Indices, typename T>
  auto operator()(T& t, std::index_sequence<Indices...>) const {
    return std::make_tuple(get<Indices>(t)...);
  }
}; // struct deref

struct inc {
  template <typename... Ts>
  void _(Ts&&...) {} // needed?? to induce side-effect of operator++(int)

  // clang-format off
  template <typename T, std::size_t... Is>
  void operator()(T& v, std::index_sequence<Is...>) { _(++std::get<Is>(v)...); }
  // clang-format on
}; // struct inc

} // namespace zip_detail

template <typename... Cs>
constexpr auto zip(Cs&&... cs) {
  struct iterator {
    using T                 = std::tuple<decltype(std::begin(cs))...>;
    using iterator_category = std::input_iterator_tag;
    using value_type        = std::decay<decltype(
      zip_detail::deref()(std::declval<T&>(), std::make_index_sequence<std::tuple_size_v<T>>()))>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T*;
    using reference         = T&;

    iterator(T&& v)
      : v_(std::forward<T>(v)) {}

    bool operator==(iterator const& o) const {
      static_assert(std::tuple_size_v<decltype(v_)> == std::tuple_size_v<decltype(o.v_)>);
      return std::get<std::tuple_size_v<decltype(v_)> - 1>(v_) ==
             std::get<std::tuple_size_v<decltype(o.v_)> - 1>(o.v_);
    }
    bool operator!=(iterator const& o) const { return !operator==(o); }

    auto operator*() {
      auto&& is = std::make_index_sequence<std::tuple_size_v<decltype(v_)>>();
      return d_(v_, is);
    }

    auto operator++() {
      auto&& is = std::make_index_sequence<std::tuple_size_v<decltype(v_)>>();
      i_(v_, is);
      return *this;
    }

    T                 v_;
    zip_detail::deref d_{};
    zip_detail::inc   i_{};
  }; // struct iterator

  struct wrapper {
    iterator&& begin() const { return std::move(b_); }
    iterator&& end() const { return std::move(e_); }

    mutable iterator b_;
    mutable iterator e_;
  }; // struct wrapper

  return wrapper{std::make_tuple(std::begin(cs)...), std::make_tuple(std::end(cs)...)};
} // zip

} // namespace iter

#endif // HELPERS_HPP
