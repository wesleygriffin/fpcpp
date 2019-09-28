/*
expected - An implementation of std::expected assuming a C++17 compiler based on
expected writting by Simon Brand (simonrbrand@gmail.com, @TartanLlama)
<https://github.com/TartanLlama/expected>.

To the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the
public domain worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication
along with this software. If not, see
<http://creativecommons.org/publicdomain/zero/1.0/>.
*/


#ifndef EXPECTED_HPP_
#define EXPECTED_HPP_

#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

namespace fktd {

// Start by forward-declaring the expected class.
template <typename T, typename E> class expected;

/*
\brief Holds a value for the unexpected case.
*/
template <class E>
class unexpected {
public:
  static_assert(!std::is_same<E, void>::value, "E must not be void");

  unexpected() = delete;
  constexpr explicit unexpected(const E& e)
    : m_val(e) {}

  constexpr explicit unexpected(E&& e)
    : m_val(std::move(e)) {}

  constexpr const E&  value() const& { return m_val; }
  constexpr E&        value() & { return m_val; }
  constexpr E&&       value() && { return std::move(m_val); }
  constexpr const E&& value() const&& { return std::move(m_val); }

private:
  E m_val;
}; // class unexpected

/*
\brief Class template argument deduction for unexpected
*/
template <class E>
unexpected(E) -> unexpected<E>;

} // namespace fktd

#endif // EXPECTED_HPP_
