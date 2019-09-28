#include <tuple>
#include <type_traits>

template <typename F, typename... CapturedArgs>
class curried {
public:
  curried(F function, CapturedArgs... args)
    : function_(function)
    , captured_(capture_by_copy(std::move(args)...)) {}

  curried(F function, std::tuple<CapturedArgs...> args)
    : function_(function)
    , captured_(std::move(args)) {}

  template <typename... Args>
  auto operator()(Args... args) const {
    auto all_args = std::tuple_cat(captured_, capture_by_copy(std::forward<Args>(args)...));
    if constexpr (std::is_invocable_v<F, CapturedArgs..., Args...>) {
      return std::apply(function_, all_args);
    } else {
      return curried<F, CapturedArgs..., Args...>(function_, all_args);
    }
  }

private:
  F                           function_;
  std::tuple<CapturedArgs...> captured_;

  template <typename... Args>
  static auto capture_by_copy(Args&&... args) {
    return std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
  }
}; // class curried

#include <ostream>
#include <string>

class person_t {
public:
  enum class genders { female, male, other };

  enum class output_formats { name_only, full_name };

  person_t()
    : m_name("John")
    , m_surname("Doe")
    , m_gender(genders::other) {}

  person_t(std::string name, genders gender, int age = 0)
    : m_name(name)
    , m_surname("Doe")
    , m_gender(gender)
    , m_age(age) {}

  person_t(std::string name, const std::string& surname, genders gender, int age = 0)
    : m_name(name)
    , m_surname(surname)
    , m_gender(gender)
    , m_age(age) {}

  std::string name() const { return m_name; }

  std::string surname() const { return m_surname; }

  genders gender() const { return m_gender; }

  int age() const { return m_age; }

  void print(std::ostream& out, output_formats format) const {
    if (format == person_t::output_formats::name_only) {
      out << name() << '\n';

    } else if (format == person_t::output_formats::full_name) {
      out << name() << ' ' << surname() << '\n';
    }
  }

private:
  std::string m_name;
  std::string m_surname;
  genders     m_gender;
  int         m_age;
};

void print_person(person_t const& person, std::ostream& out, person_t::output_formats format) {
  person.print(out, format);
}

#include <iostream>

int main() {
  person_t martha("Martha", "Jones", person_t::genders::female, 30);
  auto print_person_cd = curried(print_person);
  print_person_cd(std::cref(martha))(std::ref(std::cout))(person_t::output_formats::full_name);
  auto print_person_cd2 = curried(print_person, std::cref(martha));
  print_person_cd2(std::ref(std::cout), person_t::output_formats::name_only);
}
