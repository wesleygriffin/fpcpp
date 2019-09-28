#include <chrono>
#include <string>

class person {
public:
  enum class genders { female, male, other };

  person(std::string name, const std::string& surname, genders gender, int age)
    : name_(name)
    , surname_(surname)
    , gender_(gender)
    , age_(age) {}

  std::string name() const { return name_; }
  std::string surname() const { return surname_; }
  genders gender() const { return gender_; }
  int age() const { return age_; }

  //void time_passed(std::chrono::duration const& amount) {
  //}

private:
  std::string name_;
  std::string surname_;
  genders     gender_;
  int         age_;
};

int main() {
  person martha("Martha", "Jones", person::genders::female, 30);
}
