#include "tl/optional.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

template <class C, class T = typename C::value_type>
tl::optional<C> sort_by_frequency(C& collection) {
  std::sort(std::begin(collection), std::end(collection),
            [](auto&& p1, auto&& p2) { return p1.first > p2.first; });
  return collection;
}

template <class C,
          typename P1 = std::remove_cv_t<typename C::value_type::first_type>,
          typename P2 = std::remove_cv_t<typename C::value_type::second_type>>
tl::optional<std::vector<std::pair<P2, P1>>>
reverse_pairs(C const& collection) {
  std::vector<std::pair<P2, P1>> result(std::size(collection));
  std::transform(std::begin(collection), std::end(collection),
                 std::begin(result), [](std::pair<P1, P2> const& p) {
                   return std::make_pair(p.second, p.first);
                 });
  return result;
} // reverse_pairs

template <class C, class T = typename C::value_type>
tl::optional<std::unordered_map<T, std::size_t>> count(C const& collection) {
    std::unordered_map<T, std::size_t> counts;
    for (auto&& value : collection) counts[value] += 1;
    return counts;
} // count

tl::optional<std::vector<std::string>> split(std::string const& text,
                                             std::string_view   delimiters) {
    std::vector<std::string> ret;

    for (std::size_t prev = 0, curr = 0,
                     next = text.find_first_of(delimiters, curr);
         prev != text.npos; // prev tracks next directly to ensure termination
         prev = next, curr = next + 1,
                     next = text.find_first_of(delimiters, curr)) {
      if (curr == next) continue;
      if (next - curr > 1 ||
          delimiters.find_first_of(text[curr]) == delimiters.npos) {
        ret.push_back(text.substr(curr, next - curr)); // substr uses pos, count
      }
    }

    return ret;
} // split

tl::optional<std::string> read(std::ifstream& ifs) {
    std::ostringstream oss;
    ifs >> oss.rdbuf();
    if (ifs.fail() && !ifs.eof()) return {};
    return oss.str();
}

tl::optional<std::ifstream> open(std::filesystem::path const& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    return ifs;
}

//tl::optional<std::string> slurp(std::filesystem::path const& path) {
    //return open(path).and_then(read);
//} // slurp

#include <cstdlib>
#include <iostream>

int main() {
#if defined(_MSC_VER_)
    auto words =
      open("../../../lorem_100words.txt")
        .and_then(read)
        .and_then([](auto t) { return split(t, " \t\r\n"); })
        .and_then(count<std::vector<std::string>>)
        .and_then(reverse_pairs<std::unordered_map<std::string, std::size_t>>)
        .and_then(
          sort_by_frequency<std::vector<std::pair<std::size_t, std::string>>>);

    std::cout << "unique words: " << words->size() << "\nfrequencies:\n";
    for (auto&& word : *words) {
      std::cout << "  " << word.second << ": " << word.first << "\n";
    }
#endif
    return EXIT_SUCCESS;
}
