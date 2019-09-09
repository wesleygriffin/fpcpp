#include "helpers.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <istream>
#include <optional>
#include <vector>

#if defined(_MSC_VER)
#include <execution>
#endif

auto open_file_stream(std::filesystem::path const& path) -> std::ifstream {
  return std::ifstream(path);
}

auto count_lines_in_stream(std::istream& is) {
  return std::count(std::istreambuf_iterator<char>(is),
                    std::istreambuf_iterator<char>(), '\n');
}

template <class C>
auto open_files(C&& paths) {
  std::vector<std::ifstream> streams(paths.size());
#if defined(_MSC_VER)
  std::transform(std::execution::par_unseq, std::begin(paths), std::end(paths),
                 std::begin(streams), open_file_stream);
#else
  std::transform(std::begin(paths), std::end(paths), std::begin(streams), open_file_stream);
#endif
  return streams;
}

template <class C>
auto count_lines(C&& streams) {
  std::vector<decltype(count_lines_in_stream(streams[0]))> counts(
    std::size(streams));
#if defined(_MSC_VER)
  std::transform(std::execution::par_unseq, std::begin(streams),
                 std::end(streams), std::begin(counts), count_lines_in_stream);
#else
  std::transform(std::begin(streams), std::end(streams), std::begin(counts), count_lines_in_stream);
#endif
  return counts;
}

auto count_lines_in_files(std::vector<std::filesystem::path> const& paths) {
  return count_lines(open_files(paths));
}

auto count_lines_in_string(std::string_view s) {
  return std::accumulate(
    std::begin(s), std::end(s), 0,
    [](int n, char c) -> int { return c == '\n' ? n + 1 : n; });
}

#include <iostream>

int main() {
  std::vector<std::filesystem::path> const filenames{
    "../../../sum.cpp", "../../../wc.cpp", "../../../CMakeLists.txt",
    "../../../helpers.hpp", "../../../.clang-format"};

  auto const counts = count_lines_in_files(filenames);
  for (auto&& [f, c] : iter::zip(filenames, counts)) {
    std::cout << f << ": " << c << "\n";
  }

  auto const s = R"(testing\n
hi there\n
what what\n
foo bar\n
)";

  std::cout << count_lines_in_string(s) << "\n";

  return EXIT_SUCCESS;
}
