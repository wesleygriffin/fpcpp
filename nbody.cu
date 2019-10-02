#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

#if defined(__CUDACC__)
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif

#define DEVICE_HOST DEVICE HOST

/*!
\brief General mathematical vector class.
*/
template <typename T, std::size_t Size>
struct vec {
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type&;
  using const_reference = value_type const&;

  value_type m[Size];

  DEVICE_HOST reference operator[](size_type index) { return m[index]; }
  DEVICE_HOST const_reference operator[](size_type index) const { return m[index]; }

  template <class... Ts, std::enable_if_t<sizeof...(Ts) == Size>* = nullptr>
  explicit constexpr DEVICE_HOST vec(Ts... ts) noexcept
    : m{std::forward<Ts>(ts)...} {}

  explicit constexpr DEVICE_HOST vec(value_type a) noexcept {
    for (size_type i = 0; i < Size; ++i) m[i] = a;
  }

  constexpr DEVICE_HOST vec& operator=(value_type a) noexcept {
    for (size_type i = 0; i < Size; ++i) m[i] = a;
    return *this;
  };

  constexpr /*DEVICE_HOST*/ vec() noexcept = default; // annotations ignored on explicitly default?
}; // struct vec<T, Size>

using vec2d = vec<double, 2>;
using vec4d = vec<double, 4>;

/*!
\brief Enable \ref vec specializations to use a union for custom accessors.
*/
template <typename T, std::size_t Size, std::size_t Index>
struct component_accessor {
  using value_type = T;

  value_type m[Size];

  template <std::size_t M>
  constexpr DEVICE_HOST operator vec<T, M>() const noexcept { return {m[Index]}; }
  constexpr DEVICE_HOST value_type get() const noexcept { return m[Index]; }
  constexpr DEVICE_HOST operator T() const noexcept { return m[Index]; }
}; // struct component_accessor<T, Size, I>

template <typename T>
struct vec<T, 4> {
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type&;
  using const_reference = value_type const&;
  enum { Size = 4 };

  union {
    value_type m[Size];

    // component_accessor has the same size and layout as the array m above,
    // thus m, x, y, z, and w can be used without invalidating the union.
    component_accessor<value_type, Size, 0> x;
    component_accessor<value_type, Size, 1> y;
    component_accessor<value_type, Size, 2> z;
    component_accessor<value_type, Size, 3> w;
  };

  DEVICE_HOST reference operator[](size_type index) { return m[index]; }
  DEVICE_HOST const_reference operator[](size_type index) const { return m[index]; }

  template <class... Ts, std::enable_if_t<sizeof...(Ts) == Size>* = nullptr>
  explicit constexpr DEVICE_HOST vec(Ts... ts) noexcept
    : m{std::forward<Ts>(ts)...} {}

  explicit constexpr DEVICE_HOST vec(value_type a) noexcept {
    m[0] = m[1] = m[2] = m[3] = a;
  }

  constexpr DEVICE_HOST vec& operator=(value_type a) noexcept {
    m[0] = m[1] = m[2] = m[3] = a;
    return *this;
  };

  constexpr /*DEVICE_HOST*/ vec() noexcept = default; // annotations ignored on explicitly default?
}; // struct vec<T, 4>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main() {
  std::ifstream ifs{"../../../nbody_data/input128"};
  if (!ifs) {
    std::cout << "cannot open ../../../nbody_data/input128\n";
    return EXIT_FAILURE;
  }

  int numParticles = 0;
  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    numParticles += 1;
  }
  ifs.seekg(0);

  std::cout << "numParticles: " << numParticles << "\n";

  cudaError err;

  vec4d* positions;
  err = cudaMallocManaged(&positions, numParticles * sizeof(vec4d));
  if (err != cudaSuccess) {
    std::cout << "cannot allocate memory\n";
    return EXIT_FAILURE;
  }

  vec4d* velocities;
  err = cudaMallocManaged(&velocities, numParticles * sizeof(vec4d));
  if (err != cudaSuccess) {
    std::cout << "cannot allocate memory\n";
    return EXIT_FAILURE;
  }

  vec4d* accelerations;
  err = cudaMallocManaged(&accelerations, numParticles * sizeof(vec4d));
  if (err != cudaSuccess) {
    std::cout << "cannot allocate memory\n";
    return EXIT_FAILURE;
  }

  vec4d* accelerations0;
  err = cudaMallocManaged(&accelerations0, numParticles * sizeof(vec4d));
  if (err != cudaSuccess) {
    std::cout << "cannot allocate memory\n";
    return EXIT_FAILURE;
  }

  vec2d* energies;
  err = cudaMallocManaged(&energies, numParticles * sizeof(vec2d));
  if (err != cudaSuccess) {
    std::cout << "cannot allocate memory\n";
    return EXIT_FAILURE;
  }

  int particleIndex;
  double mass;
  while (!ifs.eof()) {
    ifs >> particleIndex >> mass;
  }
}
