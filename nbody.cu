#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include <string>
#include <system_error>

#if defined(__CUDACC__)
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif

#define DEVICE_HOST DEVICE HOST

/*!
\brief std::error_code category for cudaError
*/
class cuda_error_category_impl : public std::error_category {
public:
  ~cuda_error_category_impl() noexcept override = default;
  const char* name() const noexcept override { return "cudaError"; }
  std::string message(int ev) const override {
    return cudaGetErrorString(static_cast<cudaError_t>(ev));
  }
};

inline std::error_category const& cuda_error_category() {
  static cuda_error_category_impl sCudaErrorCategory;
  return sCudaErrorCategory;
}

inline std::error_code make_error_code(cudaError e) noexcept {
  return std::error_code(static_cast<int>(e), cuda_error_category());
}

namespace std {
template <>
struct is_error_code_enum<cudaError> : public true_type {};
} // namespace std

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

template <typename T, std::size_t Size>
constexpr DEVICE_HOST vec<T, Size> operator+(vec<T, Size> a, vec<T, Size> const& b) noexcept {
  for (std::size_t i = 0; i < Size; ++i) a.m[i] = a.m[i] + b.m[i];
  return a;
}

template <typename T, std::size_t Size>
constexpr DEVICE_HOST vec<T, Size>& operator+=(vec<T, Size>& a, vec<T, Size> const& b) noexcept {
  for (std::size_t i = 0; i < Size; ++i) a.m[i] += b.m[i];
  return a;
}

using vec2d = vec<double, 2>;
using vec3d = vec<double, 3>;
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

template <typename T, std::size_t Size, std::size_t IndexA, std::size_t IndexB>
constexpr DEVICE_HOST component_accessor<T, Size, IndexA>
                      operator+(component_accessor<T, Size, IndexA>        a,
          component_accessor<T, Size, IndexB> const& b) noexcept {
  a.m[IndexA] = a.m[IndexA] + b.m[IndexB];
  return a;
}

template <typename T, std::size_t Size, std::size_t IndexA, std::size_t IndexB>
constexpr DEVICE_HOST component_accessor<T, Size, IndexA>&
                      operator+=(component_accessor<T, Size, IndexA>&       a,
           component_accessor<T, Size, IndexB> const& b) noexcept {
  a.m[IndexA] += b.m[IndexB];
  return a;
}

template <typename T>
struct vec<T, 2> {
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type&;
  using const_reference = value_type const&;
  enum { Size = 2 };

  union {
    value_type m[Size];

    // component_accessor has the same size and layout as the array m above,
    // thus m, x, and y can be used without invalidating the union.
    component_accessor<value_type, Size, 0> x;
    component_accessor<value_type, Size, 1> y;
  };

  DEVICE_HOST reference operator[](size_type index) { return m[index]; }
  DEVICE_HOST const_reference operator[](size_type index) const { return m[index]; }

  template <class... Ts, std::enable_if_t<sizeof...(Ts) == Size>* = nullptr>
  explicit constexpr DEVICE_HOST vec(Ts... ts) noexcept
    : m{std::forward<Ts>(ts)...} {}

  explicit constexpr DEVICE_HOST vec(value_type a) noexcept {
    m[0] = m[1] = a;
  }

  constexpr DEVICE_HOST vec& operator=(value_type a) noexcept {
    m[0] = m[1] = a;
    return *this;
  };

  constexpr /*DEVICE_HOST*/ vec() noexcept = default; // annotations ignored on explicitly default?
}; // struct vec<T, 2>

template <typename T>
constexpr DEVICE_HOST vec<T, 2> operator+(vec<T, 2> a, vec<T, 2> const& b) noexcept {
  a.m[0] = a.m[0] + b.m[0];
  a.m[1] = a.m[1] + b.m[1];
  return a;
}

template <typename T>
constexpr DEVICE_HOST vec<T, 2>& operator+=(vec<T, 2>& a, vec<T, 2> const& b) noexcept {
  a.m[0] += b.m[0];
  a.m[1] += b.m[1];
  return a;
}

template <typename T>
struct vec<T, 3> {
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type&;
  using const_reference = value_type const&;
  enum { Size = 3 };

  union {
    value_type m[Size];

    // component_accessor has the same size and layout as the array m above,
    // thus m, x, y, and z can be used without invalidating the union.
    component_accessor<value_type, Size, 0> x;
    component_accessor<value_type, Size, 1> y;
    component_accessor<value_type, Size, 2> z;
  };

  DEVICE_HOST reference operator[](size_type index) { return m[index]; }
  DEVICE_HOST const_reference operator[](size_type index) const { return m[index]; }

  template <class... Ts, std::enable_if_t<sizeof...(Ts) == Size>* = nullptr>
  explicit constexpr DEVICE_HOST vec(Ts... ts) noexcept
    : m{std::forward<Ts>(ts)...} {}

  explicit constexpr DEVICE_HOST vec(value_type a) noexcept {
    m[0] = m[1] = m[2] = a;
  }

  constexpr DEVICE_HOST vec& operator=(value_type a) noexcept {
    m[0] = m[1] = m[2] = a;
    return *this;
  };

  constexpr /*DEVICE_HOST*/ vec() noexcept = default; // annotations ignored on explicitly default?
}; // struct vec<T, 3>

template <typename T>
constexpr DEVICE_HOST vec<T, 3> operator+(vec<T, 3> a, vec<T, 3> const& b) noexcept {
  a.m[0] = a.m[0] + b.m[0];
  a.m[1] = a.m[1] + b.m[1];
  a.m[2] = a.m[2] + b.m[2];
  return a;
}

template <typename T>
constexpr DEVICE_HOST vec<T, 3>& operator+=(vec<T, 3>& a, vec<T, 3> const& b) noexcept {
  a.m[0] += b.m[0];
  a.m[1] += b.m[1];
  a.m[2] += b.m[2];
  return a;
}

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

template <typename T>
constexpr DEVICE_HOST vec<T, 4> operator+(vec<T, 4> a, vec<T, 4> const& b) noexcept {
  a.m[0] = a.m[0] + b.m[0];
  a.m[1] = a.m[1] + b.m[1];
  a.m[2] = a.m[2] + b.m[2];
  a.m[3] = a.m[3] + b.m[3];
  return a;
}

template <typename T>
constexpr DEVICE_HOST vec<T, 4>& operator+=(vec<T, 4>& a, vec<T, 4> const& b) noexcept {
  a.m[0] += b.m[0];
  a.m[1] += b.m[1];
  a.m[2] += b.m[2];
  a.m[3] += b.m[3];
  return a;
}

class buffer {
public:
  using size_type = std::size_t;

  void* get() noexcept { return ptr_; }
  operator void*() noexcept { return ptr_; }

  enum class attachment_points { global, host };

  buffer(size_type sizeBytes, attachment_points attachment = attachment_points::global) {
    cudaError err = cudaMallocManaged(&ptr_, sizeBytes,
                                      attachment == attachment_points::global ? cudaMemAttachGlobal
                                                                              : cudaMemAttachHost);
    if (err != cudaSuccess && err != cudaErrorNotSupported) std::terminate();
    if (err == cudaErrorNotSupported) {
      err = cudaMalloc(&ptr_, sizeBytes);
      if (err != cudaSuccess) std::terminate();
    }
  }

  ~buffer() noexcept { cudaFree(ptr_); }

  friend void swap(buffer& a, buffer& b) noexcept { std::swap(a.ptr_, b.ptr_); }

  buffer(buffer&& other) { swap(*this, other); }
  buffer& operator=(buffer&& other) {
    swap(*this, other);
    return *this;
  }

  buffer() noexcept = default;
  buffer(buffer const&) = delete;
  buffer& operator=(buffer const&) = delete;

private:
  void* ptr_{nullptr};
}; // class buffer

template <class T>
class typed_buffer : public buffer {
public:
  using value_type = T;
  using size_type = std::size_t;

  T* get() noexcept { return reinterpret_cast<T*>(buffer::get()); }
  operator T*() noexcept { return reinterpret_cast<T*>(buffer::get()); }

  typed_buffer(size_type count, attachment_points attachment = attachment_points::global)
    : buffer{count * sizeof(T), attachment} {}

  ~typed_buffer() noexcept = default;

  friend void swap(typed_buffer& a, typed_buffer& b) noexcept {
    using std::swap;
    swap(static_cast<buffer&>(a), static_cast<buffer&>(b));
  }

  typed_buffer(typed_buffer&& other) : buffer{std::move(other)} {}
  typed_buffer& operator=(typed_buffer&& other) {
    buffer::operator=(std::move(other));
    return *this;
  }

  typed_buffer() noexcept = default;
  typed_buffer(typed_buffer const&) = delete;
  typed_buffer& operator=(typed_buffer const&) = delete;
}; // class typed_buffer

__global__ void calculate_accelerations(std::size_t numBodies, vec3d* positions, vec3d* accel) {
  std::size_t const bid = blockIdx.y * gridDim.x + blockIdx.x;
  std::size_t const tid = threadIdx.x;
  std::size_t const idx = bid * blockDim.x + tid;

  extern __shared__ vec3d shmem[]; // 128 threads per block, make dynamic
  shmem[tid] = vec3d{0.0, 0.0, 0.0};

  if (idx >= numBodies) return;
} // calculate_accelerations

__global__ void update_positions(std::size_t numBodies, double deltaTime, vec3d* positions,
                                 vec3d* velocities, vec3d* accel, vec3d* accel0) {
  std::size_t const bid = blockIdx.y * gridDim.x + blockIdx.x;
  std::size_t const tid = threadIdx.x;
  std::size_t const idx = bid * blockDim.x + tid;

  if (idx >= numBodies) return;
} // update_positions

__global__ void update_velocities(std::size_t numBodies, double deltaTime, vec3d* velocities,
                                  vec3d* accel, vec3d* accel0) {
  std::size_t const bid = blockIdx.y * gridDim.x + blockIdx.x;
  std::size_t const tid = threadIdx.x;
  std::size_t const idx = bid * blockDim.x + tid;

  if (idx >= numBodies) return;
} // update_velocities

__global__ void compute_energies(std::size_t numBodies, vec3d* positions, vec3d* velocities,
                                 vec3d* accel0, vec2d* energies) {
  std::size_t const bid = blockIdx.y * gridDim.x + blockIdx.x;
  std::size_t const tid = threadIdx.x;
  std::size_t const idx = bid * blockDim.x + tid;

  extern __shared__ vec3d shmem[]; // 128 threads per block, make dynamic
  shmem[tid] = vec3d{0.0, 0.0, 0.0};

  if (idx >= numBodies) return;
} // compute_energies

#include <cstdlib>
#include <fstream>
#include <iostream>

int main() {
  std::ifstream ifs{"../../../nbody_data/input128"};
  if (!ifs) {
    std::cout << "cannot open ../../../nbody_data/input128\n";
    return EXIT_FAILURE;
  }

  int         numBodies = 0;
  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    if (!line.empty()) numBodies += 1;
  }
  ifs.seekg(0);

  std::cout << "numBodies: " << numBodies << "\n";

  typed_buffer<vec3d> positions(numBodies);
  typed_buffer<vec3d> velocities(numBodies);
  typed_buffer<vec3d> accelerations(numBodies);
  typed_buffer<vec3d> accelerations0(numBodies);
  typed_buffer<vec2d> energies(numBodies, buffer::attachment_points::host);

  int    particleIndex;
  double mass;
  for (std::size_t i = 0; i < numBodies; ++i) {
    ifs >> particleIndex >> mass;
    ifs >> positions[i][0] >> positions[i][1] >> positions[i][2];
    ifs >> velocities[i][0] >> velocities[i][1] >> velocities[i][2];
  }

  dim3 threads, grid;
  threads.x = (numBodies < 128) ? numBodies : 128;
  grid.x    = (numBodies / 128) + 1;

  calculate_accelerations<<<grid, threads, sizeof(vec3d) * threads.x>>>(numBodies, positions,
                                                                        accelerations);
  compute_energies<<<256, 128, sizeof(vec2d) * threads.x>>>(numBodies, positions, velocities,
                                                            accelerations, energies);

  for (int i = 1; i < 256; ++i) {
    energies[0].x += energies[i].x;
    energies[0].y += energies[i].y;
  }
  std::cout << "Energies:" << energies[0].x + energies[0].y << "\t" << energies[0].x << "\t"
            << energies[0].y << "\n";
  vec2d energies0 = energies[0];

  double const dt   = 1e-3;
  double       tend = 1.0;
  double       t    = 0.0;
  int          k    = 0;

  while (t < tend) {
    update_positions<<<grid, threads>>>(numBodies, dt, positions, velocities, accelerations,
                                        accelerations0);
    calculate_accelerations<<<grid, threads, sizeof(vec3d) * threads.x>>>(numBodies, positions,
                                                                          accelerations);
    update_velocities<<<grid, threads>>>(numBodies, dt, velocities, accelerations, accelerations0);

    t += dt;
    k += 1;

    if (k % 10 == 0) {
      compute_energies<<<256, 128, sizeof(vec2d) * threads.x>>>(numBodies, positions, velocities,
                                                                accelerations, energies);

      for (int i = 1; i < 256; ++i) {
        energies[0].x += energies[i].x;
        energies[0].y += energies[i].y;
      }

      std::cout << "t= " << t << " E= " << energies[0].x + energies[0].y << " " << energies[0].x
                << " " << energies[0].y << " dE = "
                << (((energies[0].x + energies[0].y) - (energies0.x + energies0.y)) /
                    (energies0.x + energies0.y))
                << "\n";
      energies0 = energies[0];
    }
  }

  return 0;
}
