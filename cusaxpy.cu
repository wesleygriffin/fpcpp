#include <cuda_runtime.h>

void saxpy_c(int n, float a, float* x, float* y) {
  for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

__global__ void saxpy(int n, float a, float* x, float* y) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

#include <iostream>

int main() {
  cudaError err;
  int const N = 1<<20;

  float* x;
  err = cudaMallocManaged(&x, N * sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "Cannot allocate x: " << err << "\n";
    return 1;
  }

  float* y;
  err = cudaMallocManaged(&y, N * sizeof(float));
  if (err != cudaSuccess) {
    std::cout << "Cannot allocate y: " << err << "\n";
    return 1;
  }

  for (int i = 0; i < N; ++i) x[i] = static_cast<float>(i);
  for (int i = 0; i < N; ++i) y[i] = static_cast<float>(i);

  std::cout << "x: [";
  for (int i = 0; i < 10; ++i) std::cout << " " << x[i];
  std::cout << " ]\n";

  std::cout << "y: [";
  for (int i = 0; i < 10; ++i) std::cout << " " << y[i];
  std::cout << " ]\n";

  saxpy<<<4096, 256>>>(N, 2.f, x, y);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << err << "\n";
    return 1;
  }

  std::cout << "x: [";
  for (int i = 0; i < 10; ++i) std::cout << " " << x[i];
  std::cout << " ]\n";

  std::cout << "y: [";
  for (int i = 0; i < 10; ++i) std::cout << " " << y[i];
  std::cout << " ]\n";

  return 0;
}
