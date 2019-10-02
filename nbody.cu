#include <cuda_runtime.h>

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
}
