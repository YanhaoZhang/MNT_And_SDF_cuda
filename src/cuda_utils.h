// yanhao 2021.02.26
// cuda utils

#include <iostream>

// check cuda error
void FatalError(const int lineNumber = 0) {
  std::cerr << "FatalError";
  if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
  std::cerr << ". Program Terminated." << std::endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
    FatalError();
  }
}