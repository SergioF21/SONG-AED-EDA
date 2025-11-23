#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* msg);
int atomicAddWrapper(int* address, int val);
int divUp(int a, int b);
int getNextPowerOfTwo(int n);
std::vector<int> generateRandomIndices(int total, int sample_size, int seed =12345);


#endif // UTILS_H
