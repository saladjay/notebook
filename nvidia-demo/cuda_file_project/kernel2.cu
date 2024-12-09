// kernel2.cu
#include <cuda_runtime.h>
#include <iostream>
#include "kernel1.cuh"

// __device__ float deviceFunction(float x) {
//     return x * x;
// }

extern float __device__ deviceFunction(float x);

__global__ void kernel(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = deviceFunction(data[idx]);
    }
}