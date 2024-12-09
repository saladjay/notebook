// kernel1.cu
#include "kernel1.cuh"

__device__ float deviceFunction(float x) {
    return x * x;
}