// main.cu
#include <iostream>
#include <cuda_runtime.h>
// #include "kernel1.cuh"

extern void __global__ kernel(float* data, int size);

int main() {
    const int arraySize = 5;
    float data[arraySize] = {1.0f, 2.0f, 5.0f, 4.0f, 5.0f};

    float* dev_data;
    cudaMalloc((void**)&dev_data, arraySize * sizeof(float));
    cudaMemcpy(dev_data, data, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, arraySize>>>(dev_data, arraySize);
    cudaMemcpy(data, dev_data, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_data);

    for (int i = 0; i < arraySize; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}