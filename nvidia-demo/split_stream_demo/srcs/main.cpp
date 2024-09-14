#include "../includes/base_class.h" 
#include "../includes/gpu_status.h"
__global__ void kernel_function(int* data) {
    printf("Hello from kernel function \n");
    int value = data[0];
    for(int i = 0; i < 10; i++) {
        printf("Hello from kernel function\n");
    }
}



int main() {

    ImageData img_data;

    int* times{nullptr};
    std::cout << __LINE__ << std::endl;
    cudaMalloc(&times, sizeof(int));
    std::cout << __LINE__ << std::endl;
    int* times_host = new int[1];
    std::cout << __LINE__ << std::endl;
    times_host[0] = 10;
    std::cout << __LINE__ << std::endl;
    cudaMemcpy(times, times_host, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << __LINE__ << std::endl;
    kernel_function(times);
    std::cout << __LINE__ << std::endl;

    times_host[0] = 20;
    cudaMemcpy(times_host, times, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "times_host[0] = " << times_host[0] << std::endl;

    int result = get_gpu_status(0);
    std::cout << "GPU status: " << result << std::endl;
    return 0;
}