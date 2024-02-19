#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
int main(){

    std::cout<<"hello world"<<std::endl;
	int count{ 0 };
	cudaGetDeviceCount(&count);
	std::cout<<count<<std::endl;

	int32_t *device_values;
	size_t jobs{ 1000000 };
	std::cout << __LINE__ << std::endl;
	cudaMalloc((void**)&device_values, sizeof(int32_t)*jobs);
	std::cout << __LINE__ << std::endl;
	int32_t *host_values = new int32_t[jobs];
	std::cout << __LINE__ << std::endl;
	cudaSetValue1(device_values, jobs);
	std::cout << __LINE__ << std::endl;
	cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);
	std::cout << __LINE__ << std::endl;
	bool check{ true };
	for (size_t idx{ 0 }; idx < jobs; ++idx) {
		check &= host_values[idx] == 1;
	}
	std::cout << check << std::endl;
	std::cout << "finish" << std::endl;
    return 0;
}