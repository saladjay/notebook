#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thread>
#include <atomic>
#define DOWNCOUNT 10
// #include "func.h"
__global__ void cudaSetValue0_device(int32_t* values, size_t jobs) {
	for (size_t count{ 0 }; count < DOWNCOUNT; count++) {
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			values[idx] = 0;
		}
	}
}

__global__ void cudaSetValue1_device(int32_t* values, size_t jobs) {
    for(size_t count{0};count<DOWNCOUNT;count++){
        for (size_t idx{ 0 }; idx < jobs; ++idx) {
            values[idx] = 1;
        }
    }
}

__global__ void cudaSetValue2_device(int32_t* values, size_t jobs) {
	bool check_equal_one = false;
    for(size_t count{0};count<1;count++){
        for (size_t idx{ 0 }; idx < jobs; ++idx) {
			if (values[idx] == 1) {
				values[idx] = 2;
			}
			else {
				check_equal_one = true;
			}
        }
    }
	if (check_equal_one)
	{
		printf("no equal one\n");
	}
}

__global__ void cudaSetValue2_device_copy(int32_t* values, int32_t* values2, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			if (values[idx] == 1)
				values2[idx] = 2;
		}
	}
}

__global__ void cudaSetValue3_device(int32_t* values, size_t jobs) {
    for(size_t count{0};count<1;count++){
        for (size_t idx{ 0 }; idx < jobs; ++idx) {
			if(values[idx]==2)
				values[idx] = 3;	
        }
    }
}

__global__ void cudaSetValue3_device_copy(int32_t* values, int32_t* values2, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			if (values[idx] == 2)
				values2[idx] = 3;
		}
	}
}

__global__ void cudaCheckValue3_device(int32_t* values, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		bool check{ true };
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			check &= values[idx] == 3;
		}
		printf("all values equal 3 : %i\n", check);
	}
}

__global__ void cudaCheckValue1_device(int32_t* values, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		bool check{ true };
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			check &= values[idx] == 1;
		}
		printf("all values equal 1 : %i\n", check);
	}
}

__global__ void cudaCheckValue1_device_s1(int32_t* values, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		bool check{ true };
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			check &= values[idx] == 1;
		}
		printf("s1 all values equal 1 : %i\n", check);
	}
}


__global__ void cudaCheckValue1_device_s2(int32_t* values, size_t jobs) {
	for (size_t count{ 0 }; count < 1; count++) {
		bool check{ true };
		for (size_t idx{ 0 }; idx < jobs; ++idx) {
			check &= values[idx] == 1;
		}
		printf("s2 all values equal 1 : %i\n", check);
	}
}

extern "C" void cudaSetValue1(int32_t* values, size_t jobs){
    cudaSetValue1_device<<<1,1>>>(values, jobs);
}

extern "C" void cudaSetValue2(int32_t* values, size_t jobs){
    cudaSetValue2_device<<<1,1>>>(values, jobs);
};

extern "C" void cudaSetValue3(int32_t* values, size_t jobs){
    cudaSetValue3_device<<<1,1>>>(values, jobs);
};

cudaEvent_t event1, event2, event3, event4;
cudaStream_t stream1, stream2;

int main(){
    std::cout<<"hello world"<<std::endl;
	int count{ 0 };
	cudaGetDeviceCount(&count);
	std::cout<<count<<std::endl;

	int32_t *device_values, *device_values2;
	size_t jobs{ 1000000 };

	cudaMalloc((void**)&device_values, sizeof(int32_t)*jobs);
	cudaMalloc((void**)&device_values2, sizeof(int32_t)*jobs);

	int32_t *host_values = new int32_t[jobs];

	cudaSetValue1_device<<<1,1>>>(device_values, jobs);
    cudaSetValue2_device<<<1,1>>>(device_values, jobs);
    cudaSetValue3_device<<<1,1>>>(device_values, jobs);

	cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);

	bool check{ true };
	for (size_t idx{ 0 }; idx < jobs; ++idx) {
		check &= host_values[idx] == 1;
	}

	{
		auto event_flag = cudaEventQuery(event1);
		std::cout << __LINE__ << " event1 " << event_flag << cudaGetErrorString(event_flag) << std::endl;
	}

    if(auto cuda_ret = cudaStreamCreate(&stream1); cuda_ret != cudaError::cudaSuccess){
        std::cout<<"stream1 create failed"<<std::endl;
    }
    if(auto cuda_ret = cudaStreamCreate(&stream2); cuda_ret != cudaError::cudaSuccess){
        std::cout<<"stream2 create failed"<<std::endl;
    }
    if(auto cuda_ret = cudaEventCreate(&event1); cuda_ret != cudaError::cudaSuccess){
        std::cout<<"event1 create failed"<<std::endl;
    }
    if(auto cuda_ret = cudaEventCreate(&event2); cuda_ret != cudaError::cudaSuccess){
        std::cout<<"event2 create failed"<<std::endl;
    }
    
	
	//  *::cudaSuccess,
	//	*::cudaErrorNotReady,
	//	*::cudaErrorInvalidValue,
	//	*::cudaErrorInvalidResourceHandle,
	//	*::cudaErrorLaunchFailure

	{
		auto event_flag = cudaEventQuery(event1);
		std::cout<<__LINE__<<" event1 " << event_flag << cudaGetErrorString(event_flag)<< std::endl;
	}

    cudaEventRecord(event1, stream1);
	{
		auto event_flag = cudaEventQuery(event1);
		std::cout << __LINE__ << " event1 " << event_flag << cudaGetErrorString(event_flag) << std::endl;
	}
	{
		auto event_flag = cudaEventQuery(event2);
		std::cout << __LINE__ << " event2 " << event_flag << cudaGetErrorString(event_flag) << std::endl;
	}
    cudaSetValue1_device<<<1,1,0,stream1>>>(device_values, jobs);
    cudaSetValue2_device<<<1,1,0,stream1>>>(device_values, jobs);
    cudaSetValue3_device<<<1,1,0,stream1>>>(device_values, jobs);
    cudaEventRecord(event2, stream1);
    cudaEventSynchronize(event2);

	{
		auto event_flag = cudaEventQuery(event1);
		std::cout << __LINE__ << " event1 " << event_flag << cudaGetErrorString(event_flag) << std::endl;
	}

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, event1, event2);

    std::cout << elapsedTime << std::endl;

	cudaDeviceSynchronize();
	cudaEventDestroy(event1);
	cudaEventDestroy(event2);

	if (auto cuda_ret = cudaEventCreate(&event1); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event1 create failed" << std::endl;
	}
	if (auto cuda_ret = cudaEventCreate(&event2); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event2 create failed" << std::endl;
	}
	if (auto cuda_ret = cudaEventCreate(&event3); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event1 create failed" << std::endl;
	}
	if (auto cuda_ret = cudaEventCreate(&event4); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event2 create failed" << std::endl;
	}

	//std::thread
	cudaSetValue1_device << <1, 1, 0, stream1 >> > (device_values, jobs);
	cudaEventRecord(event1, stream1);
	cudaEventSynchronize(event1);

	cudaStreamWaitEvent(stream2, event1);
	cudaSetValue2_device << <1, 1, 0, stream2 >> > (device_values, jobs);
	cudaEventRecord(event2, stream2);
	cudaSetValue3_device << <1, 1, 0, stream2 >> > (device_values, jobs);
	cudaEventSynchronize(event2);

	cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);

	check = true;
	for (size_t idx{ 0 }; idx < jobs; ++idx) {
		check &= host_values[idx] == 3;
	}
	std::cout << check << std::endl;
	std::cout << "finish" << std::endl;


	cudaSetValue0_device << <1, 1 >> > (device_values, jobs);
	cudaCheckValue1_device << <1, 1 >> > (device_values, jobs);
	cudaCheckValue3_device << <1, 1 >> > (device_values, jobs);
	cudaDeviceSynchronize();
	cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);
	check = true;
	for (size_t idx{ 0 }; idx < jobs; ++idx) {
		check &= host_values[idx] == 3;
	}
	std::cout << __LINE__ << " " << check << std::endl;

	if (auto cuda_ret = cudaEventCreate(&event1); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event1 create failed" << std::endl;
	}
	if (auto cuda_ret = cudaEventCreate(&event2); cuda_ret != cudaError::cudaSuccess) {
		std::cout << "event2 create failed" << std::endl;
	}

	std::atomic<int> inference_count{ 0 };
	std::thread t1{ [&]() {
		for (size_t i = 0; i < 10; ++i) {
			if (inference_count != 0) {
				cudaStreamWaitEvent(stream1, event2, 0);
			}
			cudaSetValue1_device << <1, 1, 0, stream1 >> > (device_values, jobs);
			cudaCheckValue1_device_s1 << <1, 1, 0, stream1 >> > (device_values, jobs);
			cudaEventRecord(event1, stream1);
			cudaEventSynchronize(event1);
			//std::cout << __LINE__ << " event1 consumed" << std::endl;
			if (auto cuda_ret = cudaEventCreate(&event1); cuda_ret != cudaError::cudaSuccess) {
				std::cout << "event1 create failed" << std::endl;
			}
		}
	} };

	std::thread t2{ [&]() {
		for (size_t i = 0; i < 10; ++i) {
			cudaStreamWaitEvent(stream2, event1, 0);
			cudaCheckValue1_device_s2 << <1, 1, 0, stream2 >> > (device_values, jobs);
			//std::cout << __LINE__ << " event1 consumed" << std::endl;
			cudaSetValue2_device_copy << <1, 1, 0, stream2 >> > (device_values, device_values2, jobs);
			
			cudaSetValue3_device << <1, 1, 0, stream2 >> > (device_values2, jobs);
			cudaEventRecord(event2, stream2);
			cudaCheckValue3_device << <1, 1, 0, stream2 >> > (device_values2, jobs);
			cudaStreamSynchronize(stream2);
			++inference_count;

			//cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);
			//check = true;
			//for (size_t idx{ 0 }; idx < jobs; ++idx) {
			//	check &= host_values[idx] == 3;
			//}
			//std::cout << __LINE__ << " " <<  check << std::endl;

			//if (auto cuda_ret = cudaEventCreate(&event2); cuda_ret != cudaError::cudaSuccess) {
			//	std::cout << "event2 create failed" << std::endl;
			//}
		}
	} };
	t1.detach();
	t2.join();
	cudaMemcpy(host_values, device_values, sizeof(int32_t)*jobs, cudaMemcpyDeviceToHost);

	check = true;
	for (size_t idx{ 0 }; idx < jobs; ++idx) {
		check &= host_values[idx] == 3;
	}
	std::cout << __LINE__ << std::endl;
	std::cout << check << std::endl;
	std::cout << "finish" << std::endl;
    return 0;
}