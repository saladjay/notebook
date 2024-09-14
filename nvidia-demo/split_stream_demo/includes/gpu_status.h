#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
int get_gpu_status(int gpu_id) {    

    nvmlReturn_t result;
    unsigned int device_count, i;

    // Initialize NVML
    result = nvmlInit();
    if (result!= NVML_SUCCESS) {
        printf("Failed to initialize NVML: %d\n", result);
        return 1;
    }

    // Get device count
    result = nvmlDeviceGetCount(&device_count);
    if (result!= NVML_SUCCESS) {
        printf("Failed to get device count: %d\n", result);
        return 1;
    }

    // Check if GPU id is valid
    if (gpu_id >= device_count) {
        printf("Invalid GPU id: %d\n", gpu_id);
        return 1;
    }

    for (i = 0; i < device_count; i++)
    {
        nvmlDevice_t  device;
        char          name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlPciInfo_t pci;

        // 通过设备句柄查询设备以执行操作
        // 除了使用 PCI 总线 ID 之外，还可以通过其他特性来查询设备句柄，例如：
        // nvmlDeviceGetHandleBySerial
        // nvmlDeviceGetHandleByPciBusId
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get handle for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get name of device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        // pci.busId 对于识别您在物理上正在与哪个设备进行通信非常有用
        // 使用 PCI 标识符，您还可以将 nvmlDevice 句柄匹配到 CUDA 设备。
        result = nvmlDeviceGetPciInfo(device, &pci);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get pci info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("%u. %s [%s]\n", i, name, pci.busId);

        // 查询显存信息
        nvmlMemory_t memory;

        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get memory info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Memory(NVML): %f MiB / %f MiB\n", memory.used / 1024.0 / 1024.0, memory.total / 1024.0 / 1024.0);
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        size_t used_memory = total_memory - free_memory;
        printf("    Memory(CUDA): %f MiB / %f MiB\n", used_memory / 1024.0 / 1024.0, total_memory / 1024.0 / 1024.0);

        // 查询显卡驱动版本
        char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        result = nvmlSystemGetDriverVersion(driver_version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get driver info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Driver Version: %s\n", driver_version);

        // 查询显卡频率
        unsigned int graph_clock, mem_clock;
        // result = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &clock); // 不支持
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graph_clock);
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Graphics Clock: %u MHz\n", graph_clock);
        printf("    Memory Clock: %u MHz\n", mem_clock);

        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &graph_clock);
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &mem_clock);
        if(NVML_SUCCESS != result)
        {
            printf("Failed to get max clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Graphics Max Clock: %u MHz\n", graph_clock);
        printf("    Memory Max Clock: %u MHz\n", mem_clock);

        result = nvmlDeviceSetGpuLockedClocks(device, graph_clock, graph_clock);
        result = nvmlDeviceSetMemoryLockedClocks(device, mem_clock, mem_clock);
        printf("    Locked clocks set ***********************\n");
        if (NVML_SUCCESS != result)
        {
            printf("Failed to set locked clocks for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graph_clock);
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Graphics Clock: %u MHz\n", graph_clock);
        printf("    Memory Clock: %u MHz\n", mem_clock);


        result = nvmlDeviceResetGpuLockedClocks(device);
        result = nvmlDeviceResetMemoryLockedClocks(device);
        printf("    Locked clocks reset ***********************\n");
        if (NVML_SUCCESS != result)
        {   
            printf("Failed to reset locked clocks for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }   

        
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graph_clock);
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Graphics Clock: %u MHz\n", graph_clock);
        printf("    Memory Clock: %u MHz\n", mem_clock);

        // 查询显卡温度
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get temperature for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Temperature: %u C\n", temp);

        // 查询利用率
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get utilization for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Utilization: GPU: %u%%, Memory: %u%%\n", utilization.gpu, utilization.memory);



    }
    


    // Get handle for the GPU device
    nvmlDevice_t device;    

    nvmlDeviceGetHandleByIndex(gpu_id, &device);

    // Get GPU status
    nvmlUtilization_t utilization;
    nvmlMemory_t memory;    
    nvmlDeviceGetUtilizationRates(device, &utilization);
    nvmlDeviceGetMemoryInfo(device, &memory);

    // Print GPU status    
    printf("GPU %d: %d%% utilization, %d MiB used / %d MiB total\n", gpu_id, utilization.gpu, memory.used / 1024 / 1024, memory.total / 1024 / 1024);

    // Clean up
    nvmlShutdown();

    return 0;
}