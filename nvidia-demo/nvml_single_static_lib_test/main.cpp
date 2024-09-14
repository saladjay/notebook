
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
        // size_t free_memory, total_memory;
        // cudaMemGetInfo(&free_memory, &total_memory);
        // size_t used_memory = total_memory - free_memory;
        // printf("    Memory(CUDA): %f MiB / %f MiB\n", used_memory / 1024.0 / 1024.0, total_memory / 1024.0 / 1024.0);

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


int main() {
    get_gpu_status(0);
    return 0;
}

 /***************************************************************************\
|*                                                                           *|
|*      Copyright 2010-2016 NVIDIA Corporation.  All rights reserved.        *|
|*                                                                           *|
|*   NOTICE TO USER:                                                         *|
|*                                                                           *|
|*   This source code is subject to NVIDIA ownership rights under U.S.       *|
|*   and international Copyright laws.  Users and possessors of this         *|
|*   source code are hereby granted a nonexclusive, royalty-free             *|
|*   license to use this code in individual and commercial software.         *|
|*                                                                           *|
|*   NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE     *|
|*   CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR         *|
|*   IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH      *|
|*   REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF         *|
|*   MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR          *|
|*   PURPOSE. IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL,            *|
|*   INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES          *|
|*   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN      *|
|*   AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING     *|
|*   OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE      *|
|*   CODE.                                                                   *|
|*                                                                           *|
|*   U.S. Government End Users. This source code is a "commercial item"      *|
|*   as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting       *|
|*   of "commercial computer  software" and "commercial computer software    *|
|*   documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)   *|
|*   and is provided to the U.S. Government only as a commercial end item.   *|
|*   Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through        *|
|*   227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the       *|
|*   source code with only those rights set forth herein.                    *|
|*                                                                           *|
|*   Any use of this source code in individual and commercial software must  *| 
|*   include, in the user documentation and internal comments to the code,   *|
|*   the above Disclaimer and U.S. Government End Users Notice.              *|
|*                                                                           *|
|*                                                                           *|
 \***************************************************************************/

// #include <stdio.h>
// #include <nvml.h>

// static const char * convertToComputeModeString(nvmlComputeMode_t mode)
// {
//     switch (mode)
//     {
//         case NVML_COMPUTEMODE_DEFAULT:
//             return "Default";
//         case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
//             return "Exclusive_Thread";
//         case NVML_COMPUTEMODE_PROHIBITED:
//             return "Prohibited";
//         case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
//             return "Exclusive Process";
//         default:
//             return "Unknown";
//     }
// }

// int main(void)
// {
//     nvmlReturn_t result;
//     unsigned int device_count, i;

//     // First initialize NVML library
//     result = nvmlInit();
//     if (NVML_SUCCESS != result)
//     { 
//         printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

//         printf("Press ENTER to continue...\n");
//         getchar();
//         return 1;
//     }

//     result = nvmlDeviceGetCount(&device_count);
//     if (NVML_SUCCESS != result)
//     { 
//         printf("Failed to query device count: %s\n", nvmlErrorString(result));
//         goto Error;
//     }
//     printf("Found %u device%s\n\n", device_count, device_count != 1 ? "s" : "");

//     printf("Listing devices:\n");    
//     for (i = 0; i < device_count; i++)
//     {
//         nvmlDevice_t device;
//         char name[NVML_DEVICE_NAME_BUFFER_SIZE];
//         nvmlPciInfo_t pci;
//         nvmlComputeMode_t compute_mode;

//         // Query for device handle to perform operations on a device
//         // You can also query device handle by other features like:
//         // nvmlDeviceGetHandleBySerial
//         // nvmlDeviceGetHandleByPciBusId
//         result = nvmlDeviceGetHandleByIndex(i, &device);
//         if (NVML_SUCCESS != result)
//         { 
//             printf("Failed to get handle for device %u: %s\n", i, nvmlErrorString(result));
//             goto Error;
//         }

//         result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
//         if (NVML_SUCCESS != result)
//         { 
//             printf("Failed to get name of device %u: %s\n", i, nvmlErrorString(result));
//             goto Error;
//         }
        
//         // pci.busId is very useful to know which device physically you're talking to
//         // Using PCI identifier you can also match nvmlDevice handle to CUDA device.
//         result = nvmlDeviceGetPciInfo(device, &pci);
//         if (NVML_SUCCESS != result)
//         { 
//             printf("Failed to get pci info for device %u: %s\n", i, nvmlErrorString(result));
//             goto Error;
//         }

//         printf("aaaa%u. %s [%s]\n", i, name, pci.busId);

//         // This is a simple example on how you can modify GPU's state
//         result = nvmlDeviceGetComputeMode(device, &compute_mode);
//         if (NVML_ERROR_NOT_SUPPORTED == result)
//             printf("\t This is not CUDA capable device\n");
//         else if (NVML_SUCCESS != result)
//         { 
//             printf("Failed to get compute mode for device %u: %s\n", i, nvmlErrorString(result));
//             goto Error;
//         }
//         else
//         {
//             // try to change compute mode
//             printf("\t Changing device's compute mode from '%s' to '%s'\n", 
//                     convertToComputeModeString(compute_mode), 
//                     convertToComputeModeString(NVML_COMPUTEMODE_PROHIBITED));

//             result = nvmlDeviceSetComputeMode(device, NVML_COMPUTEMODE_PROHIBITED);
//             if (NVML_ERROR_NO_PERMISSION == result)
//                 printf("\t\t Need root privileges to do that: %s\n", nvmlErrorString(result));
//             else if (NVML_ERROR_NOT_SUPPORTED == result)
//                 printf("\t\t Compute mode prohibited not supported. You might be running on\n"
//                        "\t\t windows in WDDM driver model or on non-CUDA capable GPU\n");
//             else if (NVML_SUCCESS != result)
//             {
//                 printf("\t\t Failed to set compute mode for device %u: %s\n", i, nvmlErrorString(result));
//                 goto Error;
//             } 
//             else
//             {
//                 printf("\t Restoring device's compute mode back to '%s'\n", 
//                         convertToComputeModeString(compute_mode));
//                 result = nvmlDeviceSetComputeMode(device, compute_mode);
//                 if (NVML_SUCCESS != result)
//                 { 
//                     printf("\t\t Failed to restore compute mode for device %u: %s\n", i, nvmlErrorString(result));
//                     goto Error;
//                 }
//             }
//         }
//     }

//     result = nvmlShutdown();
//     if (NVML_SUCCESS != result)
//         printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

//     printf("All done.\n");

//     printf("Press ENTER to continue...\n");
//     getchar();
//     return 0;

// Error:
//     result = nvmlShutdown();
//     if (NVML_SUCCESS != result)
//         printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

//     printf("Press ENTER to continue...\n");
//     getchar();
//     return 1;
// }
