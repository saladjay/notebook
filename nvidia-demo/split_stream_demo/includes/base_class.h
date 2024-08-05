#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct ImageData 
{
    /* image info */
    size_t width{0}; // image width
    size_t height{0}; // image height
    size_t channel{0}; // image channel

    /* preprocess */
    size_t data_addr{0}; // pointer to matrix data
    size_t data_size{0}; // size of matrix data
    size_t data_idx{0}; // current index of data
    size_t offset_on_host{0}; // offset on host pinned memory
    size_t offset_on_device{0}; // offset on device memory

    /* postprocess */
    float* post_data{nullptr};
    size_t post_size{0};
    size_t post_data_offset{0};
};


