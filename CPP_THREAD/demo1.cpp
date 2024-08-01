#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <random>
#include <algorithm>
#include <vector>

class Semaphore_ready {
public:
    Semaphore_ready() = default;
    ~Semaphore_ready() = default;
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock);
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;

};

struct Data 
{
    /* data */
    size_t data_addr{0};
    size_t data_size{0};
    size_t data_idx{0};
    size_t offset_on_host{0};
    size_t offset_on_device{0};

    /* postprocess */
    float* post_data{nullptr};
    size_t post_size{0};
    size_t post_data_offset{0};
};




int main(){

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, 10000000);
    
    size_t nb_tasks{10};
    size_t nb_single_data_len{10000}; // data size for each task

    std::vector<std::vector<size_t>> data_vec(nb_tasks);
    std::for_each(data_vec.begin(), data_vec.end(), [&nb_single_data_len, &generator, &distribution](auto& vec){
        vec.resize(nb_single_data_len);
        std::generate(vec.begin(), vec.end(), [&generator, &distribution](){
            return distribution(generator);
        });
    });

    // std::for_each(data_vec.begin(), data_vec.end(), [](auto& vec){
    //     std::for_each(vec.begin(), vec.end(), [](auto& data){
    //         std::cout << data << " ";
    //     });
    //     std::cout << std::endl;
    // });

    std::vector<Data> data_info(nb_tasks);
    std::for_each(data_info.begin(), data_info.end(), [&data_vec, &nb_single_data_len](auto& info){
        static size_t idx{0};
        info.data_addr = (size_t)(&data_vec[idx][0]);
        info.data_size = nb_single_data_len * sizeof(size_t);
        info.data_idx = idx++;
        info.offset_on_host = info.data_idx * nb_single_data_len;
        info.post_data_offset = info.data_idx;
        std::cout <<"data_vec:"<< (void*)data_vec[info.data_idx].data() << " data_addr: " << (void*)info.data_addr << " data_idx: " << info.data_idx << std::endl;
    });

    size_t *big_data_buffer = new size_t[nb_tasks * nb_single_data_len];

    std::thread load_thread([&data_info, &big_data_buffer, &nb_tasks, &nb_single_data_len](){
        std::for_each(data_info.begin(), data_info.end(), [&big_data_buffer, &nb_tasks, &nb_single_data_len](auto& info){
            size_t* data_ptr = (size_t*)(info.data_addr);
            std::memcpy(big_data_buffer + info.data_idx * nb_single_data_len, data_ptr, info.data_size);
        });
    });

    load_thread.join();

    // for (size_t i = 0; i < nb_tasks; i++)
    // {
    //     for (size_t j = 0; j < nb_single_data_len; j++)
    //     {
    //         if(data_vec[i][j] != big_data_buffer[i * nb_single_data_len + j]){
    //             std::cout << "Error: data not match" << i <<" "<< j << std::endl;
    //             return 0; 
    //         }
    //     }
    // }
    // std::cout << "Data match" << std::endl;

    size_t *big_post_data_buffer = new size_t[nb_tasks];
    std::thread process_thread([&big_data_buffer, &nb_tasks, &nb_single_data_len, &big_post_data_buffer](){
        for (size_t i = 0; i < nb_tasks; i++)
        {
            size_t min {SIZE_MAX};
            for (size_t j = 0; j < nb_single_data_len; j++) {
                min = std::min(min, big_data_buffer[i * nb_single_data_len + j]);
            }
            big_post_data_buffer[i] = min;
        }
    });

    process_thread.join();
    
 
    return 0;
}