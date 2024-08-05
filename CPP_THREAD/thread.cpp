#include <iostream>
#include <thread>

void print_msg(int id) {
    std::cout << "Thread " << id << " is running." << std::endl;
}

int main() {
    auto max_threads = std::thread::hardware_concurrency(); // Get the number of hardware threads
    std::cout << "Number of hardware threads: " << max_threads << std::endl;
    std::cout << "thread id:"<<std::this_thread::get_id() << std::endl;
    return 0;
}