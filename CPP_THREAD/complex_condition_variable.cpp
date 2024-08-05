#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>


class Semaphore_ready {
public:
    Semaphore_ready() = default;
    ~Semaphore_ready() = default;
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_one();
    }

    void wait() {
        std::cout<<"wait 0 name:"<< std::this_thread::get_id()<<std::endl;
        std::unique_lock<std::mutex> lock(mutex_);
        std::cout<<"wait 1 name:"<< std::this_thread::get_id()<<std::endl;
        cv_.wait(lock);
        std::cout<<"wait 2 name:"<< std::this_thread::get_id()<<std::endl;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;

};


int main() {
    Semaphore_ready sem1;
    Semaphore_ready sem2;

    std::thread t1([&]() {
        for (int i = 0; i < 20; i++) {
            sem1.wait();
            std::cout << " t1 step1: " << i << std::endl;      
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "t1 step2: " << i; 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));   
            sem2.notify();
    }});            

    std::thread t2([&]() {
        for (int i = 0; i < 20; i++) {
            std::cout << "t2 step1: " << i;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));            
            sem1.notify();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            sem2.wait();
            std::cout << " t1 step2: " << i << std::endl;   
        }            
    });

    t1.join();
    t2.join();

    return 0;
}



 
