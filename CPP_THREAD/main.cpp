#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void wait_thread(){
    std::unique_lock<std::mutex> lck(mtx);
    while(!ready){
        std::cout<<"Waiting...\n";
        if(cv.wait_for(lck, std::chrono::seconds(1)) == std::cv_status::timeout){
            std::cout<<"Timeout\n";
        }else{
            std::cout<<"Notified\n";
        }
    }    
    std::cout << "Thread " << std::this_thread::get_id() << " continue execution." << std::endl;
}

void notify_thread(){
    std::this_thread::sleep_for(std::chrono::seconds(5));
    {
        std::lock_guard<std::mutex> lck(mtx);
        ready = true;
    }
    cv.notify_one();
}

int main(){
    std::thread t1(wait_thread);
    std::thread t2(notify_thread);
    t1.join();
    t2.join();
    return 0;
}
