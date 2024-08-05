#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready_Premature_notice = false;

void wait_thread_Premature_notice(){
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::unique_lock<std::mutex> lck(mtx);
    while(ready_Premature_notice){
        std::cout<<"Waiting...\n";
        if(cv.wait_for(lck, std::chrono::seconds(5)) == std::cv_status::timeout){
            std::cout<<"Timeout\n";
            break;
        }else{
            std::cout<<"Notified\n";
        }
    }    
    std::cout << __FUNCTION__ << "Thread " << std::this_thread::get_id() << " continue execution." << std::endl;
}

void notify_thread_Premature_notice(){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lck(mtx); // lock the mutex
        ready_Premature_notice = true;
    }
    cv.notify_one();
    std::cout << __FUNCTION__ << "Thread " << std::this_thread::get_id() << " notified." << std::endl;
}

bool ready = false;

void wait_thread(){
    std::unique_lock<std::mutex> lck(mtx);
    while(!ready){
        std::cout<<"Waiting...\n";
        if(cv.wait_for(lck, std::chrono::seconds(5)) == std::cv_status::timeout){
            std::cout<<"Timeout\n";
            return;
        }else{
            std::cout<<"Notified\n";
        }
    }    
    std::cout << __FUNCTION__ << "Thread " << std::this_thread::get_id() << " continue execution." << std::endl;
}

void notify_thread(){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lck(mtx); // lock the mutex
        ready = true;
    }
    cv.notify_one();
    std::cout << __FUNCTION__ << "Thread " << std::this_thread::get_id() << " notified." << std::endl;
}


std::mutex m2;
std::condition_variable cv2;
std::string data2;
bool ready2 = false;
bool processed2 = false;
 
void worker_thread()
{
    // wait until main() sends data
    std::unique_lock lk(m2);
    cv2.wait(lk, []{ return ready; });

    // after the wait, we own the lock
    std::cout << "Worker thread is processing data\n";
    data2 += " after processing";

    // send data back to main()
    processed2 = true;
    std::cout << "Worker thread signals data processing completed\n";

    // manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lk.unlock();
    cv.notify_one();
}

int example_of_cpp_reference()
{
    std::thread worker(worker_thread);

    data2 = "Example data";
    // send data to the worker thread
    {
        std::lock_guard lk(m2);
        ready = true;
        std::cout << "main() signals data ready for processing\n";
    }
    cv.notify_one();
    std::cout<<"start to wait for the worker"<<std::endl;
    // wait for the worker
    {
        std::unique_lock lk(m2);
        cv.wait(lk, []{ return processed2; });
    }
    std::cout << "Back in main(), data = " << data2.c_str() << '\n';
 
    worker.join();
}

int main(){
    std::thread t1(wait_thread_Premature_notice);
    std::thread t2(notify_thread_Premature_notice);
    t1.join();
    t2.join();

    std::this_thread::sleep_for(std::chrono::seconds(6));
    std::cout<<"premature notice example is done"<<std::endl;
    std::cout<<"----------------------------"<<std::endl;
    std::thread t3(wait_thread);
    std::thread t4(notify_thread);
    t3.join();
    t4.join();


    std::cout<<"normal example is done"<<std::endl;
    std::cout<<"----------------------------"<<std::endl;
    example_of_cpp_reference();
    return 0;
}
