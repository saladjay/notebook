#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <queue>
#include <chrono>
using namespace std;

condition_variable cond_var;
mutex m;
int main()
{
    int c = 0;
    bool done = false;
    cout << boolalpha;
    queue<int> goods;

    thread producer([&](){
        for (int i = 0; i < 10; ++i) {
            m.lock();
            goods.push(i);
            c++;
            cout << "produce " << i << endl;
            m.unlock();
            cond_var.notify_one();
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        done = true;
        cout << "producer done." << endl;
        cond_var.notify_one();
    });

    thread consumer([&](){
        unique_lock<mutex> lock(m);
        while(!done || !goods.empty()){
            /*
            cond_var.wait(lock, [&goods, &done](){
                        cout << "spurious wake check" << done <<endl;
                        return (!goods.empty() || done);
            });
            */  
            while(goods.empty())
            {
                cout<< "consumer wait" <<endl;
                cout<< "consumer owns lock " << lock.owns_lock() <<endl;
                cond_var.wait(lock);
            }
            if (!goods.empty()){
                cout << "consume " << goods.front()<<endl;
                goods.pop();
                c--;
            }
        }
    });

    producer.join();
    consumer.join();
    cout << "Net: " << c << endl;
}