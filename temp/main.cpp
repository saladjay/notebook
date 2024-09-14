#include <iostream>
#include <memory>
#include <mutex>
#include <fstream>
struct alignas(float) Detection_v8_severe {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float severe;
};


int main1() {    

    std::cout<<sizeof(Detection_v8_severe)<<std::endl;
    std::cout<< sizeof(float)<<std::endl;
    return 0;
}

template<typename T>
class SingletonT 
{
public:
	//GetInstance
	template<typename ...Args>
	static std::shared_ptr<T> GetInstance(Args&&... args) {
		if (!sington_) {
			std::lock_guard<std::mutex> lock(mutex_);
			if (nullptr == sington_) {
				sington_ = std::make_shared<T>(std::forward<Args>(args)...);
			}
		}
		return sington_;
	}

	//DelInstance
	static void DelInstance() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (sington_) {
			sington_.reset();
			sington_ = nullptr;
		}
	}

private:
	SingletonT() = default;	
	SingletonT(const SingletonT&) = delete;
	SingletonT& operator=(const SingletonT&) = delete;
	~SingletonT() = default;

private:
	static std::shared_ptr<T> sington_;
	static std::mutex mutex_;
};

template<typename T>
std::shared_ptr<T> SingletonT<T>::sington_ = nullptr;

template<typename T>
std::mutex SingletonT<T>::mutex_;


template<typename T>
class SingletonLazy
{
public:
	/*static T& Instance()
	{
		static std::once_flag flag;
		std::call_once(flag, [&]() {
			instance_.reset(new T);
		});
		return *instance_;
	}*/
	
	template<typename... Args>
	static T& Instance(Args&&...args)
	{
		static std::once_flag flag;
		std::call_once(flag, [&]() {
			instance_.reset(new T(std::forward<Args>(args)...));
		});
		return *instance_;
	}

	static std::unique_ptr<T> instance_;

private:
	SingletonLazy() = default;
	SingletonLazy(const SingletonLazy&) = delete;
	SingletonLazy& operator=(const SingletonLazy&) = delete;
	~SingletonLazy() = default;
};
template<typename T>
std::unique_ptr<T> SingletonLazy<T>::instance_;

class MyClass
{
public:
	MyClass()
	{
		std::cout << "MyClass()" << std::endl;
	};

	virtual ~MyClass()
	{
		std::cout << "~MyClass()" << std::endl;
	};

	void fun()
	{
		std::cout << "this is fun." << std::endl;
	}
	void fun2()
	{
		std::cout << "this is fun2." << std::endl;
	}
private:
	std::string m_strData;
};

class MyClass1 
{
public:
	MyClass1(const std::string& data) : data_(data)
	{
		//std::cout << "MyClass1 Constructor" << std::endl;
		std::cout << data_.data() << std::endl;
	};

    MyClass1()
	{
		std::cout << "MyClass1 Constructor" << std::endl;
		// std::cout << data_.data() << std::endl;
	};

	virtual ~MyClass1()
	{
		std::cout << "MyClass1 Destructor "<< (void*)this << std::endl;
	};

	void fun()
	{
		std::cout << "this is fun." << std::endl;
	}
	void fun2()
	{
		std::cout << "this is fun2." << std::endl;
	}
private:
	std::string data_;
};


std::shared_ptr<MyClass1> getClass1(){
    static std::shared_ptr<MyClass1> instance;
    if (!instance) {
        instance = SingletonT<MyClass1>::GetInstance();
    }
    return instance;
}



int main()
{
	std::cout << "Hello World!\n";

    std::ofstream ofs("test.txt", std::ios::out);
    if(ofs.is_open()){
        ofs << "Hello World!\n";
        ofs.close();
    }

    // size_t width{20}, height{20}, channels{3};
    // uint8_t *src = new uint8_t[width * height * channels];
    // memset(src, 0, width * height);
    // memset(src + width * height , 100, width * height);
    // memset(src + width * height * 2, 200, width * height);
    // for (size_t i = 0; i < width * height * channels; i++) {
    //     std::cout << static_cast<int> (src[i]) << " ";
    // }

    // auto inst = getClass1();
    // inst->fun();
    // inst->fun2();

    // auto inst1 = getClass1();
    // inst1->fun();
    // inst1->fun2();

    // auto inst2 = getClass1();
    // inst2->fun();
    // inst2->fun2();
	
	// //SingletonT(std::shared_ptr方式)
	// auto _inst = SingletonT<MyClass1>::GetInstance("MyClass1 Constructor");
	// SingletonT<MyClass1>::DelInstance();

	// //SingletonLazy(std::unique_ptr + std::call_once)
	// auto instMyClass1 = SingletonLazy<MyClass1>::Instance("MyClass1");
	// instMyClass1.fun();
	// instMyClass1.fun2();

	// auto instMyClass = SingletonLazy<MyClass>::Instance();
	// instMyClass.fun();
	// instMyClass.fun2();

	system("pause");
	return 1;
}
