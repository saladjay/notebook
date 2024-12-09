#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>  // 包含 std::setprecision 和 std::fixed
int main()
{
    std::ifstream file("float_array.bin", std::ios::binary);
    if (file.is_open())
    {
        double arr[4];
        file.read(reinterpret_cast<char *>(arr), sizeof(arr));
        std::cout << arr[0] << std::endl;
    }
    file.close();

    std::ifstream file2("weight.bin", std::ios::binary);

    if (file2.is_open())
    {
        std::cout<<"bin current position:"<<file2.tellg()<<std::endl;
        int32_t nb_layers;
        file2.read(reinterpret_cast<char *>(&nb_layers), sizeof(nb_layers));
        std::cout << "nb_layers:" << nb_layers << std::endl;
        std::cout<<"bin current position:"<<file2.tellg()<<std::endl;
        for (int32_t i = 0; i < nb_layers; i++)
        {
            int32_t nb_name_length;
            file2.read(reinterpret_cast<char *>(&nb_name_length), sizeof(nb_name_length));
            std::cout << "nb_name_length:" << nb_name_length << std::endl;
            std::cout<<"bin current position:"<<file2.tellg()<<std::endl;

            char* name = new char[nb_name_length];
            file2.read(name, nb_name_length);
            std::cout << "name:" << name << std::endl;
            std::cout<<"name:";
            std::stringstream ss;
            for(size_t i=0;i<nb_name_length;i++){
                std::cout<<name[i];
                
            }
            ss.write(name,nb_name_length);
            std::cout<<std::endl;
            
            std::cout<<"name:"<<ss.str()<<std::endl;
            std::cout<<"bin current position:"<<file2.tellg()<<std::endl;

            int32_t nb_weights;
            file2.read(reinterpret_cast<char *>(&nb_weights), sizeof(nb_weights));
            std::cout << "nb_weights:" << nb_weights << std::endl;
            std::cout<<"bin current position:"<<file2.tellg()<<std::endl;

            for (int j = 0; j < nb_weights; j++)
            {
                double weight;
                file2.read(reinterpret_cast<char *>(&weight), sizeof(weight));
                std::cout << "weight1:" << weight;   
            }
            std::cout << std::endl;
            std::cout<<"bin current position:"<<file2.tellg()<<std::endl;
        }
    }

    uint32_t* arr2 = new uint32_t[4];
    float test_float{ 1.23456789 };
    memcpy(&arr2[1], &test_float, sizeof(float));
    float test_float2;
    memcpy(&test_float2, &arr2[1], sizeof(float));
    std::cout << std::setprecision(20);
    std::cout << "test_float2:" << test_float2 << std::endl;
    std::cout << "test_float1:" << test_float << std::endl;
    return 0;
}