#include <iostream>
#include <fstream>

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
        int32_t nb_layers;
        file2.read(reinterpret_cast<char *>(&nb_layers), sizeof(nb_layers));
        std::cout << "nb_layers:" << nb_layers << std::endl;
        for (int32_t i = 0; i < nb_layers; i++)
        {
            int32_t nb_name_length;
            file2.read(reinterpret_cast<char *>(&nb_name_length), sizeof(nb_name_length));
            std::cout << "nb_name_length:" << nb_name_length << std::endl;

            char* name = new char[nb_name_length];
            file2.read(name, nb_name_length);
            std::cout << "name:" << name << std::endl;

            int32_t nb_weights;
            file2.read(reinterpret_cast<char *>(&nb_weights), sizeof(nb_weights));
            std::cout << "nb_weights:" << nb_weights << std::endl;

            for (int i = 0; i < nb_weights; i++)
            {
                double weight;
                file2.read(reinterpret_cast<char *>(&weight), sizeof(weight));
                std::cout << "weight:" << weight;
            }
            std::cout << std::endl;
        }
    }
}