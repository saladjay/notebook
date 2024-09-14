#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <io.h>
#include <iomanip>
#include "NvInfer.h"



// result value check of cuda runtime
#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

using namespace nvinfer1;

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)    (CEIL_DIVIDE(X, Y) * (Y))

void loadPluginFile(const std::string &path);

template<typename T>
__global__ static void printGPUKernel(T const *const in, int const n);

// Do not enable this function here, it leads to many errors about cub
template<typename T>
void printGPU(T const *const in, int const n = 10, cudaStream_t stream = 0);

// TensorRT journal
class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

int readEngine(const std::string &engine_path, char *&trt_model_stream, size_t &size)
{
    int ret = 0;

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        ret = -1;
        std::cerr << "file is not good" << std::endl;
        return ret;
    }
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if (!static_cast<bool>(trt_model_stream))
    {
        ret = -2;
        std::cerr << "failed to alloc memory for read engine" << std::endl;
        return ret;
    }
    file.read(trt_model_stream, size);
    file.close();

    return ret;
}


// Print data in the array
template<typename T>
void printArrayRecursion(const T *pArray, Dims32 dim, int iDim, int iStart)
{
    if (iDim == dim.nbDims - 1)
    {
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << double(pArray[iStart + i]) << " ";
        }
    }
    else
    {
        int nElement = 1;
        for (int i = iDim + 1; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            printArrayRecursion<T>(pArray, dim, iDim + 1, iStart + i * nElement);
        }
    }
    std::cout << std::endl;
    return;
}

// Get the size in byte of a TensorRT data type
// Get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    // case DataType::kINT64:
    //     return 8;
    default:
        return 4;
    }
}

// Get the string of a TensorRT shape
// Get the string of a TensorRT shape
std::string shapeToString(Dims32 dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    case DataType::kUINT8:
        return std::string("UINT8");
    case DataType::kFP8:
        return std::string("FP8  ");
    // case DataType::kINT64:
    //     return std::string("INT64");
    default:
        return std::string("Unknown");
    }
}







template<typename T>
void printArrayInformation(
    T const *const     pArray,
    std::string const &name,
    Dims32 const      &dim,
    bool const         bPrintInformation = false,
    bool const         bPrintArray       = false,
    int const          n                 = 10)
{
    // Print shape information
    //int nElement = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>());
    std::cout << std::endl;
    std::cout << name << ": " << typeid(T).name() << ", " << shapeToString(dim) << std::endl;

    // Print statistic information of the array
    if (bPrintInformation)
    {
        int nElement = 1; // number of elements with batch dimension
        for (int i = 0; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }

        double sum      = double(pArray[0]);
        double absSum   = double(fabs(double(pArray[0])));
        double sum2     = double(pArray[0]) * double(pArray[0]);
        double diff     = 0.0;
        double maxValue = double(pArray[0]);
        double minValue = double(pArray[0]);
        for (int i = 1; i < nElement; ++i)
        {
            sum += double(pArray[i]);
            absSum += double(fabs(double(pArray[i])));
            sum2 += double(pArray[i]) * double(pArray[i]);
            maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
            minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
            diff += abs(double(pArray[i]) - double(pArray[i - 1]));
        }
        double mean = sum / nElement;
        double var  = sum2 / nElement - mean * mean;

        std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
        std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
        std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
        std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
        std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
        std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
        std::cout << std::endl;

        // print first n element and last n element
        for (int i = 0; i < n; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
        for (int i = nElement - n; i < nElement; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
    }

    // print the data of the array
    if (bPrintArray)
    {
        printArrayRecursion<T>(pArray, dim, 0, 0);
    }

    return;
};

void printNetwork(INetworkDefinition *network);

std::vector<ITensor *> buildMnistNetwork(IBuilderConfig *config, INetworkDefinition *network, IOptimizationProfile *profile);

// plugin debug function
#ifdef DEBUG
    #define WHERE_AM_I() printf("%14p[%s]\n", this, __func__);
    #define PRINT_FORMAT_COMBINATION()                                    \
        do                                                                \
        {                                                                 \
            std::cout << "    pos=" << pos << ":[";                       \
            for (int i = 0; i < nbInputs + nbOutputs; ++i)                \
            {                                                             \
                std::cout << dataTypeToString(inOut[i].desc.type) << ","; \
            }                                                             \
            std::cout << "],[";                                           \
            for (int i = 0; i < nbInputs + nbOutputs; ++i)                \
            {                                                             \
                std::cout << formatToString(inOut[i].desc.format) << ","; \
            }                                                             \
            std::cout << "]->";                                           \
            std::cout << "res=" << res << std::endl;                      \
        } while (0);

#else
    #define WHERE_AM_I()
    #define PRINT_FORMAT_COMBINATION()
#endif // ifdef DEBUG




using namespace nvinfer1;

const std::string trtFile {"model.trt"};
const char       *inputTensorName {"inputT0"};
Dims32            shape {3, {3, 4, 5}};
static Logger     gLogger(ILogger::Severity::kERROR);


void run()
{
    IRuntime    *runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine {nullptr};

    if (_access(trtFile.c_str(), 0) == 0)
    {
        char  *trt_model_stream{nullptr};
        size_t size{0};
        readEngine(trtFile, trt_model_stream, size);
        engine = runtime->deserializeCudaEngine(trt_model_stream, size, nullptr);
    }
    else
    {
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(0);
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();

        ITensor *inputTensor = network->addInput(inputTensorName, DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        std::cout<<"abcde"<<std::endl;
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        std::cout << __FILE__ << ":" << __LINE__ << " engineString=" << engineString << std::endl;
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        std::cout << __FILE__ << ":" << __LINE__ << " engineString=" << engineString << std::endl;
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        std::cout << __FILE__ << ":" << __LINE__ << " engineString=" << engineString << std::endl;
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        std::cout << __FILE__ << ":" << __LINE__ << " engineString=" << engineString << std::endl;
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << __FILE__ << ":" << __LINE__ << " engineString=" << engineString << std::endl;
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Fail building engine" << std::endl;
            return;
        }
        std::cout << "Succeed building engine" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Fail saving engine" << std::endl;
            return;
        }
        std::cout << "Succeed saving engine (" << trtFile << ")" << std::endl;

        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    }

    if (engine == nullptr)
    {
        std::cout << "Fail getting engine for inference" << std::endl;
        return;
    }
    std::cout << "Succeed getting engine for inference" << std::endl;

    int const                 nIO = engine->getNbIOTensors();
    std::vector<const char *> tensorNameList(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList[i] = engine->getIOTensorName(i);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(inputTensorName, shape);

    for (auto const name : tensorNameList)
    {
        TensorIOMode mode = engine->getTensorIOMode(name);
        std::cout << (mode == TensorIOMode::kINPUT ? "Input " : "Output");
        std::cout << "-> ";
        std::cout << dataTypeToString(engine->getTensorDataType(name)) << ", ";
        std::cout << shapeToString(engine->getTensorShape(name)) << ", ";
        std::cout << shapeToString(context->getTensorShape(name)) << ", ";
        std::cout << name << std::endl;
    }

    std::map<std::string, std::tuple<void *, void *, int>> bufferMap;
    for (auto const name : tensorNameList)
    {
        Dims32 dim {context->getTensorShape(name)};
        int    nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
        void  *hostBuffer   = (void *)new char[nByte];
        void  *deviceBuffer = nullptr;
        CHECK(cudaMalloc(&deviceBuffer, nByte));
        bufferMap[name] = std::make_tuple(hostBuffer, deviceBuffer, nByte);
    }

    float *pInputData = static_cast<float *>(std::get<0>(bufferMap[inputTensorName])); // We certainly know the data type of input tensors
    for (int i = 0; i < std::get<2>(bufferMap[inputTensorName]) / sizeof(float); ++i)
    {
        pInputData[i] = float(i);
    }

    for (auto const name : tensorNameList)
    {
        context->setTensorAddress(name, std::get<1>(bufferMap[name]));
    }

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            void *hostBuffer   = std::get<0>(bufferMap[name]);
            void *deviceBuffer = std::get<1>(bufferMap[name]);
            int   nByte        = std::get<2>(bufferMap[name]);
            CHECK(cudaMemcpy(deviceBuffer, hostBuffer, nByte, cudaMemcpyHostToDevice));
        }
    }

    context->enqueueV3(0);

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
        {
            void *hostBuffer   = std::get<0>(bufferMap[name]);
            void *deviceBuffer = std::get<1>(bufferMap[name]);
            int   nByte        = std::get<2>(bufferMap[name]);
            CHECK(cudaMemcpy(hostBuffer, deviceBuffer, nByte, cudaMemcpyDeviceToHost));
        }
    }

    for (auto const name : tensorNameList)
    {
        void *hostBuffer = std::get<0>(bufferMap[name]);
        printArrayInformation(static_cast<float *>(hostBuffer), name, context->getTensorShape(name), false, true);
    }

    for (auto const name : tensorNameList)
    {
        void *hostBuffer   = std::get<0>(bufferMap[name]);
        void *deviceBuffer = std::get<1>(bufferMap[name]);
        delete[] static_cast<char *>(hostBuffer);
        CHECK(cudaFree(deviceBuffer));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}