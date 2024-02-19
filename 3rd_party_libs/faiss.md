# Windows编译Faiss

- Faiss 1.7.4
- Windows 10
- cmake 3.25.2
- visual studio 2017
- MKL 2024.0

安装完成后将在 `C:\Program Files (x86)\intel` 中找到，将其拷贝复制到其他目录，将其组织成下列结构：

```bash
/path/to/mkl
├─common
├─compiler
├─compiler_ide
├─mkl
│  ├─bin
│  ├─include
│  ├─lib
│  └─share
├─tbb
└─tcm
```







# Linux编译Faiss

- Faiss 1.7.4
- Ubuntu 18.04
- cmake 3.28.0-rc3
- gcc 7.5.0
- g++ 7.5.0

## 1. 下载Faiss

```shell
git clone https://github.com/facebookresearch/faiss.git
```



## 2. 下载MKL并安装

export MKLROOT=/home/pc/intel/oneapi

## 3. 编译Faiss

cmake .. -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=/home/pc/intel/oneapi/mkl/latest/ -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_INSTALL_PREFIX=../install

cmake --build . --config Release -j --target install

