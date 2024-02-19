# Torch

## Linux

运行环境版本：

- ubuntu 1804
- cuda runtime 11.8
- cudnn 8.9.3
- driver 535.98 / cuda driver 12.2
- torch 1.13.0

### 安装依赖

最好创建一个新的虚拟环境

```shell
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```



### 下载仓库

```shell
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0
```

执行git submodule update --init --recursive --jobs 0 巨慢无比。可以通过搭梯子或者修改git source来解决

### 修改内容

编译原因torch1.13.0没有兼容11.8的版本。

1. 需要修改setup.py的cmdclass。不然会遇到

   CMake Error : file INSTALL cannot set permissions on "/usr/local/include": Operation not permitted。目测是因为setup.py里使用一个自定义的install函数替换了setuptools.command.install.install函数

```python
    cmdclass = {
        'bdist_wheel': wheel_concatenate,
        'build_ext': build_ext,
        'clean': clean,
        'install': setuptools.command.install.install,
        'sdist': sdist,
    }
```

2. 官方版本的torch是启用BUILD_SPLIT_CUDA的，在CMakeLists.txt里是没有默认启用的。启用该变量可以拆分torch_cuda，拆成libtorch_cuda_cpp.so和libtorch_cuda_cpp.so。如果想要保持.so名字和官方一致，需要在编译前执行

   ```shell
   export BUILD_SPLIT_CUDA=1
   ```

3. 增加CUDA version，在源码torch/utils/cpp_extension.py的文件里，修改CUDA_GCC_VERSIONS和CUDA_CLANG_VERSIONS里增加CUDA11.8的内容。

   ```python
   # The following values were taken from the following GitHub gist that
   # summarizes the minimum valid major versions of g++/clang++ for each supported
   # CUDA version: https://gist.github.com/ax3l/9489132
   CUDA_GCC_VERSIONS = {
       '10.2': (MINIMUM_GCC_VERSION, (8, 0, 0)),
       '11.1': (MINIMUM_GCC_VERSION, (10, 0, 0)),
       '11.2': (MINIMUM_GCC_VERSION, (10, 2, 1)),
       '11.3': (MINIMUM_GCC_VERSION, (10, 2, 1)),
       '11.4': ((6, 0, 0), (11, 5, 0)),
       '11.5': ((6, 0, 0), (11, 5, 0)),
       '11.6': ((6, 0, 0), (11, 5, 0)),
       '11.7': ((6, 0, 0), (11, 5, 0)),
       '11.8': ((6, 0, 0), (11, 5, 0)),
   }
   
   CUDA_CLANG_VERSIONS = {
       '10.2': ((3, 3, 0), (8, 0, 0)),
       '11.1': ((6, 0, 0), (9, 0, 0)),
       '11.2': ((6, 0, 0), (9, 0, 0)),
       '11.3': ((6, 0, 0), (11, 0, 0)),
       '11.4': ((6, 0, 0), (11, 0, 0)),
       '11.5': ((6, 0, 0), (12, 0, 0)),
       '11.6': ((6, 0, 0), (12, 0, 0)),
       '11.7': ((6, 0, 0), (13, 0, 0)),
       '11.8': ((6, 0, 0), (13, 0, 0)),
   }
   ```

   另外在supported_arches里增加8.9

   ```python
       supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                           '7.0', '7.2', '7.5', '8.0', '8.6', '8.9']
   ```

   上诉这两项是为了方便detectron2调用torch。不然会出现不兼容sm_arch的报错而无法导入和正确在40系显卡上编译torchvision

   

### 编译

```shell
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

### 打包

```
python setup.py bdist_wheel
```

## Windows

[Visual Studio 较旧的下载 - 2019、2017、2015 和以前的版本 (microsoft.com)](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)

## Building from source

### Include optional components

There are two supported components for Windows PyTorch: MKL and MAGMA. Here are the steps to build with them.

```bat
REM Make sure you have 7z and curl installed.

REM Download MKL files
curl https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z -k -O
"C:/Program Files/7-Zip/7z" x -aoa mkl_2020.2.254.7z -omkl

REM Download MAGMA files
REM version available:
REM 2.5.4 (CUDA 10.1 10.2 11.0 11.1) x (Debug Release)
REM 2.5.3 (CUDA 10.1 10.2 11.0) x (Debug Release)
REM 2.5.2 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
REM 2.5.1 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
set CUDA_PREFIX=cuda102
set CONFIG=release
curl -k https://s3.amazonaws.com/ossci-windows/magma_2.5.4_%CUDA_PREFIX%_%CONFIG%.7z -o magma.7z
"C:/Program Files/7-Zip/7z" x -aoa magma.7z -omagma

REM Setting essential environment variables
set "CMAKE_INCLUDE_PATH=%cd%\mkl\include"
set "LIB=%cd%\mkl\lib;%LIB%"
set "MAGMA_HOME=%cd%\magma"
```



### Speeding CUDA build for Windows

Visual Studio doesn’t support parallel custom task currently. As an alternative, we can use `Ninja` to parallelize CUDA build tasks. It can be used by typing only a few lines of code.

```bat
REM Let's install ninja first.
pip install ninja

REM Set it as the cmake generator
set CMAKE_GENERATOR=Ninja
```



根据一下官方编译信息，可以得知，visual studio 2019编译会相对顺利，事实上也是如此，visual studio 2017中的msvc 14.16编译会卡住，具体现象和原因没有保存图片。把下面的CMAKE_GENERATOR_TOOLSET_VERSION改成14.16，可以设置成visual studioi 2017的msvc 14.16进行编译。还需要手动设置一次对应目录的CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\cl.exe

```bat
git checkout v1.13.1-rc1
git submodule sync
git submodule update --init --recursive --jobs 0
```

```bat
:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as `Could NOT find OpenMP`.
:: 这个不必做，因为下载magma和mkl的时候已经设置好了
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake >= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py install
```

```bat
D:
cd github\pytorch
set "CMAKE_INCLUDE_PATH=%cd%\mkl\include"
set "LIB=%cd%\mkl\lib;%LIB%"
set "MAGMA_HOME=%cd%\magma"
conda activate pytorch_compile_env
set CMAKE_GENERATOR=Ninja
set DISTUTILS_USE_SDK=1
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe
python setup.py install
python setup.py bdist_wheel
```

