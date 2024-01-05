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

