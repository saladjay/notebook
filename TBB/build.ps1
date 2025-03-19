cd oneTBB
# powershell 检测目录是否存在
if (-not (Test-Path build)) {
    mkdir build
}
cd build
# powershell 打印当前目录
$pwd
cmake -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 2017" -T host=x64 -A x64 -DCMAKE_INSTALL_PREFIX="../../TBB" -DTBB_TEST=OFF ..
cmake --build .
cmake --install .

# cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 2017" -T host=x64 -A x64                                                                                                                                                                                                                                                                                                       
# cmake --build ./build --config release 