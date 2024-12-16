cmake_minimum_required(VERSION 3.14)
project(Pybind11Example)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 手动指定 Python3 的路径（调整为你的实际路径）
set(Python3_EXECUTABLE /opt/miniconda/envs/r42/bin/python3)
set(Python3_INCLUDE_DIR /opt/miniconda/envs/r42/include/python3.12)
set(Python3_LIBRARY /opt/miniconda/envs/r42/lib/libpython3.12.so)

# 找到 Python3 和 Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 添加可执行文件
add_executable(example example.cpp)

# 链接 Pybind11 和 Python3 库
target_link_libraries(example PRIVATE pybind11::embed Python3::Python)
