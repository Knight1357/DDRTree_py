cmake_minimum_required(VERSION 3.12)
project(DDRTree_cpp)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS system filesystem)

# 如果找到 Boost，打印相关信息
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})  # 添加 Boost 头文件路径
    link_directories(${Boost_LIBRARY_DIRS})    # 添加 Boost 库路径
endif()

# 手动指定 Python3 的路径（调整为你的实际路径）
set(Python3_EXECUTABLE /opt/miniconda/envs/r42/bin/python3)
set(Python3_INCLUDE_DIR /opt/miniconda/envs/r42/include/python3.12)
set(Python3_LIBRARY /opt/miniconda/envs/r42/lib/libpython3.12.so)

# 查找Pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# 添加子目录，包含DDRTree_cpp的代码
include_directories(${CMAKE_SOURCE_DIR}/DDRTree_cpp)

# 添加Pybind11模块
pybind11_add_module(DDRTree_cpp
    Pybind11Exports.cpp  # Pybind11绑定代码
    DDRTree.cpp          # 核心C++代码
)

# 链接Python库
target_link_libraries(DDRTree_cpp PRIVATE Python3::Python)

# 设置输出目录为 DDRTree_py
set_target_properties(DDRTree_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../DDRTree_py
    OUTPUT_NAME "DDRTree_cpp"  # 输出文件名
)

# 输出调试信息，确认路径
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Shared library will be placed in: ${CMAKE_SOURCE_DIR}/../DDRTree_py")
