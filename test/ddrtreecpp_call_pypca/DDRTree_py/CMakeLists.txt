cmake_minimum_required(VERSION 3.12)
project(call_python_from_cpp LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 手动指定 Python3 的路径（确保路径正确）
set(Python3_EXECUTABLE /opt/miniconda/envs/r42/bin/python3)
set(Python3_INCLUDE_DIR /opt/miniconda/envs/r42/include/python3.12)
set(Python3_LIBRARY /opt/miniconda/envs/r42/lib/libpython3.12.so)

# 查找 Python3 和 Pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# # 指定源文件 (test_DDRTree_py_sqdist.cpp 位于上一级目录)
# set(SOURCES
#     ../test_DDRTree_py_sqdist.cpp
# )

# # 添加可执行程序
# add_executable(call_python_from_cpp ${SOURCES})

# # 链接 Pybind11 和 Python 库
# target_link_libraries(call_python_from_cpp
#     PRIVATE pybind11::embed   # 链接 Pybind11 的嵌入功能
#     Python3::Python           # 链接 Python3 库
# )

set(SOURCES
    ../test_DDRTree_py_pca.cpp
)

add_executable(call_pca_python ${SOURCES})

target_link_libraries(call_pca_python
    PRIVATE pybind11::embed
    Python3::Python
)

# 设置可执行文件的输出目录
set_target_properties(call_pca_python PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../
)

# 输出调试信息
message(STATUS "Python3 Interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python3 Library: ${Python3_LIBRARY}")
message(STATUS "Executable will be placed in: ${CMAKE_SOURCE_DIR}/../")
