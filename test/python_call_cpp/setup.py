from setuptools import setup, Extension
import pybind11

# 获取 pybind11 的头文件路径
pybind11_include = pybind11.get_include()

# 定义扩展模块
ext_modules = [
    Extension(
        "example",                    # 模块名称
        ["example.cpp"],              # C++ 源文件
        include_dirs=[pybind11_include], # Pybind11 的头文件路径
        language="c++"                   # 指定语言为 C++
    )
]

# 使用 setup() 构建模块
setup(
    name="example",
    version="0.1",
    description="A simple example of calling C++ code from Python using pybind11",
    ext_modules=ext_modules,
)
