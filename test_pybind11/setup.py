from setuptools import setup, Extension
import pybind11

# 使用 pybind11 和 setuptools 创建 Python 扩展
ext_modules = [
    Extension(
        "example",  # 扩展模块的名称
        ["example.cpp"],  # C++ 源文件
        include_dirs=[
            pybind11.get_include(),  # pybind11 的头文件目录
            "/opt/miniconda/envs/r42/lib/R/library/Rcpp/include",
            "/opt/miniconda/envs/r42/lib/R/library/RcppEigen/include",
            "/opt/miniconda/envs/r42/lib/R/include"
        ],
        language="c++",  # 指定语言为 C++
        extra_compile_args=["-std=c++11"],  # 编译选项
    )
]

# 使用 setuptools 设置
setup(
    name="example",  # 模块名称
    version='1.0',
    ext_modules=ext_modules,  # 扩展模块
    install_requires=["numpy"],  # 安装依赖
)
