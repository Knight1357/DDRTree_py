from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import numpy as np
import pybind11

class build_ext_subclass(build_ext):
    def run(self):
        # 如果需要 Boost 或其他依赖，这里可以动态设置环境变量
        os.environ['CXXFLAGS'] = f"-I{os.getenv('CONDA_PREFIX', '')}/include"
        os.environ['LIBRARY_PATH'] = f"{os.getenv('CONDA_PREFIX', '')}/lib"
        build_ext.run(self)

# 定义 C++ 扩展模块
ext_modules = [
    Extension(
        name='ddr_tree_py.ddr_tree_wrapper',  # Python 中的模块名称
        sources=[
            'ddr_tree_py/_core/DDRTree_wrapper.cpp',
            'ddr_tree_py/_core/DDRTree.cpp',
        ],
        include_dirs=[
            'ddr_tree_py/_core',  # C++ 头文件所在目录
            pybind11.get_include(),  # pybind11 的头文件
            np.get_include(),  # NumPy 的头文件
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],  # 使用 C++11 标准
    )
]

# 配置 setup
setup(
    name='ddr-tree-py',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python package for DDRTree, with C++ implementation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ddr_tree',
    packages=['ddr_tree_py'],  # 仅包含 Python 包 ddr_tree
    ext_modules=ext_modules,  # 指定 C++ 扩展模块
    cmdclass={'build_ext': build_ext_subclass},  # 自定义构建类
    install_requires=[
        'pybind11',  # 绑定工具
        'numpy',  # 数值计算
        'scipy',  # 线性代数
    ],
    tests_require=['pytest'],  # 测试工具
    test_suite='tests',  # 测试目录
)
