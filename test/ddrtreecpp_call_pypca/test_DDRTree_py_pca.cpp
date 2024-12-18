#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <random>

namespace py = pybind11;

int main()
{
    py::scoped_interpreter guard{}; // 初始化 Python 解释器

    try
    {
        // 添加 DDRTree_py 目录到 Python 模块搜索路径
        std::string ddrtree_path = "./DDRTree_py"; // 当前目录下的 DDRTree_py
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("append")(ddrtree_path);

        // 导入 DDRTree 模块
        py::module ddrtree = py::module::import("DDRTree");

        // 生成 2000x2000 随机数据矩阵
        int rows = 1000000, cols = 2000;
        std::vector<double> data(rows * cols);

        // 使用随机数填充数据
        std::mt19937 generator(42);                              // 固定随机种子，方便复现
        std::normal_distribution<double> distribution(0.0, 1.0); // 正态分布

        for (auto &value : data)
        {
            value = distribution(generator);
        }

        // 将数据传递给 py::array_t
        py::array_t<double> C({rows, cols}, data.data());

        // 主成分数量 L
        int L = 10;

        // 调用 pca_projection_python
        auto pca_result = ddrtree.attr("pca_projection_python")(C, L);

        // 打印结果
        std::cout << "PCA projection result:" << std::endl;
        auto result = pca_result.cast<py::array_t<double>>().unchecked<2>();

        for (ssize_t i = 0; i < result.shape(0); ++i)
        {
            for (ssize_t j = 0; j < result.shape(1); ++j)
            {
                std::cout << result(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (const py::error_already_set &e)
    {
        std::cerr << "Python Error: " << e.what() << std::endl;
    }

    return 0;
}
