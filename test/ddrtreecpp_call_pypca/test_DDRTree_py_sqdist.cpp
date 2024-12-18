#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // 初始化 Python 解释器

    try {
        // 添加 DDRTree_py 目录到 Python 模块搜索路径
        std::string ddrtree_path = "./DDRTree_py"; // 当前目录下的 DDRTree_py
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("append")(ddrtree_path);

        // 导入 DDRTree 模块
        py::module ddrtree = py::module::import("DDRTree");

        // 测试调用 sqdist
        std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> data_b = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};

        py::array_t<double> a({2, 3}, data_a.data());
        py::array_t<double> b({2, 3}, data_b.data());

        auto sqdist_result = ddrtree.attr("sqdist_python")(a, b);
        std::cout << "sqdist result:" << std::endl;

        auto r = sqdist_result.cast<py::array_t<double>>().unchecked<2>();
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            for (ssize_t j = 0; j < r.shape(1); ++j) {
                std::cout << r(i, j) << " ";
            }
            std::cout << std::endl;
        }
    } catch (const py::error_already_set &e) {
        std::cerr << "Python Error: " << e.what() << std::endl;
    }

    return 0;
}
