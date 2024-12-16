#include <pybind11/embed.h> // 引入 pybind11 嵌入模块
#include <iostream>

namespace py = pybind11;

int main() {
    // 初始化 Python 解释器
    py::scoped_interpreter guard{};

    try {
        // 导入 example.py 模块
        py::module script = py::module::import("example");

        // 调用 add 函数
        int result_add = script.attr("add")(3, 5).cast<int>();
        std::cout << "add(3, 5) = " << result_add << std::endl;

        // 调用 say_hello 函数
        std::string result_say_hello = script.attr("say_hello")("World").cast<std::string>();
        std::cout << result_say_hello << std::endl;
    } catch (const std::exception &e) {
        // 捕获任何异常
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
