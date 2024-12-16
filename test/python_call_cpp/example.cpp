#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// C++函数：两个整数相加
int add(int a, int b) {
    return a + b;
}

// C++函数：返回一个问候字符串
std::string say_hello(const std::string &name) {
    return "Hello, " + name + "!";
}

// 定义 Pybind11 模块
PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "Example module written in C++ using pybind11"; // 模块文档字符串
    m.def("add", &add, "A function that adds two numbers");
    m.def("say_hello", &say_hello, "A function that greets the user");
}
