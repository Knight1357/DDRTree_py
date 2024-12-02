// pybind11 头文件和命名空间
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // 用于 Eigen 类型支持
#include <Eigen/Dense>
namespace py = pybind11;

double add(double i, double j)
{
    py::print("hello");
    return i + j;
}

py::dict test_array_input(py::array_t<double> input_array)
{
    // 检查输入是否为 2D numpy array
    if (input_array.ndim() != 2 || input_array.dtype() != py::dtype::of<double>())
    {
        throw std::invalid_argument("Input must be a 2D numpy array of type float64.");
    }

    // 打印输入数组的基本信息
    py::print("Input array shape:", input_array.shape(0), "x", input_array.shape(1));
    py::print("Is C-contiguous:", input_array.flags() & py::array::c_style ? "Yes" : "No");
    py::print("Is Fortran-contiguous:", input_array.flags() & py::array::f_style ? "Yes" : "No");

    // 将 numpy 数组映射到 Eigen 矩阵
    Eigen::Map<const Eigen::MatrixXd> matrix(input_array.data(), input_array.shape(0), input_array.shape(1));
    py::print("Successfully mapped to Eigen matrix.");

    // 返回输入矩阵的行数和列数作为结果
    py::dict result;
    result["rows"] = matrix.rows();
    result["cols"] = matrix.cols();

    // 计算矩阵的元素和并返回
    double sum = matrix.sum();
    result["sum"] = sum;
    py::print("Matrix sum:", sum);

    return result;
}

PYBIND11_MODULE(example, m)
{
    // 可选，说明这个模块是做什么的
    m.doc() = "pybind11 example plugin";
    // def("给python调用方法名"， &实际操作的函数， "函数功能说明"，默认参数). 其中函数功能说明为可选
    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 1.0, py::arg("j") = 2.0);
    m.def("test_array_input", &test_array_input, "Test function for py::array_t<double>");
}
