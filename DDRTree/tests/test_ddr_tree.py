import unittest
import numpy as np
from sklearn.datasets import load_iris
from ddr_tree_py import DDRTree

class TestDDRTree(unittest.TestCase):
    
    def setUp(self):
        # 加载一个小数据集用于测试
        iris = load_iris()
        self.X = iris.data[:10].T  # 转置以符合 (D, N) 的形状

    def test_ddr_tree_output_shape(self):
        # 简单测试以检查输出维度
        DDRTree_res = DDRTree(self.X, dimensions=2, maxIter=5, sigma=1e-2, lambda_param=1, ncenter=3, param_gamma=10, tol=1e-2, verbose=False)

        # 提取结果进行验证
        Z = DDRTree_res['Z']
        Y = DDRTree_res['Y']
        W = DDRTree_res['W']

        # 检查输出的维度
        self.assertEqual(Z.shape[0], 2, "Z 应该有 2 行。")
        self.assertEqual(Y.shape[0], 2, "Y 应该有 2 行。")
        self.assertEqual(W.shape[0], 2, "W 应该有 2 行。")

    def test_ddr_tree_parameter_constraints(self):
        # 测试使用无效参数以检查错误处理
        with self.assertRaises(ValueError):
            DDRTree(self.X, dimensions=5, maxIter=5, sigma=1e-2, lambda_param=1, ncenter=3, param_gamma=10, tol=1e-2, verbose=False)
        
        # ncenter 不应大于 X 的列数
        with self.assertRaises(ValueError):
            DDRTree(self.X, dimensions=2, ncenter=20, maxIter=5, sigma=1e-2, lambda_param=1, param_gamma=10, tol=1e-2, verbose=False)

if __name__ == '__main__':
    unittest.main()
