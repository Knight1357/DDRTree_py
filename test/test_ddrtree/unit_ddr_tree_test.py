import unittest
import numpy as np
try:
    from ddr_tree import DDRTree_reduce_dim
except ImportError:
    print("无法导入 ddr_tree 模块或 DDRTree_reduce_dim 函数，请检查模块是否安装正确及函数名是否准确。")
    raise

class TestDDRTreeReduceDim(unittest.TestCase):

    def setUp(self):
        """
        在每个测试用例执行前设置测试参数和生成测试数据
        """
        # 设置随机种子，保证每次生成的数据可重复（便于调试）
        np.random.seed(42)

        # 减小参数数值，以产生相对小一些的数据量用于测试
        self.n_samples = 1000      # 样本数量（N）
        self.n_features = 50      # 原始特征维度（D）
        self.dimensions = 2       # 降维后的维度（d）
        self.num_clusters = 3     # 聚类数量（K）
        self.maxiter = 30         # 最大迭代次数
        self.sigma = 1e-3         # 高斯核参数
        self.lambda_ = 0.1        # 正则化参数
        self.gamma = 10           # 权重参数
        self.eps = 1e-3           # 收敛阈值
        self.verbose = True       # 是否输出详细信息

        # 生成可控的随机数据
        self.R_X = np.random.rand(self.n_samples, self.n_features)          # (N x D)
        self.R_Z = np.random.rand(self.dimensions, self.n_samples)          # (d x N)
        self.R_Y = np.random.rand(self.dimensions, self.num_clusters)       # (d x K)
        self.R_W = np.random.rand(self.n_features, self.dimensions)         # (D x d)

    def test_DDRTree_reduce_dim(self):
        """
        测试DDRTree_reduce_dim函数的功能
        """
        result = DDRTree_reduce_dim(
            self.R_X, self.R_Z, self.R_Y, self.R_W,
            self.dimensions, self.maxiter, self.num_clusters,
            self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
        )

        # 断言返回结果是一个字典类型
        self.assertIsInstance(result, dict)

        # 检查字典中是否包含预期的键
        expected_keys = ['W', 'Z', 'stree', 'Y', 'Q', 'R', 'objective_vals']
        for key in expected_keys:
            self.assertIn(key, result)

        # 检查返回结果中各矩阵形状是否符合预期
        self.assertEqual(result['W'].shape, (self.n_features, self.dimensions))
        self.assertEqual(result['Z'].shape, (self.dimensions, self.n_samples))
        self.assertEqual(result['stree'].shape, (self.n_samples, self.n_samples))  # 稀疏矩阵转为密集矩阵后的形状
        self.assertEqual(result['Y'].shape, (self.dimensions, self.num_clusters))
        self.assertEqual(result['Q'].shape, (self.dimensions, self.n_samples))
        self.assertEqual(result['R'].shape, (self.dimensions, self.n_samples))

        # 检查 objective_vals 是否为数值类型并且其长度符合预期
        self.assertIsInstance(result['objective_vals'], np.ndarray)
        self.assertEqual(result['objective_vals'].ndim, 1)
        self.assertGreater(len(result['objective_vals']), 0)

    def test_invalid_input(self):
        """
        测试无效输入数据的处理
        """
        # 测试：输入数据维度不符
        with self.assertRaises(ValueError):
            DDRTree_reduce_dim(
                self.R_X.T, self.R_Z, self.R_Y, self.R_W,
                self.dimensions, self.maxiter, self.num_clusters,
                self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
            )

        # 测试：输入数据类型不符
        with self.assertRaises(ValueError):
            DDRTree_reduce_dim(
                self.R_X.astype(np.float32), self.R_Z, self.R_Y, self.R_W,
                self.dimensions, self.maxiter, self.num_clusters,
                self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
            )

        # 测试：缺少必要的参数
        with self.assertRaises(TypeError):
            DDRTree_reduce_dim(
                self.R_X, self.R_Z, self.R_Y, self.R_W,
                self.dimensions, self.maxiter, self.num_clusters,
                self.sigma, self.lambda_, self.gamma, self.eps
            )

if __name__ == '__main__':
    unittest.main()
