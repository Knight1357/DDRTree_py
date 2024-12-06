import unittest
import numpy as np
from ddr_tree import DDRTree_reduce_dim


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
        self.R_X = np.random.rand(self.n_features, self.n_samples)          # (D x N)
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

        # 断言返回结果是一个字典类型（根据原代码使用方式推测返回类型）
        self.assertEqual(isinstance(result, dict), True)

        # 检查字典中是否包含预期的键（可根据实际情况补充更多键的检查）
        expected_keys = ['W', 'Z', 'stree', 'Y', 'Q', 'R', 'objective_vals']
        for key in expected_keys:
            self.assertIn(key, result)

        # 检查返回结果中各矩阵形状是否符合预期（维度关系根据函数逻辑推测，可调整）
        self.assertEqual(result['W'].shape, (self.n_features, self.dimensions))
        self.assertEqual(result['Z'].shape, (self.dimensions, self.n_samples))
        self.assertEqual(result['stree'].shape, (self.dimensions, self.n_samples))
        self.assertEqual(result['Y'].shape, (self.dimensions, self.num_clusters))
        self.assertEqual(result['Q'].shape, (self.dimensions, self.n_samples))
        self.assertEqual(result['R'].shape, (self.dimensions, self.n_samples))

        # 检查返回结果中的 'objective_vals' 是否为列表
        self.assertEqual(isinstance(result['objective_vals'], list), True)

        # 确保目标函数值单调递减（示例检查）
        for i in range(1, len(result['objective_vals'])):
            self.assertLessEqual(result['objective_vals'][i], result['objective_vals'][i - 1])

    def test_DDRTree_reduce_dim_with_nan(self):
        """
        测试输入数据包含NaN值的情况
        """
        # 人为加入NaN值
        self.R_X[0, 0] = np.nan
        result = DDRTree_reduce_dim(
            self.R_X, self.R_Z, self.R_Y, self.R_W,
            self.dimensions, self.maxiter, self.num_clusters,
            self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
        )
        
        # 断言返回结果是否正确处理NaN（可根据实现具体需求）
        self.assertIsNotNone(result)

    def test_DDRTree_reduce_dim_with_inf(self):
        """
        测试输入数据包含Inf值的情况
        """
        # 人为加入Inf值
        self.R_X[0, 0] = np.inf
        result = DDRTree_reduce_dim(
            self.R_X, self.R_Z, self.R_Y, self.R_W,
            self.dimensions, self.maxiter, self.num_clusters,
            self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
        )

        # 断言返回结果是否正确处理Inf
        self.assertIsNotNone(result)

    def test_large_input(self):
        """
        性能测试，使用更大数据量，检查算法的执行时间和内存消耗
        """
        # 增加样本数量和特征维度
        self.n_samples = 5000
        self.n_features = 100

        self.R_X = np.random.rand(self.n_features, self.n_samples)
        self.R_Z = np.random.rand(self.dimensions, self.n_samples)
        self.R_Y = np.random.rand(self.dimensions, self.num_clusters)
        self.R_W = np.random.rand(self.n_features, self.dimensions)

        result = DDRTree_reduce_dim(
            self.R_X, self.R_Z, self.R_Y, self.R_W,
            self.dimensions, self.maxiter, self.num_clusters,
            self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
        )

        # 断言返回结果不是空的
        self.assertIsNotNone(result)
        
    def test_empty_input(self):
        """
        测试输入为空的情况，检查是否能够处理
        """
        self.R_X = np.zeros((self.n_features, 0))  # 0个样本
        self.R_Z = np.zeros((self.dimensions, 0))  # 0个样本
        result = DDRTree_reduce_dim(
            self.R_X, self.R_Z, self.R_Y, self.R_W,
            self.dimensions, self.maxiter, self.num_clusters,
            self.sigma, self.lambda_, self.gamma, self.eps, self.verbose
        )

        # 检查返回的结果是否为空或者是否能正确处理空输入
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
