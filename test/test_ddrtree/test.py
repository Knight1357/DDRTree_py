import numpy as np
import ddr_tree


def inspect_input(name, arr):
    print(f"{name}:")
    print(f"  Type: {type(arr)}")
    if isinstance(arr, np.ndarray):
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  First few elements:\n{arr[:5, :5] if arr.ndim == 2 else arr[:5]}")
    else:
        print("  Not a numpy array")




# 模拟输入数据，确保类型和形状正确

X = np.random.rand(100, 50).astype(np.float64)  # arg0
Z = np.random.rand(2, 50).astype(np.float64)  # arg1
Y = np.random.rand(2, 10).astype(np.float64)  # arg2
W = np.random.rand(100, 2).astype(np.float64)  # arg3
dimensions=2  # arg4
maxiter=20  # arg5
num_clusters=10  # arg6
sigma=1e-3  # arg7
lambda_=5.0  # arg8
gamma=10.0  # arg9
eps=1e-3  # arg10
verbose=True  # arg11

# 打印输入数据的形状
# inspect_input("X", X)
# inspect_input("Z", Z)
# inspect_input("Y", Y)
# inspect_input("W", W)

# 使用 DDRTree 降维
result = ddr_tree.DDRTree_reduce_dim(
    X, Z, Y, W,
    dimensions=dimensions,
    maxiter=maxiter,
    num_clusters=num_clusters,
    sigma=sigma,
    lambda_=lambda_,
    gamma=gamma,
    eps=eps,
    verbose=verbose
)

# 查看结果
print("Projection matrix W:\n", result["W"])
print("Reduced dimensions Z:\n", result["Z"])

# 打印最小生成树的三元组形式
print("Spanning tree:\n", result["stree"])
