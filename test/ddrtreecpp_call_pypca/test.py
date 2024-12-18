import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(current_dir, "DDRTree_py")
sys.path.append(so_path)

print("Python 模块搜索路径：", sys.path)

try:
    import DDRTree_cpp
    print("DDRTree_cpp 模块导入成功！")
except ImportError as e:
    print(f"导入 DDRTree_cpp 失败: {e}")
