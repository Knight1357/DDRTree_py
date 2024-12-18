import time
from loguru import logger
import datetime
import scanpy as sc

def time_func(func):
    """
    便捷的记录函数运行时间
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        start_time_fmt = datetime.datetime.fromtimestamp(start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        run_time = str(round((end_time - start_time) / 60, 2))

        logger.info(
            f"函数 {func.__name__} 开始时间： {start_time_fmt}，"
            f"整体运行时间: {run_time} min"
        )
        return result

    return wrapper


def memory_usage(step="此处"):
    memory_usage = sc.logging.get_memory_usage()
    logger.info(
        f"代码运行到{step}内存共使用：{memory_usage[0]:.2f} GB，相比前一次记录，内存变化：{memory_usage[1]:.2f} GB"
    )
    