"""
Monotonic Alignment Search (MAS) - Cython 优化实现
复现版本：保持算法逻辑一致，但代码实现有调整
"""
import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(
    int[:,::1] path, 
    float[:,::1] value, 
    int t_x, 
    int t_y, 
    float max_neg_val
) nogil:
    """
    为单个样本计算单调对齐路径
    
    Args:
        path: 输出路径矩阵 [t_x, t_y]
        value: 输入值矩阵 [t_x, t_y]（会被修改）
        t_x: 文本序列长度
        t_y: 音频序列长度
        max_neg_val: 最大负值（用于边界处理）
    """
    cdef int x, y
    cdef float v_prev, v_cur
    cdef int index = t_x - 1

    # 前向传播：计算累积最大值
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            # 计算当前值和前一个值
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            
            # 更新累积值
            value[x, y] = max(v_cur, v_prev) + value[x, y]

    # 后向传播：回溯找到最优路径
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(
    int[:,:,::1] paths, 
    float[:,:,::1] values, 
    int[::1] t_xs, 
    int[::1] t_ys, 
    float max_neg_val=-1e9
) nogil:
    """
    批量计算单调对齐路径（并行处理）
    
    Args:
        paths: 输出路径矩阵 [batch, t_x, t_y]
        values: 输入值矩阵 [batch, t_x, t_y]（会被修改）
        t_xs: 每个样本的文本长度 [batch]
        t_ys: 每个样本的音频长度 [batch]
        max_neg_val: 最大负值（用于边界处理）
    """
    cdef int b = values.shape[0]
    cdef int i
    
    # 并行处理每个样本
    for i in prange(b, nogil=True):
        maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)

