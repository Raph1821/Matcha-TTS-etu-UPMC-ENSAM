import numpy as np
import torch

try:
    from matcha.utils.monotonic_align.core import maximum_path_c
except ImportError:
    # Fallback to pure Python implementation if Cython extension is not available
    def maximum_path_c(paths, values, t_xs, t_ys, max_neg_val=-1e9):
        """Pure Python fallback implementation"""
        b = values.shape[0]
        for i in range(b):
            t_x, t_y = int(t_xs[i]), int(t_ys[i])
            value = values[i, :t_x, :t_y]
            path = paths[i, :t_x, :t_y]
            
            # Forward pass
            for y in range(t_y):
                for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                    if x == y:
                        v_cur = max_neg_val
                    else:
                        v_cur = value[x, y-1]
                    if x == 0:
                        if y == 0:
                            v_prev = 0.
                        else:
                            v_prev = max_neg_val
                    else:
                        v_prev = value[x-1, y-1]
                    value[x, y] = max(v_cur, v_prev) + value[x, y]
            
            # Backward pass
            index = t_x - 1
            for y in range(t_y - 1, -1, -1):
                path[index, y] = 1
                if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
                    index = index - 1


def maximum_path(value, mask):
    """Cython optimised version (with Python fallback).
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)

