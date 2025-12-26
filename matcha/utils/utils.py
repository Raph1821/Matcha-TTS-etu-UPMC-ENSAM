import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_spectrogram(spectrogram, title="Mel Spectrogram"):
    """
    Affiche un spectrogramme de Mel.
    """
    # Si le spectrogramme a une dimension de batch [1, C, T], on la retire
    if len(spectrogram.shape) == 3:
        spectrogram = spectrogram.squeeze(0)
    
    # Conversion en numpy pour Matplotlib
    data = spectrogram.cpu().numpy() if isinstance(spectrogram, torch.Tensor) else spectrogram
    
    plt.figure(figsize=(12, 5))
    img = plt.imshow(data, aspect='auto', origin='lower', cmap='Purples')
    
    plt.colorbar(img, label='Log Intensity')
    plt.title(title)
    plt.xlabel('Time (Frames)')
    plt.ylabel('Mel Channels')
    plt.tight_layout()
    plt.show()


def save_figure_to_numpy(fig):
    """将matplotlib图形转换为numpy数组"""
    # 使用tostring_rgb()或buffer_rgba()取决于matplotlib版本
    try:
        # 旧版本matplotlib使用tostring_rgb()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        data = data.reshape(height, width, 3)
    except (AttributeError, TypeError):
        # 新版本matplotlib使用buffer_rgba()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(buf, dtype=np.uint8)
        # 转换为RGB（去掉alpha通道）
        data = data.reshape(height, width, 4)[:, :, :3]
    
    return data


def plot_tensor(tensor):
    """
    将张量转换为图像数组，用于日志记录
    
    Args:
        tensor: 输入张量，可以是torch.Tensor或numpy数组
                形状可以是 [C, T] 或 [B, C, T]
    
    Returns:
        numpy数组，形状为 [H, W, 3]，RGB格式
    """
    # 转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    # 处理批次维度
    if tensor_np.ndim == 3:
        tensor_np = tensor_np[0]  # 取第一个样本
    
    # 确保是2D数组 [C, T]
    if tensor_np.ndim != 2:
        raise ValueError(f"Expected 2D tensor [C, T], got shape {tensor_np.shape}")
    
    # 创建图形
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor_np, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    
    # 转换为numpy数组
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    
    return data