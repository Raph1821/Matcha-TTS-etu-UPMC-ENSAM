"""
Utilitaires de visualisation pour les spectrogrammes et tenseurs
Version de reproduction: réimplémentation avec support amélioré des types
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_spectrogram(spectrogram, title="Mel Spectrogram"):
    """
    Affiche un spectrogramme de Mel.
    """
    if len(spectrogram.shape) == 3:
        spectrogram = spectrogram.squeeze(0)
    
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
    """
    Convertit une figure matplotlib en tableau numpy RGB.
    
    Supporte les anciennes et nouvelles versions de matplotlib.
    """
    try:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        data = data.reshape(height, width, 3)
    except (AttributeError, TypeError):
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(height, width, 4)[:, :, :3]
    
    return data


def plot_tensor(tensor):
    """
    Convertit un tenseur en tableau d'images RGB pour l'enregistrement des logs.
    
    Args:
        tensor: Tenseur d'entrée (torch.Tensor ou numpy.ndarray)
               Forme acceptée: [C, T] ou [B, C, T]
    
    Returns:
        Tableau numpy de forme [H, W, 3] en format RGB
    """
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    if tensor_np.ndim == 3:
        tensor_np = tensor_np[0]
    
    if tensor_np.ndim != 2:
        raise ValueError(f"Expected 2D tensor [C, T], got shape {tensor_np.shape}")
    
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor_np, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    
    return data