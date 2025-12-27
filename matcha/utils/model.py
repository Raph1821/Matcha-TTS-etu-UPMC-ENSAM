"""
Utilitaires pour les modèles

Fonctions utilitaires pour le traitement des séquences, l'alignement et la normalisation
des données dans le pipeline Matcha-TTS.
"""

import numpy as np
import torch
import torch.nn.functional as F


def create_sequence_mask(sequence_lengths, max_sequence_length=None):
    """
    Crée un masque booléen pour les séquences de longueurs variables.
    
    Args:
        sequence_lengths: longueurs réelles des séquences, shape (batch_size,)
        max_sequence_length: longueur maximale (si None, utilise le maximum de sequence_lengths)
        
    Returns:
        masque booléen, shape (batch_size, max_sequence_length)
        True pour les positions valides, False pour le padding
    """
    if max_sequence_length is None:
        max_sequence_length = sequence_lengths.max().item()
    
    # Créer un tenseur d'indices [0, 1, 2, ..., max_length-1]
    position_indices = torch.arange(
        max_sequence_length,
        dtype=sequence_lengths.dtype,
        device=sequence_lengths.device
    )
    
    # Comparer chaque position avec la longueur réelle de chaque séquence
    # shape: (batch_size, max_sequence_length)
    return position_indices.unsqueeze(0) < sequence_lengths.unsqueeze(1)


def adjust_length_for_downsampling(sequence_length, num_downsampling_layers=2):
    """
    Ajuste la longueur d'une séquence pour être compatible avec les couches de sous-échantillonnage.
    
    La longueur doit être divisible par 2^num_downsampling_layers pour éviter les problèmes
    de dimension lors du sous-échantillonnage dans l'architecture U-Net.
    
    Args:
        sequence_length: longueur de la séquence à ajuster
        num_downsampling_layers: nombre de couches de sous-échantillonnage
        
    Returns:
        longueur ajustée (entier)
    """
    downsampling_factor = 2 ** num_downsampling_layers
    
    # Arrondir à la valeur supérieure divisible par le facteur
    adjusted_length = torch.ceil(sequence_length / downsampling_factor) * downsampling_factor
    
    # Convertir en entier (sauf en mode ONNX export)
    if not torch.onnx.is_in_onnx_export():
        return int(adjusted_length.item())
    else:
        return adjusted_length


def _reverse_padding_shape(padding_config):
    """
    Inverse l'ordre des dimensions dans la configuration de padding.
    
    Utilisé pour convertir la configuration de padding de format (H, W) à (W, H).
    
    Args:
        padding_config: liste de listes représentant la configuration de padding
        
    Returns:
        configuration de padding inversée, aplatie en une seule liste
    """
    reversed_config = padding_config[::-1]
    flattened = [item for sublist in reversed_config for item in sublist]
    return flattened


def build_alignment_path(duration_tensor, attention_mask):
    """
    Construit une carte d'alignement monotone à partir des durées prédites.
    
    Cette fonction génère un chemin d'alignement entre les tokens de texte et les frames
    de spectrogramme mel en utilisant les durées cumulatives.
    
    Args:
        duration_tensor: durées prédites pour chaque token, shape (batch_size, num_tokens)
        attention_mask: masque d'attention, shape (batch_size, num_tokens, num_frames)
        
    Returns:
        carte d'alignement, shape (batch_size, num_tokens, num_frames)
    """
    device = duration_tensor.device
    batch_size, num_tokens, num_frames = attention_mask.shape
    
    # Calculer les durées cumulatives pour chaque token
    cumulative_durations = torch.cumsum(duration_tensor, dim=1)
    
    # Initialiser la carte d'alignement
    alignment_path = torch.zeros(
        batch_size, num_tokens, num_frames,
        dtype=attention_mask.dtype,
        device=device
    )
    
    # Aplatir les durées cumulatives pour traitement par batch
    cumulative_flat = cumulative_durations.view(batch_size * num_tokens)
    
    # Créer un masque basé sur les durées cumulatives
    cumulative_mask = create_sequence_mask(cumulative_flat, num_frames)
    cumulative_mask = cumulative_mask.to(attention_mask.dtype)
    
    # Remodeler en (batch_size, num_tokens, num_frames)
    alignment_path = cumulative_mask.view(batch_size, num_tokens, num_frames)
    
    # Calculer les différences pour obtenir les frontières d'alignement
    # Padding pour permettre la soustraction avec la position précédente
    padding_config = [[0, 0], [1, 0], [0, 0]]
    padded_path = F.pad(alignment_path, _reverse_padding_shape(padding_config))
    
    # Soustraire la position précédente pour obtenir les frontières
    alignment_path = alignment_path - padded_path[:, :-1, :]
    
    # Appliquer le masque d'attention
    alignment_path = alignment_path * attention_mask
    
    return alignment_path


def compute_duration_loss(predicted_log_durations, target_log_durations, sequence_lengths):
    """
    Calcule la perte MSE entre les durées prédites et les durées cibles.
    
    Args:
        predicted_log_durations: durées prédites en échelle logarithmique,
            shape (batch_size, 1, num_tokens)
        target_log_durations: durées cibles en échelle logarithmique,
            shape (batch_size, 1, num_tokens)
        sequence_lengths: longueurs réelles des séquences, shape (batch_size,)
        
    Returns:
        perte de durée normalisée (scalaire)
    """
    # Calculer la différence au carré
    squared_diff = (predicted_log_durations - target_log_durations) ** 2
    
    # Somme des différences au carré, normalisée par la somme des longueurs
    total_squared_diff = torch.sum(squared_diff)
    total_length = torch.sum(sequence_lengths)
    
    return total_squared_diff / total_length


def apply_normalization(data_tensor, mean_value, std_value):
    """
    Normalise les données en soustrayant la moyenne et en divisant par l'écart-type.
    
    Args:
        data_tensor: données à normaliser, shape (..., feature_dim, sequence_length)
        mean_value: valeur moyenne (peut être float, list, numpy array ou torch.Tensor)
        std_value: écart-type (peut être float, list, numpy array ou torch.Tensor)
        
    Returns:
        données normalisées, même shape que data_tensor
    """
    # Convertir mean_value en tenseur si nécessaire
    if not isinstance(mean_value, (float, int)):
        if isinstance(mean_value, list):
            mean_tensor = torch.tensor(mean_value, dtype=data_tensor.dtype, device=data_tensor.device)
        elif isinstance(mean_value, np.ndarray):
            mean_tensor = torch.from_numpy(mean_value).to(data_tensor.device)
        elif isinstance(mean_value, torch.Tensor):
            mean_tensor = mean_value.to(data_tensor.device)
        else:
            mean_tensor = mean_value
        mean_tensor = mean_tensor.unsqueeze(-1)
    else:
        mean_tensor = mean_value

    # Convertir std_value en tenseur si nécessaire
    if not isinstance(std_value, (float, int)):
        if isinstance(std_value, list):
            std_tensor = torch.tensor(std_value, dtype=data_tensor.dtype, device=data_tensor.device)
        elif isinstance(std_value, np.ndarray):
            std_tensor = torch.from_numpy(std_value).to(data_tensor.device)
        elif isinstance(std_value, torch.Tensor):
            std_tensor = std_value.to(data_tensor.device)
        else:
            std_tensor = std_value
        std_tensor = std_tensor.unsqueeze(-1)
    else:
        std_tensor = std_value

    # Appliquer la normalisation: (x - mean) / std
    normalized = (data_tensor - mean_tensor) / std_tensor
    
    return normalized


def apply_denormalization(data_tensor, mean_value, std_value):
    """
    Dénormalise les données en multipliant par l'écart-type et en ajoutant la moyenne.
    
    Opération inverse de apply_normalization: x * std + mean
    
    Args:
        data_tensor: données normalisées, shape (..., feature_dim, sequence_length)
        mean_value: valeur moyenne (peut être float, list, numpy array ou torch.Tensor)
        std_value: écart-type (peut être float, list, numpy array ou torch.Tensor)
        
    Returns:
        données dénormalisées, même shape que data_tensor
    """
    # Convertir mean_value en tenseur si nécessaire
    if isinstance(mean_value, (float, int)):
        mean_tensor = mean_value
    else:
        if isinstance(mean_value, list):
            mean_tensor = torch.tensor(mean_value, dtype=data_tensor.dtype, device=data_tensor.device)
        elif isinstance(mean_value, np.ndarray):
            mean_tensor = torch.from_numpy(mean_value).to(data_tensor.device)
        elif isinstance(mean_value, torch.Tensor):
            mean_tensor = mean_value.to(data_tensor.device)
        else:
            mean_tensor = mean_value
        mean_tensor = mean_tensor.unsqueeze(-1)

    # Convertir std_value en tenseur si nécessaire
    if isinstance(std_value, (float, int)):
        std_tensor = std_value
    else:
        if isinstance(std_value, list):
            std_tensor = torch.tensor(std_value, dtype=data_tensor.dtype, device=data_tensor.device)
        elif isinstance(std_value, np.ndarray):
            std_tensor = torch.from_numpy(std_value).to(data_tensor.device)
        elif isinstance(std_value, torch.Tensor):
            std_tensor = std_value.to(data_tensor.device)
        else:
            std_tensor = std_value
        std_tensor = std_tensor.unsqueeze(-1)

    # Appliquer la dénormalisation: x * std + mean
    denormalized = data_tensor * std_tensor + mean_tensor
    
    return denormalized


# Alias pour compatibilité avec le code existant
sequence_mask = create_sequence_mask
fix_len_compatibility = adjust_length_for_downsampling
generate_path = build_alignment_path
duration_loss = compute_duration_loss
normalize = apply_normalization
denormalize = apply_denormalization
