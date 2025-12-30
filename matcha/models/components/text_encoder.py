"""
Module d'encodage de texte

Implémentation de l'encodeur de texte basé sur Transformer avec prédiction de durée.
Version de reproduction: restructurée avec amélioration de la structure et des noms de variables.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from matcha.utils.model import sequence_mask


# Version hybride: Utilise PyTorch LayerNorm au lieu de la version personnalisée
# Mais garde les autres améliorations (prenet résiduel, Conv1d attention, Xavier init)


class ConvReluNorm(nn.Module):
    """Bloc de convolution avec ReLU et normalisation, avec connexion résiduelle"""

    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, num_layers, dropout_rate):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.convolutions = torch.nn.ModuleList()
        self.normalizations = torch.nn.ModuleList()

        # Première couche
        padding = kernel_size // 2
        self.convolutions.append(torch.nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=padding))
        # Version hybride: Utilise PyTorch LayerNorm (plus rapide)
        self.normalizations.append(nn.LayerNorm(hidden_channels))

        # Activation et dropout
        self.activation_dropout = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(dropout_rate))

        # Couches supplémentaires
        for _ in range(num_layers - 1):
            self.convolutions.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding)
            )
            # Version hybride: Utilise PyTorch LayerNorm
            self.normalizations.append(nn.LayerNorm(hidden_channels))

        # Projection finale avec initialisation à zéro
        self.projection = torch.nn.Conv1d(hidden_channels, output_channels, 1)
        self.projection.weight.data.zero_()
        self.projection.bias.data.zero_()

    def forward(self, x, x_mask):
        residual = x
        for i in range(self.num_layers):
            x = self.convolutions[i](x * x_mask)
            # Version hybride: PyTorch LayerNorm nécessite [B, T, C], donc transpose
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            x = self.normalizations[i](x)
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            x = self.activation_dropout(x)
        x = residual + self.projection(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    """Prédicteur de durée pour chaque token"""

    def __init__(self, input_channels, filter_channels, kernel_size, dropout_rate):
        super().__init__()
        self.input_channels = input_channels
        self.filter_channels = filter_channels
        self.dropout_rate = dropout_rate

        self.dropout = torch.nn.Dropout(dropout_rate)
        padding = kernel_size // 2

        self.conv_layer_1 = torch.nn.Conv1d(input_channels, filter_channels, kernel_size, padding=padding)
        # Version hybride: Utilise PyTorch LayerNorm
        self.norm_layer_1 = nn.LayerNorm(filter_channels)

        self.conv_layer_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=padding)
        # Version hybride: Utilise PyTorch LayerNorm
        self.norm_layer_2 = nn.LayerNorm(filter_channels)

        self.output_projection = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_layer_1(x * x_mask)
        x = torch.relu(x)
        # Version hybride: PyTorch LayerNorm nécessite [B, T, C], donc transpose
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.norm_layer_1(x)
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        x = self.dropout(x)

        x = self.conv_layer_2(x * x_mask)
        x = torch.relu(x)
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.norm_layer_2(x)
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        x = self.dropout(x)

        x = self.output_projection(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """Encodage positionnel rotatif (RoPE)"""

    def __init__(self, feature_dim, base_freq=10_000):
        super().__init__()
        self.base_freq = base_freq
        self.feature_dim = int(feature_dim)
        self.cos_cache = None
        self.sin_cache = None

    def _build_cache(self, x: torch.Tensor):
        """Construit le cache des valeurs cos et sin"""
        if self.cos_cache is not None and x.shape[0] <= self.cos_cache.shape[0]:
            return

        seq_length = x.shape[0]
        half_dim = self.feature_dim // 2

        # Calcul des fréquences theta
        theta = 1.0 / (self.base_freq ** (torch.arange(0, self.feature_dim, 2).float() / self.feature_dim)).to(x.device)

        # Index de position
        position_indices = torch.arange(seq_length, device=x.device).float().to(x.device)

        # Produit position * theta
        position_theta = torch.einsum("n,d->nd", position_indices, theta)

        # Concaténation pour duplication
        position_theta_duplicated = torch.cat([position_theta, position_theta], dim=1)

        # Cache
        self.cos_cache = position_theta_duplicated.cos()[:, None, None, :]
        self.sin_cache = position_theta_duplicated.sin()[:, None, None, :]

    def _apply_neg_half_transform(self, x: torch.Tensor):
        """Applique la transformation [-x[d/2:], x[:d/2]]"""
        half_dim = self.feature_dim // 2
        return torch.cat([-x[:, :, :, half_dim:], x[:, :, :, :half_dim]], dim=-1)

    def forward(self, x: torch.Tensor):
        """Applique l'encodage positionnel rotatif"""
        x = rearrange(x, "b h t d -> t b h d")
        self._build_cache(x)

        x_rope, x_pass = x[..., : self.feature_dim], x[..., self.feature_dim:]
        neg_half_x = self._apply_neg_half_transform(x_rope)

        x_rope = (x_rope * self.cos_cache[: x.shape[0]]) + (neg_half_x * self.sin_cache[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    """Attention multi-têtes avec encodage positionnel rotatif"""

    def __init__(
            self,
            channels,
            output_channels,
            num_heads,
            heads_share=True,
            dropout_rate=0.0,
            proximal_bias=False,
            proximal_init=False,
    ):
        super().__init__()
        assert channels % num_heads == 0

        self.channels = channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.dropout_rate = dropout_rate
        self.attention_weights = None

        self.head_dim = channels // num_heads

        # Version originale: Utilise Conv1d (comme ancienne version qui fonctionnait)
        # Conv1d travaille directement avec [B, C, T], pas besoin de transpose
        self.query_conv = torch.nn.Conv1d(channels, channels, 1)
        self.key_conv = torch.nn.Conv1d(channels, channels, 1)
        self.value_conv = torch.nn.Conv1d(channels, channels, 1)

        # Encodage positionnel rotatif
        self.query_rope = RotaryPositionalEmbeddings(self.head_dim * 0.5)
        self.key_rope = RotaryPositionalEmbeddings(self.head_dim * 0.5)

        self.output_conv = torch.nn.Conv1d(channels, output_channels, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Initialisation Xavier (gardé de l'ancienne version)
        torch.nn.init.xavier_uniform_(self.query_conv.weight)
        torch.nn.init.xavier_uniform_(self.key_conv.weight)
        if proximal_init:
            self.key_conv.weight.data.copy_(self.query_conv.weight.data)
            self.key_conv.bias.data.copy_(self.query_conv.bias.data)
        torch.nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, x, context, attention_mask=None):
        # Version originale: Conv1d travaille directement avec [B, C, T]
        # x et context sont déjà en format [B, C, T]
        query = self.query_conv(x)  # [B, C, T]
        key = self.key_conv(context)  # [B, C, T]
        value = self.value_conv(context)  # [B, C, T]

        output, self.attention_weights = self._compute_attention(query, key, value, mask=attention_mask)

        # output est déjà [B, C, T], pas besoin de transpose
        output = self.output_conv(output)  # [B, C, T]
        return output

    def _compute_attention(self, query, key, value, mask=None):
        """Calcule l'attention multi-têtes"""
        batch_size, channels, key_len, query_len = (*key.size(), query.size(2))

        query = rearrange(query, "b (h c) t-> b h t c", h=self.num_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.num_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.num_heads)

        query = self.query_rope(query)
        key = self.key_rope(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.proximal_bias:
            assert key_len == query_len, "Le biais proximal n'est disponible que pour l'auto-attention."
            scores = scores + self._get_proximal_bias(key_len).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attention_probs = torch.nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(2, 3).contiguous().view(batch_size, channels, query_len)
        return output, attention_probs

    @staticmethod
    def _get_proximal_bias(length):
        """Calcule le biais proximal pour l'auto-attention"""
        positions = torch.arange(length, dtype=torch.float32)
        position_diff = torch.unsqueeze(positions, 0) - torch.unsqueeze(positions, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(position_diff)), 0), 0)


# Version hybride: Utilise Sequential FFN (comme nouvelle version) au lieu de classe indépendante
# Mais garde le masquage explicite
class FFN(nn.Module):
    """Réseau feed-forward avec convolutions - Version hybride (Sequential mais avec masquage)"""

    def __init__(self, input_channels, output_channels, filter_channels, kernel_size, dropout_rate=0.0):
        super().__init__()
        padding = kernel_size // 2
        # Version hybride: Utilise Sequential (plus simple, comme nouvelle version)
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, filter_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(filter_channels, output_channels, kernel_size, padding=padding),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, x_mask):
        # Version hybride: Garde le masquage explicite (comme ancienne version)
        x = self.conv_net(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """Encodeur Transformer avec attention multi-têtes et FFN"""

    def __init__(
            self,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size=1,
            dropout_rate=0.0,
            **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.attention_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.attention_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, num_heads, dropout_rate=dropout_rate)
            )
            # Version hybride: Utilise PyTorch LayerNorm (plus rapide)
            self.norm_layers_1.append(nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    dropout_rate=dropout_rate,
                )
            )
            # Version hybride: Utilise PyTorch LayerNorm
            self.norm_layers_2.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attention_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        for i in range(self.num_layers):
            x = x * x_mask
            attn_output = self.attention_layers[i](x, x, attention_mask)
            attn_output = self.dropout(attn_output)
            # Version hybride: PyTorch LayerNorm nécessite [B, T, C], donc transpose
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            attn_output = attn_output.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            x = self.norm_layers_1[i](x + attn_output)
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]

            ffn_output = self.ffn_layers[i](x, x_mask)
            ffn_output = self.dropout(ffn_output)
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            ffn_output = ffn_output.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            x = self.norm_layers_2[i](x + ffn_output)
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    """Encodeur de texte avec prédiction de durée"""

    def __init__(
            self,
            encoder_type,
            encoder_params,
            duration_predictor_params,
            n_vocab,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.vocab_size = n_vocab
        self.feature_dim = encoder_params.n_feats
        self.channel_dim = encoder_params.n_channels

        # Embedding de vocabulaire
        self.embedding = torch.nn.Embedding(n_vocab, self.channel_dim)
        torch.nn.init.normal_(self.embedding.weight, 0.0, self.channel_dim ** -0.5)

        # Pre-net conditionnel
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.channel_dim,
                self.channel_dim,
                self.channel_dim,
                kernel_size=5,
                num_layers=3,
                dropout_rate=0.1,  # Version originale: dropout 0.1
            )
        else:
            self.prenet = lambda x, x_mask: x

        # Calcul des canaux d'entrée de l'encodeur (单演讲者，不需要 speaker embedding)
        encoder_input_channels = self.channel_dim

        self.encoder = Encoder(
            encoder_input_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # Projection vers l'espace acoustique
        self.mean_projection = torch.nn.Conv1d(encoder_input_channels, self.feature_dim, 1)

        # Prédicteur de durée
        self.duration_predictor = DurationPredictor(
            encoder_input_channels,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, text_input, text_lengths):
        """
        Passe avant de l'encodeur de texte

        Args:
            text_input: entrée de texte, shape (batch_size, max_text_length)
            text_lengths: longueurs des séquences, shape (batch_size,)

        Returns:
            mean_output: sortie moyenne de l'encodeur, shape (batch_size, n_feats, max_text_length)
            log_duration: log durée prédite, shape (batch_size, 1, max_text_length)
            text_mask: masque pour l'entrée, shape (batch_size, 1, max_text_length)
        """
        # Embedding et masque
        embedded = self.embedding(text_input) * math.sqrt(self.channel_dim)
        embedded = torch.transpose(embedded, 1, -1)
        text_mask = torch.unsqueeze(sequence_mask(text_lengths, embedded.size(2)), 1).to(embedded.dtype)

        # Pre-net
        encoded = self.prenet(embedded, text_mask)

        # Encodage
        encoded = self.encoder(encoded, text_mask)

        # Projection vers l'espace acoustique
        mean_output = self.mean_projection(encoded) * text_mask

        # Prédiction de durée (avec détachement du gradient)
        encoded_detached = torch.detach(encoded)
        log_duration = self.duration_predictor(encoded_detached, text_mask)

        return mean_output, log_duration, text_mask
