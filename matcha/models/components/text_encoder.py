import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Pour les réarrangements de tenseurs




class DurationPredictor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        filter_channels: int,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        padding: str = "same",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Couches
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=filter_channels,
            kernel_size=kernel_size,
            padding=self._get_padding(padding, kernel_size),
        )
        self.norm1 = nn.LayerNorm(filter_channels)
        self.dropout1 = nn.Dropout(p_dropout)

        self.conv2 = nn.Conv1d(
            in_channels=filter_channels,
            out_channels=filter_channels,
            kernel_size=kernel_size,
            padding=self._get_padding(padding, kernel_size),
        )
        self.norm2 = nn.LayerNorm(filter_channels)
        self.dropout2 = nn.Dropout(p_dropout)

        # Projection finale
        self.proj = nn.Conv1d(
            in_channels=filter_channels,
            out_channels=1,
            kernel_size=1,
        )

    def _get_padding(self, padding_type: str, kernel_size: int) -> int:
        if padding_type == "same":
            return (kernel_size - 1) // 2
        elif padding_type == "valid":
            return 0
        else:
            raise ValueError("padding doit être 'same' ou 'valid'.")

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        if x_mask is not None:
            # x_mask est de forme [batch, 1, seq_len]
            # Étend le masque à [batch, hidden_channels, seq_len] pour correspondre à x
            x_mask_conv = x_mask.expand(-1, self.input_channels, -1)  # [batch, hidden_channels, seq_len]
            x = x * x_mask_conv  # Masque l'entrée

        # Conv1D_1 + ReLU + LayerNorm + Dropout
        x = self.conv1(x)
        x = torch.relu(x)
        if x_mask is not None:
            x = x * x_mask_conv  # Masque après ReLU

        x = x.transpose(1, 2)  # [B, T, filter_channels] pour LayerNorm
        x = self.norm1(x)
        x = x.transpose(1, 2)  # Retour à [B, filter_channels, T]
        x = self.dropout1(x)

        # Conv1D_2 + ReLU + LayerNorm + Dropout
        x = self.conv2(x)
        x = torch.relu(x)
        if x_mask is not None:
            # Met à jour x_mask_conv pour filter_channels
            x_mask_conv = x_mask.expand(-1, self.filter_channels, -1)
            x = x * x_mask_conv  # Masque après ReLU

        x = x.transpose(1, 2)  # [B, T, filter_channels] pour LayerNorm
        x = self.norm2(x)
        x = x.transpose(1, 2)  # Retour à [B, filter_channels, T]
        x = self.dropout2(x)

        # Projection vers [B, 1, T]
        x = self.proj(x)
        if x_mask is not None:
            x = x * x_mask  # x_mask est déjà de forme [batch, 1, seq_len]

        return x





class RotaryPositionalEmbeddings(nn.Module):
    """
    Implémentation des Rotary Positional Embeddings (RoPE).
    Applique une rotation aux paires de features pour encoder la position dans une séquence.
    """
    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        self.base = base  # Base pour le calcul des fréquences (θ_i = 1/base^(2i/d))
        self.d = d        # Dimension des features à rotater (doit être pair)
        self.cos_cached = None  # Cache pour les valeurs cosinus
        self.sin_cached = None  # Cache pour les valeurs sinus

    def _build_cache(self, x: torch.Tensor):
        """Calcule et met en cache les valeurs cos/sin pour les positions."""
        seq_len = x.shape[-2]  # Longueur de la séquence
        # θ_i = 1 / (base^(2i/d)) pour i = 0, 2, 4, ..., d-2
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        # Indices de position [0, 1, 2, ..., seq_len-1]
        seq_idx = torch.arange(seq_len, device=x.device).float()
        # Produit des indices de position par θ_i
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta)
        # Concatène pour obtenir [θ₀, θ₁, ..., θ₀, θ₁, ...]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # Met en cache cos et sin (forme: 1 x 1 x seq_len x d)
        self.cos_cached = idx_theta2.cos()[None, None, :, :]
        self.sin_cached = idx_theta2.sin()[None, None, :, :]

    def _neg_half(self, x: torch.Tensor):
        """Sépare et négative la deuxième moitié des features pour la rotation."""
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Applique RoPE à x.
        Args:
            x: Tenseur d'entrée de forme [batch, heads, seq_len, d] où d est pair.
        Returns:
            Tenseur avec RoPE appliqué, même forme que x.
        """
        seq_len = x.shape[-2]
        self._build_cache(x)  # Met à jour le cache si nécessaire

        # Sépare les features en deux parties:
        # - x_rope: les d premières dimensions (seront rotées)
        # - x_pass: les dimensions restantes (ne seront pas modifiées)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # Applique la rotation: x_rope * cos + neg_half(x_rope) * sin
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:, :, :seq_len]) + (neg_half_x * self.sin_cached[:, :, :seq_len])

        # Recombine les features rotées et non rotées
        return torch.cat((x_rope, x_pass), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_dim: int = None,
        base: int = 10000,
    ):
        """
        Args:
            embed_dim: Dimension totale des embeddings.
            num_heads: Nombre de têtes d'attention.
            dropout: Taux de dropout.
            rotary_dim: Dimension des features à rotater avec RoPE (doit être pair).
                        Si None, utilise embed_dim // num_heads.
            base: Base pour le calcul des fréquences de RoPE.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        assert self.rotary_dim % 2 == 0, "rotary_dim doit être pair pour RoPE."

        # Couches de projection Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # RoPE pour queries et keys
        self.query_rotary_pe = RotaryPositionalEmbeddings(d=self.rotary_dim, base=base)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d=self.rotary_dim, base=base)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        """
        Args:
            x: Tenseur d'entrée de forme [batch, seq_len, embed_dim].
            attn_mask: Masque d'attention de forme [batch, seq_len] ou [batch, 1, seq_len].
            need_weights: Si True, retourne aussi les poids d'attention.
        Returns:
            Tenseur de sortie de forme [batch, seq_len, embed_dim].
            Optionnellement, les poids d'attention si need_weights=True.
        """
        batch_size, seq_len, _ = x.shape

        # Projection Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch, seq_len, embed_dim]

        # Reshape pour les têtes: [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Applique RoPE aux queries et keys
        q = self.query_rotary_pe(q)
        k = self.key_rotary_pe(k)

        # Calcul des scores d'attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, seq_len, seq_len]

        # Applique le masque (si fourni)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len] ou [batch, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Pondération des values
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape pour la sortie
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Projection de sortie
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        else:
            return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int,
        p_dropout: float,
        rotary_dim: int = None,
        base: int = 10000,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Remplace nn.MultiheadAttention par MultiHeadAttentionWithRoPE
        self.attn = MultiHeadAttentionWithRoPE(
            embed_dim=hidden_channels,
            num_heads=n_heads,
            dropout=p_dropout,
            rotary_dim=rotary_dim,
            base=base,
        )

        # Réseau convolutif 
        self.conv_net = nn.Sequential(
            nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.Dropout(p_dropout)
        )

        # Normalisation 
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tenseur d'entrée de forme [batch, hidden_channels, seq_len].
            mask: Masque pour ignorer le padding. Forme [batch, seq_len].
        Returns:
            Tenseur de sortie de forme [batch, hidden_channels, seq_len].
        """
        x_residual = x  # [batch, hidden_channels, seq_len]

        # --- Attention multi-têtes avec RoPE ---
        # Transpose pour [batch, seq_len, hidden_channels] (format attendu par MultiHeadAttentionWithRoPE)
        x_transpose = x.transpose(1, 2)

        # Applique l'attention avec RoPE
        attn_out = self.attn(x_transpose, attn_mask=mask)

        # Connection résiduelle + normalisation
        x = self.norm1(x_transpose + attn_out)
        x = x.transpose(1, 2)  # Retour à [batch, hidden_channels, seq_len]

        # --- Réseau convolutif ---
        conv_out = self.conv_net(x)
        x = x + conv_out  # Connection résiduelle

        # Normalisation
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_channels]
        x = self.norm2(x)
        x = x.transpose(1, 2)  # Retour à [batch, hidden_channels, seq_len]

        return x



class TextEncoder(nn.Module):
    """
    Encodeur de texte pour la TTS avec RoPE.
    """
    def __init__(
        self,
        n_vocab,          # Taille du vocabulaire (ex: 148 pour LJSpeech)
        out_channels,     # Dimension de sortie (ex: 80 pour Mel-spectrogramme)
        hidden_channels,  # Dimension interne (ex: 256)
        filter_channels,  # Dimension intermédiaire pour les convolutions (ex: 512)
        n_heads,          # Nombre de têtes d'attention (ex: 4)
        n_layers,         # Nombre de blocs EncoderBlock (ex: 6)
        kernel_size,      # Taille du noyau des convolutions (ex: 5)
        p_dropout,         # Taux de dropout (ex: 0.1)
        # Ajoute les paramètres pour DurationPredictor
        duration_filter_channels: int = 256,
        duration_kernel_size: int = 3,
        duration_p_dropout: float = 0.1
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels

        # Couche d'embedding pour le texte
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # Pile de blocs encodeurs avec RoPE
        self.encoder_stack = nn.ModuleList([
            EncoderBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

        # Projection finale vers la dimension acoustique
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

        # Ajoute le DurationPredictor
        self.duration_predictor = DurationPredictor(
            input_channels=hidden_channels,  # Prend les embeddings du texte
            filter_channels=duration_filter_channels,
            kernel_size=duration_kernel_size,
            p_dropout=duration_p_dropout,
        )
        

    def forward(self, x, x_lengths):
        # Masque pour ignorer le padding
        max_len = x.size(1)
        # Crée un masque de forme [batch, seq_len]
        mask = torch.arange(max_len, device=x.device)[None, :] >= x_lengths[:, None]
        # Ajoute une dimension pour obtenir [batch, 1, seq_len]
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len]

        # Embedding des IDs de texte + scaling
        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)

        # Transpose pour [batch, hidden_channels, max_text_len]
        x = x.transpose(1, 2)

        # Passe à travers chaque bloc encodeur
        for block in self.encoder_stack:
            x = block(x, mask=mask.squeeze(1))  # mask de forme [batch, seq_len] pour EncoderBlock

        # Projection vers la sortie acoustique
        mu = self.proj(x)

        # Prédit les durées à partir des embeddings finaux
        log_durations = self.duration_predictor(x, x_mask=mask)  # mask est [batch, 1, seq_len]

        return mu, log_durations, mask

    



'''
TEST CODE 
'''

if __name__ == "__main__":

#MULTIHEADATTENTION


    # Hyperparamètres
    embed_dim = 256
    num_heads = 4
    rotary_dim = 64  # Doit être <= head_dim (ici, head_dim = 256/4 = 64)
    batch_size = 2
    seq_len = 10

    # Initialisation
    attention = MultiHeadAttentionWithRoPE(embed_dim, num_heads, rotary_dim=rotary_dim)

    # Entrée factice
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = attention(x)
    print("Forme de la sortie:", output.shape)  # torch.Size([2, 10, 256])






#PREDICTION DE DUREE


    # Hyperparamètres
    input_channels = 256
    filter_channels = 256
    kernel_size = 3
    p_dropout = 0.1
    batch_size = 2
    seq_len = 10

    # Initialisation
    duration_predictor = DurationPredictor(
        input_channels, filter_channels, kernel_size, p_dropout
    )

    # Entrée factice : [B, input_channels, T]
    x = torch.randn(batch_size, input_channels, seq_len)
    x_mask = torch.ones(batch_size, 1, seq_len).bool()  # 1 = valide, 0 = padding
    x_mask[:, :, 5:] = 0  # Simule du padding à partir de l'indice 5

    # Forward pass
    durations = duration_predictor(x, x_mask=x_mask)
    print("Forme des durées prédites:", durations.shape)  # torch.Size([2, 1, 10])
    print("Durées prédites (premier exemple):", durations[0, 0, :])


#TEXT ENCODER

    # Hyperparamètres
    n_vocab = 148
    out_channels = 80
    hidden_channels = 256
    filter_channels = 512
    n_heads = 4
    n_layers = 6
    kernel_size = 5
    p_dropout = 0.1

    # Initialisation
    text_encoder = TextEncoder(
        n_vocab, out_channels, hidden_channels, filter_channels,
        n_heads, n_layers, kernel_size, p_dropout,
        duration_filter_channels=256,
        duration_kernel_size=3,
        duration_p_dropout=0.1,
    )

    # Entrée factice
    x = torch.LongTensor([[12, 45, 67, 0, 0], [89, 23, 0, 0, 0]])
    x_lengths = torch.LongTensor([3, 2]) #durée de chaque exemple sans padding 

    # Forward pass
    mu, mask, durations = text_encoder(x, x_lengths)
    print("Forme de mu:", mu.shape)  # torch.Size([2, 80, 5])
    print("Forme des durées:", durations.shape)  # torch.Size([2, 1, 5])
    print("Durées prédites (premier exemple):", durations[0, 0, :3])  # Affiche les durées pour les 3 premiers phonèmes
