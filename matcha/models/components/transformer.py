"""
Module Transformer pour le décodage

Implémentation des blocs Transformer avec support de différentes fonctions d'activation
et mécanismes d'attention pour la génération de spectrogrammes mel.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from diffusers.models.attention import (
        GEGLU,
        GELU,
        AdaLayerNorm,
        AdaLayerNormZero,
        ApproximateGELU,
    )
    from diffusers.models.attention_processor import Attention
    from diffusers.models.lora import LoRACompatibleLinear
    from diffusers.utils.torch_utils import maybe_allow_in_graph
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    # Implémentations de repli
    class LoRACompatibleLinear(nn.Linear):
        pass
    
    def maybe_allow_in_graph(cls):
        return cls


class SnakeBeta(nn.Module):
    """
    Fonction Snake modifiée qui utilise des paramètres séparés pour l'amplitude des composantes périodiques
    
    Forme:
        - Entrée: (B, C, T)
        - Sortie: (B, C, T), même forme que l'entrée
    
    Paramètres:
        - alpha: paramètre entraînable qui contrôle la fréquence
        - beta: paramètre entraînable qui contrôle l'amplitude
    
    Références:
        - Cette fonction d'activation est une version modifiée basée sur cet article de Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    
    Exemples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialisation.
        ENTRÉE:
            - in_features: forme de l'entrée
            - alpha: paramètre entraînable qui contrôle la fréquence
            - beta: paramètre entraînable qui contrôle l'amplitude
            alpha est initialisé à 1 par défaut, valeurs plus élevées = fréquence plus élevée.
            beta est initialisé à 1 par défaut, valeurs plus élevées = amplitude plus élevée.
            alpha sera entraîné avec le reste du modèle.
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # Initialiser alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Échelle logarithmique, initialisée à zéros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # Échelle linéaire, initialisée à uns
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Passe avant de la fonction.
        Applique la fonction à l'entrée élément par élément.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class FeedForward(nn.Module):
    """
    Couche feed-forward.
    
    Paramètres:
        dim (`int`): Le nombre de canaux dans l'entrée.
        dim_out (`int`, *optionnel*): Le nombre de canaux dans la sortie. Si non donné, par défaut à `dim`.
        mult (`int`, *optionnel*, par défaut 4): Le multiplicateur à utiliser pour la dimension cachée.
        dropout (`float`, *optionnel*, par défaut 0.0): La probabilité de dropout à utiliser.
        activation_fn (`str`, *optionnel*, par défaut `"geglu"`): Fonction d'activation à utiliser dans feed-forward.
        final_dropout (`bool` *optionnel*, par défaut False): Appliquer un dropout final.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Initialiser act_fn, s'assurer qu'il y a toujours une valeur
        act_fn = None
        
        if not DIFFUSERS_AVAILABLE:
            # Implémentations de repli
            if activation_fn == "gelu":
                act_fn = nn.GELU()
            elif activation_fn == "geglu":
                # Approximation simple de GEGLU
                class SimpleGEGLU(nn.Module):
                    def __init__(self, dim_in, dim_out):
                        super().__init__()
                        self.proj = nn.Linear(dim_in, dim_out * 2)
                    def forward(self, x):
                        x, gate = self.proj(x).chunk(2, dim=-1)
                        return x * nn.functional.gelu(gate)
                act_fn = SimpleGEGLU(dim, inner_dim)
            elif activation_fn == "snakebeta" or activation_fn == "snake":
                # "snake" et "snakebeta" utilisent tous deux SnakeBeta
                act_fn = SnakeBeta(dim, inner_dim)
            else:
                # Par défaut utiliser GELU
                act_fn = nn.GELU()
        else:
            if activation_fn == "gelu":
                act_fn = GELU(dim, inner_dim)
            elif activation_fn == "gelu-approximate":
                act_fn = GELU(dim, inner_dim, approximate="tanh")
            elif activation_fn == "geglu":
                act_fn = GEGLU(dim, inner_dim)
            elif activation_fn == "geglu-approximate":
                act_fn = ApproximateGELU(dim, inner_dim)
            elif activation_fn == "snakebeta" or activation_fn == "snake":
                # "snake" et "snakebeta" utilisent tous deux SnakeBeta
                act_fn = SnakeBeta(dim, inner_dim)
            else:
                # Par défaut utiliser GELU
                act_fn = GELU(dim, inner_dim)
        
        # S'assurer que act_fn a été assigné
        if act_fn is None:
            raise ValueError(f"Fonction d'activation inconnue: {activation_fn}")

        self.net = nn.ModuleList([])
        # Projection d'entrée
        self.net.append(act_fn)
        # Dropout de projection
        self.net.append(nn.Dropout(dropout))
        # Projection de sortie
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # FF comme utilisé dans Vision Transformer, MLP-Mixer, etc. ont un dropout final
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


if DIFFUSERS_AVAILABLE:
    @maybe_allow_in_graph
    class BasicTransformerBlock(nn.Module):
        """
        Un bloc Transformer de base.
        
        Paramètres:
            dim (`int`): Le nombre de canaux dans l'entrée et la sortie.
            num_attention_heads (`int`): Le nombre de têtes à utiliser pour l'attention multi-têtes.
            attention_head_dim (`int`): Le nombre de canaux dans chaque tête.
            dropout (`float`, *optionnel*, par défaut 0.0): La probabilité de dropout à utiliser.
            cross_attention_dim (`int`, *optionnel*): La taille du vecteur encoder_hidden_states pour l'attention croisée.
            only_cross_attention (`bool`, *optionnel*):
                S'il faut utiliser uniquement des couches d'attention croisée. Dans ce cas, deux couches d'attention croisée sont utilisées.
            double_self_attention (`bool`, *optionnel*):
                S'il faut utiliser deux couches d'auto-attention. Dans ce cas, aucune couche d'attention croisée n'est utilisée.
            activation_fn (`str`, *optionnel*, par défaut `"geglu"`): Fonction d'activation à utiliser dans feed-forward.
            num_embeds_ada_norm (:
                obj: `int`, *optionnel*): Le nombre de pas de diffusion utilisés pendant l'entraînement. Voir `Transformer2DModel`.
            attention_bias (:
                obj: `bool`, *optionnel*, par défaut `False`): Configurer si les attentions doivent contenir un paramètre de biais.
        """

        def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
        ):
            super().__init__()
            self.only_cross_attention = only_cross_attention

            self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
            self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

            if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
                raise ValueError(
                    f"`norm_type` est défini à {norm_type}, mais `num_embeds_ada_norm` n'est pas défini. Veuillez vous assurer de"
                    f" définir `num_embeds_ada_norm` si vous définissez `norm_type` à {norm_type}."
                )

            # Définir 3 blocs. Chaque bloc a sa propre couche de normalisation.
            # 1. Auto-Attention
            if self.use_ada_layer_norm:
                self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_zero:
                self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            else:
                self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )

            # 2. Attention croisée
            if cross_attention_dim is not None or double_self_attention:
                # Nous n'utilisons actuellement AdaLayerNormZero que pour l'auto-attention où il n'y aura qu'un seul bloc d'attention.
                # C'est-à-dire que le nombre de morceaux de modulation retournés depuis AdaLayerZero n'aurait pas de sens s'ils étaient retournés pendant
                # le deuxième bloc d'attention croisée.
                self.norm2 = (
                    AdaLayerNorm(dim, num_embeds_ada_norm)
                    if self.use_ada_layer_norm
                    else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
                )
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )  # est auto-attention si encoder_hidden_states est none
            else:
                self.norm2 = None
                self.attn2 = None

            # 3. Feed-forward
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

            # Laisser la taille de chunk par défaut à None
            self._chunk_size = None
            self._chunk_dim = 0

        def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
            """Définit le feed-forward par chunks"""
            self._chunk_size = chunk_size
            self._chunk_dim = dim

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            # Notez que la normalisation est toujours appliquée avant le calcul réel dans les blocs suivants.
            # 1. Auto-Attention
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            # 2. Attention croisée
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" peut être utilisé pour économiser la mémoire
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"Dimension de `hidden_states` à chunker: {norm_hidden_states.shape[self._chunk_dim]} doit être divisible par la taille de chunk: {self._chunk_size}. Assurez-vous de définir une `chunk_size` appropriée lors de l'appel à `unet.enable_forward_chunking`."
                    )

                num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                ff_output = torch.cat(
                    [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                    dim=self._chunk_dim,
                )
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states
else:
    # Bloc BasicTransformerBlock de repli si diffusers n'est pas disponible
    class BasicTransformerBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            activation_fn: str = "geglu",
            **kwargs
        ):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = nn.MultiheadAttention(dim, num_attention_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
            
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            **kwargs
        ):
            # Auto-attention
            norm_hidden_states = self.norm1(hidden_states)
            attn_output, _ = self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states)
            hidden_states = hidden_states + attn_output
            
            # Feed-forward
            norm_hidden_states = self.norm2(hidden_states)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = hidden_states + ff_output
            
            return hidden_states
