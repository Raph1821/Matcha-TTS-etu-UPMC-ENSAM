import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Encode le temps 't' en vecteurs sinusoïdaux"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DecoderBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        # Le temps est injecté via une couche linéaire
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2)
        )
        
        # Convolutions principales (Dilated ConvNet simplifié)
        self.block1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, 1, 1),
        )

    def forward(self, x, time_emb, mask=None):
        # x: [Batch, Channels, Time]
        
        # 1. On projette le temps et on l'ajoute au signal (Scale & Shift)
        # time_emb : [Batch, Channels*2] -> [Batch, Channels*2, 1]
        time_style = self.mlp(time_emb).unsqueeze(-1)
        scale, shift = time_style.chunk(2, dim=1)
        
        h = x * (scale + 1) + shift
        
        # 2. Convolution
        h = self.block1(h)
        
        # 3. Connexion résiduelle
        return (x + h) * mask if mask is not None else (x + h)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # Encodage du temps
        self.time_dim = hidden_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

        # Entrée : On concatène le signal bruité (in_channels) + la condition mu (in_channels)
        self.init_conv = nn.Conv1d(in_channels * 2, hidden_channels, 1)

        # Pile de blocs décodeurs
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_channels, hidden_channels),
            DecoderBlock(hidden_channels, hidden_channels),
            DecoderBlock(hidden_channels, hidden_channels),
        ])

        # Sortie : on prédit la vitesse (même dimension que le Mel)
        self.final_conv = nn.Conv1d(hidden_channels, out_channels, 1)
        
        # Initialisation à zéro de la dernière couche pour la stabilité (Flow Matching astuce)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t, mu, mask):
        # x: [B, 80, T] (Signal bruité)
        # t: [B] (Temps)
        # mu: [B, 80, T] (Condition acoustique venant de l'encodeur)
        
        # 1. Traitement du temps
        t_emb = self.time_mlp(t) # [B, hidden]

        # 2. Fusion Signal + Condition
        # On colle mu avec x pour guider la génération
        x_input = torch.cat([x, mu], dim=1) 
        x = self.init_conv(x_input)
        
        if mask is not None:
             mask = mask.unsqueeze(1) # [B, 1, T]
             x = x * mask

        # 3. Passage dans les blocs
        for block in self.blocks:
            x = block(x, t_emb, mask)

        # 4. Projection finale
        return self.final_conv(x)