import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange
from matcha.models.components.transformer import BasicTransformerBlock

class SinusoidalPosEmb(nn.Module):
    """
    Encode le temps 't' en vecteurs sinusoïdaux
    Le décodeur reçoit un temps t (entre 0 et 1) qui indique où on en est (positional embedding) dans le processus de flow 
    matching. Mais un seul nombre (scalar) n'est pas assez riche pour que le réseau comprenne bien :

    t = 0.5 → juste 1 nombre
    Encodage sinusoïdal → 192 nombres (vecteur riche en information) 
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "La dimension doit être paire pour SinusoidalPosEmb"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeStepEmbeddingNet(nn.Module):
    """
    Crée un embedding pour un timestep (int ou float) à utiliser dans le décodeur TTS.
    Souvent utilisé dans les architectures type diffusion ou attention-based TTS.
    """
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
    
class Block1D(nn.Module):
    """
    Bloc de base 1D avec Conv1D, GroupNorm et Mish activation.
    Utilisé dans le décodeur pour le traitement des séquences temporelles.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x , mask):
        output = self.block (x*mask)
        return output*mask

class Resnet1D(nn.Module):
    """
    Bloc ResNet 1D avec injection de timestep embedding.
    Utilisé dans le décodeur pour capturer des dépendances temporelles complexes.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(),nn.Linear(time_emb_dim,out_channels))
        self.block1 = Block1D(in_channels, out_channels, kernel_size, padding, groups)
        self.block2 = Block1D(out_channels, out_channels, kernel_size, padding, groups)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) 
    
    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        time_out = self.mlp(time_emb)
        h += time_out[:, :, None]
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)  
        return output

class Downsample1D(nn.Module):
    """
    Module de downsampling 1D.
    Utilisé pour réduire la résolution temporelle dans le décodeur.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample1D(nn.Module):
    """Couche de sur-échantillonnage 1D avec convolution optionnelle.

    Paramètres:
        channels (int): Nombre de canaux en entrée et sortie.
        out_channels (int, optionnel): Nombre de canaux de sortie. Par défaut égal à `channels`.
    """
    def __init__(self, channels,out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)

class Decoder(nn.Module):
    """
    Decoder U-Net 1D simplifié pour le flow matching.
    Utilise uniquement des blocs Transformer (pas de Conformer).
    
    channels=(256, 256), n_blocks=1, num_heads=4

    Input (in_channels)
        ↓
    Down Block 1: ResNet → Transformer (4 heads, dim 64) → Downsample
        256 canaux, 1 transformer
        ↓
    Down Block 2: ResNet → Transformer (4 heads, dim 64) → Conv
        256 canaux, 1 transformer
        ↓
    Mid Block 1: ResNet → Transformer
        256 canaux ← num_mid_blocks=2
        ↓
    Mid Block 2: ResNet → Transformer
        256 canaux
        ↓
    Up Block 1: ResNet → Transformer → Upsample
        256 canaux
        ↓
    Up Block 2: ResNet → Transformer → Conv
        256 canaux
        ↓
    Output (out_channels)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256,256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.time_embeddings = SinusoidalPosEmb(dim=in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimeStepEmbeddingNet(in_channels, time_embed_dim)

        self.Downsampling_Blocks = nn.ModuleList([])
        self.Mid_Blocks = nn.ModuleList([])
        self.Upsampling_Blocks = nn.ModuleList([])

        output_channels = in_channels
        for i in range(len(channels)):
            input_channels = output_channels
            output_channels = channels[i]
            is_last = i == len(channels) - 1

            resnet = Resnet1D(input_channels, output_channels, time_embed_dim)
            transformer = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channels,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                )
                for _ in range(n_blocks)
            ])

            downsample = (
                Downsample1D(output_channels) 
                if not is_last 
                else nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
            )

            self.Downsampling_Blocks.append(nn.ModuleList([
                resnet,
                transformer,
                downsample,
            ]))

        for i in range(num_mid_blocks): 
            resnet = Resnet1D(channels[-1], channels[-1], time_embed_dim)
            transformer = nn.ModuleList([
                BasicTransformerBlock(
                    dim=channels[-1],
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                )
                for _ in range(n_blocks)
            ])
            self.Mid_Blocks.append(nn.ModuleList([
                resnet,
                transformer,
            ]))

        reserved_channels = channels[::-1] + (channels[0],)
        for i in range(len(reserved_channels)-1):
            input_channels = reserved_channels[i]
            output_channels = reserved_channels[i+1]
            is_last = i == (len(reserved_channels) - 2)

            resnet = Resnet1D(input_channels*2, output_channels, time_embed_dim)
            transformer = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channels,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                )
                for _ in range(n_blocks)
            ])

            upsample = (
                Upsample1D(output_channels) 
                if not is_last 
                else nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
            )

            self.Upsampling_Blocks.append(nn.ModuleList([
                resnet,
                transformer,
                upsample,
            ]))

        self.final_conv = nn.Conv1d(channels[0], channels[0], kernel_size=3, padding=1)
        self.final_norm = nn.GroupNorm(8, channels[0])
        self.final_act = nn.Mish()
        self.final_proj = nn.Conv1d(channels[0], self.out_channels, kernel_size=1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialisation des poids"""
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self,x,mask,mu,t,cond=None):
        """
        Forward pass du decoder U-Net.

        Args:
            x (torch.Tensor): Bruit, shape (batch_size, in_channels, time)
            mask (torch.Tensor): Masque, shape (batch_size, 1, time)
            mu (torch.Tensor): Sortie de l'encoder, shape (batch_size, n_feats, time)
            t (torch.Tensor): Timesteps, shape (batch_size,)
            cond (torch.Tensor, optional): Conditioning additionnel (non utilisé)

        Returns:
            torch.Tensor: Prédiction, shape (batch_size, out_channels, time)
        """
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        hiddens = []
        masks = [mask]
        
        for resnet, transformer_blocks, downsample in self.Downsampling_Blocks:
            mask_down = masks[-1]
            
            x = resnet(x, mask_down, t)
            
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            
            hiddens.append(x)
            
            x = downsample(x * mask_down)
            if isinstance(downsample, Downsample1D):
                new_mask_size = (mask_down.shape[-1] + 1) // 2
            else:
                new_mask_size = mask_down.shape[-1]
            new_mask = mask_down[:, :, :new_mask_size]
            masks.append(new_mask)

        masks = masks[:-1]
        mask_mid = masks[-1]
        
        for resnet, transformer_blocks in self.Mid_Blocks:
            x = resnet(x, mask_mid, t)
            
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.Upsampling_Blocks:
            mask_up = masks.pop()
            hidden = hiddens.pop()
            
            if x.shape[-1] != hidden.shape[-1]:
                x = F.interpolate(x, size=hidden.shape[-1], mode='nearest', align_corners=None)
            
            x = pack([x, hidden], "b * t")[0]
            
            x = resnet(x, mask_up, t)
            
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            
            x = upsample(x * mask_up)
            
            if isinstance(upsample, Upsample1D):
                new_mask_size = mask_up.shape[-1] * 2
            else:
                new_mask_size = x.shape[-1]
            if new_mask_size > mask_up.shape[-1]:
                mask_up = F.interpolate(mask_up, size=new_mask_size, mode='nearest', align_corners=None)
            else:
                mask_up = mask_up[:, :, :new_mask_size]

        x = self.final_conv(x * mask_up)
        x = self.final_norm(x)
        x = self.final_act(x)
        output = self.final_proj(x * mask_up)

        return output * mask