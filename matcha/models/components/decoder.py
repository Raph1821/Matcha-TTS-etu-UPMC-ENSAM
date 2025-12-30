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
        assert dim % 2 == 0, "Dimension required by SinusoidalPosEmb must be even"

    def forward(self, x, scale=1000):    #  t varie entre 0 et 1 -> les différences entre les timesteps sont trop petites -> on le scale pour avoir une meilleure répartition des fréquences
        # 处理标量输入（0维张量）
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)  # [Batch_size, Half_Dim] -> tenseur [t] de forme [B] -> [B,1] x vecteur de fréquences [emb] de forme [Half_Dim] -> [1, Half_Dim] == [B, Half_Dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeStepEmbeddingNet(nn.Module):
    """
    Crée un embedding pour un timestep (int ou float) à utiliser dans le décodeur TTS.
    Souvent utilisé dans les architectures type diffusion ou attention-based TTS.
    """
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        
        # 1. Première couche linéaire : in_channels → time_embed_dim
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        
        # 2. Fonction d'activation (SiLU, ReLU, GELU, etc.)
        self.act = nn.SiLU()  
        
        # 3. Deuxième couche linéaire : time_embed_dim → time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, sample):
        # 4. Pipeline de transformation
        sample = self.linear_1(sample)        # Projection
        sample = self.act(sample)             # Non-linéarité
        sample = self.linear_2(sample)        # Projection finale
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
            nn.GroupNorm(groups, out_channels),     # indépendante du batch
            nn.Mish(),  # plus lisse, gradients plus stables (meilleure que ReLU)
        )

    # Le masque est crucial si :
        # padding variable
        # données temporelles irrégulières
        # modèles type Transformer / diffusion

    def forward(self, x , mask):    # architecture consciente du padding / séquence
        output = self.block (x*mask)
        return output*mask

class Resnet1D(nn.Module):
    """
    Bloc ResNet 1D avec injection de timestep embedding.
    Utilisé dans le décodeur pour capturer des dépendances temporelles complexes.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, padding=1, groups=8):
        super().__init__()
        # MLP pour le timestep embedding
        self.mlp = nn.Sequential(nn.Mish(),nn.Linear(time_emb_dim,out_channels))
        
        # Première convolution
        self.block1 = Block1D(in_channels, out_channels, kernel_size, padding, groups)
        
        # Deuxième convolution
        self.block2 = Block1D(out_channels, out_channels, kernel_size, padding, groups)
        
        # Connexion résiduelle (skip connection)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) 
    
    def forward(self, x, mask, time_emb):
        
        # Première convolution
        h = self.block1(x, mask)        # [Batch, out_channels, Time]

        # Injection du timestep
        time_out = self.mlp(time_emb)   # [Batch, out_channels]
        h += time_out[:, :, None]       # [Batch, out_channels, 1] -> broadcasting sur la dimension temporelle
        
        # Deuxième convolution
        h = self.block2(h, mask)        # [Batch, out_channels, Time]

        # Connexion résiduelle
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
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
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

        # Encodage du timestep
        self.time_embeddings = SinusoidalPosEmb(dim=in_channels)
        time_embed_dim = channels[0] * 4    # Convention des diffusion models : le timestep embedding est 4 fois plus grand que la dimension de base du réseau
        self.time_mlp = TimeStepEmbeddingNet(in_channels, time_embed_dim)

        # Listes de blocs
        self.Downsampling_Blocks = nn.ModuleList([])
        self.Mid_Blocks = nn.ModuleList([])
        self.Upsampling_Blocks = nn.ModuleList([])

        # Construction des downsampling blocks
        output_channels = in_channels
        for i in range(len(channels)):  # 2 fois par block pour downsampling
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

        # Construction des mid blocks
        for i in range(num_mid_blocks): # 2 fois par bloc pour mid block dans ce cas 
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

        # Construction des upsampling blocks
        reserved_channels = channels[::-1] + (channels[0],)
        for i in range(len(reserved_channels)-1):
            input_channels = reserved_channels[i]
            output_channels = reserved_channels[i+1]
            is_last = i == (len(reserved_channels) - 2)

            # x2 car on concatène avec skip connection
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

        # Couche de sortie finale
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
                nn.init.constant_(m.weight, 1)  # initialise le fateur de gain gamma à 1
                nn.init.constant_(m.bias, 0)    # initialise le biais beta à 0
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
        # Encoder le timestep
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # Concaténer x et mu (condition)
        x = pack([x, mu], "b * t")[0]   # même batch même time dimension -> concat sur la dimension des channels -> x : (batch_size, in_channels + n_feats, time)

        # Downsampling blocks avec skip connections
        hiddens = []    # pour stocker les skip connections
        masks = [mask]  # pour stocker les masques à chaque résolution
        
        for resnet, transformer_blocks, downsample in self.Downsampling_Blocks:
            mask_down = masks[-1]   # récupérer le masque correspondant à la résolution actuelle
            
            # ResNet block
            x = resnet(x, mask_down, t)
            
            # Transformer blocks
            x = rearrange(x, "b c t -> b t c")  # changer la forme pour le transformer à (batch_size, time, channels)
            mask_down = rearrange(mask_down, "b 1 t -> b t")    # le transformer attend un masque de forme (batch_size, time)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            
            # Sauvegarder pour skip connection
            hiddens.append(x)
            
            # Downsample
            x = downsample(x * mask_down)       # Réduction de la résolution temporelle
            # Mettre à jour le mask : taille après sous-échantillonnage
            if isinstance(downsample, Downsample1D):
                # Conv1d stride=2 sous-échantillonnage, taille divisée par 2
                new_mask_size = (mask_down.shape[-1] + 1) // 2
            else:
                # Le dernier block ne fait pas de sous-échantillonnage, taille inchangée
                new_mask_size = mask_down.shape[-1]
            # Créer un nouveau mask, maintenir la cohérence avec la taille après sous-échantillonnage
            new_mask = mask_down[:, :, :new_mask_size]
            masks.append(new_mask)

        # Mid blocks
        masks = masks[:-1]      # retirer le dernier masque (pas de downsample après le mid block)
        mask_mid = masks[-1]    # masque à la résolution actuelle
        
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

        # Up blocks avec skip connections
        for resnet, transformer_blocks, upsample in self.Upsampling_Blocks:
            mask_up = masks.pop()   # récupérer le masque correspondant à la résolution actuelle
            hidden = hiddens.pop()
            
            # S'assurer que les dimensions de la connexion skip correspondent
            if x.shape[-1] != hidden.shape[-1]:
                # Si les dimensions ne correspondent pas, utiliser l'interpolation pour ajuster
                x = F.interpolate(x, size=hidden.shape[-1], mode='nearest', align_corners=None)
            
            # Concaténer avec skip connection
            x = pack([x, hidden], "b * t")[0]    # concat features actuelles avec skip connection
            
            # ResNet block
            x = resnet(x, mask_up, t)
            
            # Transformer blocks
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            
            # Upsample
            x = upsample(x * mask_up)
            
            # Mettre à jour le mask : taille après sur-échantillonnage
            if isinstance(upsample, Upsample1D):
                # ConvTranspose1d stride=2 sur-échantillonnage, taille doublée
                new_mask_size = mask_up.shape[-1] * 2
            else:
                # Le dernier block ou utilisation d'interpolation, taille peut être différente
                new_mask_size = x.shape[-1]
            # Créer un nouveau mask, maintenir la cohérence avec la taille après sur-échantillonnage
            if new_mask_size > mask_up.shape[-1]:
                # Étendre le mask
                mask_up = F.interpolate(mask_up, size=new_mask_size, mode='nearest', align_corners=None)
            else:
                mask_up = mask_up[:, :, :new_mask_size]

        # Couches finales
        x = self.final_conv(x * mask_up)
        x = self.final_norm(x)
        x = self.final_act(x)
        output = self.final_proj(x * mask_up)

        return output * mask