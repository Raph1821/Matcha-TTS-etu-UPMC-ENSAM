class EncoderBlock(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, p_dropout):
        super().__init__()
        # 1. Multi-Head Attention
        self.attn = nn.MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, batch_first=True)
        
        # 2. Réseau à convolution (Feed Forward alternatif)
        self.conv_net = nn.Sequential(
            nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.Dropout(p_dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x, mask=None):
        # x shape: [batch, hidden_channels, len]
        x_transpose = x.transpose(1, 2) # [batch, len, hidden_channels]
        
        # Attention (avec masque si nécessaire)
        attn_out, _ = self.attn(x_transpose, x_transpose, x_transpose, key_padding_mask=mask)
        x = self.norm1(x_transpose + attn_out)
        
        # Convolution
        x = x.transpose(1, 2)
        conv_out = self.conv_net(x)
        x = self.norm2((x + conv_out).transpose(1, 2)).transpose(1, 2)
        
        return x

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, 
                 n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # Création de la pile de blocs
        self.encoder_stack = nn.ModuleList([
            EncoderBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths):
        # x: [batch, max_text_len]
        
        # Masque pour ignorer le padding (très important en TTS)
        # Crée un masque True là où il y a du padding
        max_len = x.size(1)
        mask = torch.arange(max_len).to(x.device)[None, :] >= x_lengths[:, None]

        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        x = x.transpose(1, 2) # [batch, channels, len]

        for block in self.encoder_stack:
            x = block(x, mask=mask)

        # mu: [batch, out_channels, len]
        mu = self.proj(x)
        
        return mu