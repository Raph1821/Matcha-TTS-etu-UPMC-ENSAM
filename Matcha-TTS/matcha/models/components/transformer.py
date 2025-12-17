"""
Transformer 核心组件模块 - 完全重写版本

本模块提供 Matcha-TTS 解码器中使用的 Transformer 基础组件，所有实现均为从零开始，
不依赖外部库（如 diffusers），但保持与原有接口完全兼容。

模块组成:
1. SnakeBeta: 周期性激活函数，适用于音频信号处理
2. FeedForward: 前馈神经网络层，支持多种激活函数
3. BasicTransformerBlock: Transformer 基础块，包含自注意力、交叉注意力和前馈网络

这些组件被 decoder.py 使用，用于构建 U-Net 风格的解码器架构。
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 模块 1: 激活函数实现
# ============================================================================

class SnakeBeta(nn.Module):
    """
    SnakeBeta 周期性激活函数
    
    专门为音频信号设计的激活函数，能够捕获信号的周期性特征。
    公式: SnakeBeta(x) = x + (1/β) * sin²(x * α)
    
    参数:
        - α (alpha): 控制周期性频率的可训练参数
        - β (beta): 控制周期性幅度的可训练参数
    
    输入形状: (B, T, C) 或 (B, C, T)
    输出形状: 与输入相同
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        初始化 SnakeBeta 激活函数
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            alpha: α 参数的初始值，默认 1.0
            alpha_trainable: α 和 β 参数是否可训练
            alpha_logscale: 是否使用对数尺度初始化
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        
        # 线性投影层
        self.proj = nn.Linear(in_features, out_features)

        # 初始化 α 和 β 参数
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            # 对数尺度：初始化为零
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            # 线性尺度：初始化为指定值
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        """前向传播：应用 SnakeBeta 激活函数"""
        x = self.proj(x)
        
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        # SnakeBeta 公式: x + (1/β) * sin²(x * α)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函数
    
    公式: GELU(x) = x * Φ(x)，其中 Φ(x) 是标准正态分布的累积分布函数
    近似实现: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    
    def __init__(self, dim, inner_dim, approximate=None):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.approximate = approximate
        self.proj = nn.Linear(dim, inner_dim)
    
    def forward(self, x):
        x = self.proj(x)
        if self.approximate == "tanh":
            # 近似 GELU
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        else:
            # 标准 GELU
            return F.gelu(x)


class GEGLU(nn.Module):
    """
    Gated GELU (GEGLU) 激活函数
    
    门控版本的 GELU，将输入分成两部分，一部分经过 GELU，另一部分作为门控信号。
    公式: GEGLU(x) = GELU(x[:d]) ⊙ x[d:]
    """
    
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        # 输出维度是 inner_dim，但需要 2*inner_dim 的输入
        self.proj = nn.Linear(dim, inner_dim * 2)
    
    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(x) * gate


class ApproximateGELU(nn.Module):
    """近似 GELU 激活函数（使用 tanh 近似）"""
    
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.proj = nn.Linear(dim, inner_dim)
    
    def forward(self, x):
        x = self.proj(x)
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# ============================================================================
# 模块 2: 注意力机制实现
# ============================================================================

class Attention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    支持两种模式:
    1. 自注意力 (Self-Attention): query, key, value 都来自同一输入
    2. 交叉注意力 (Cross-Attention): query 来自输入，key/value 来自编码器
    
    注意力公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        upcast_attention: bool = False,
    ):
        """
        初始化注意力层
        
        Args:
            query_dim: Query 的维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout 概率
            bias: 是否使用偏置
            cross_attention_dim: 交叉注意力的编码器维度（None 表示自注意力）
            upcast_attention: 是否使用高精度计算
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5  # 缩放因子 1/√d_k
        self.cross_attention_dim = cross_attention_dim
        self.upcast_attention = upcast_attention
        
        inner_dim = heads * dim_head
        
        # Query 投影
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        
        # Key 和 Value 投影
        if cross_attention_dim is None:
            # 自注意力：key 和 value 来自同一输入
            self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        else:
            # 交叉注意力：key 和 value 来自编码器
            self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        
        # 输出投影
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        前向传播
        
        Args:
            hidden_states: 输入特征，形状 (B, T, C)
            encoder_hidden_states: 编码器特征（交叉注意力时使用），形状 (B, T_enc, C_enc)
            attention_mask: 注意力掩码，形状 (B, T) 或 (B, T, T)
            
        Returns:
            注意力输出，形状 (B, T, C)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算 Query
        q = self.to_q(hidden_states)
        
        # 计算 Key 和 Value
        if encoder_hidden_states is None:
            # 自注意力
            k = self.to_k(hidden_states)
            v = self.to_v(hidden_states)
        else:
            # 交叉注意力
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
        
        # 重塑为多头形式: (B, T, C) -> (B, H, T, d)
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # 计算注意力分数: QK^T / √d_k
        if self.upcast_attention:
            # 高精度计算
            with torch.autocast(device_type="cuda", enabled=False):
                q = q.float()
                k = k.float()
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 处理不同维度的掩码
            if attention_mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T) - 广播到所有头和时间步
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (B, T_q, T_k) -> (B, 1, T_q, T_k)
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                # (B, H, T_q, T_k) - 已经是正确的形状
                pass
            
            # 将掩码中的 0 位置设为负无穷（掩码为 1 的位置保留，0 的位置设为 -inf）
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        
        # Softmax 归一化
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        attn_output = torch.matmul(attn_probs, v)
        
        # 重塑回原始形状: (B, H, T, d) -> (B, T, H*d)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.heads * self.dim_head)
        
        # 输出投影
        output = self.to_out(attn_output)
        
        return output


# ============================================================================
# 模块 3: 归一化层实现
# ============================================================================

class AdaLayerNorm(nn.Module):
    """
    自适应层归一化 (Adaptive Layer Normalization)
    
    根据扩散模型的时间步进行自适应归一化，用于条件生成。
    结合了 LayerNorm 和时间步嵌入。
    """
    
    def __init__(self, embedding_dim: int, num_embeddings: int):
        """
        初始化 AdaLayerNorm
        
        Args:
            embedding_dim: 特征维度
            num_embeddings: 时间步嵌入的数量（扩散步数）
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # 时间步嵌入
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        
        # 归一化层
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 输入特征，形状 (B, T, C)
            timestep: 时间步，形状 (B,)
            
        Returns:
            归一化后的特征，形状 (B, T, C)
        """
        # 获取时间步嵌入
        emb = self.emb(timestep)
        emb = self.silu(emb)
        emb = self.linear(emb)
        
        # 分离缩放和偏移参数
        scale, shift = emb.chunk(2, dim=-1)
        
        # LayerNorm
        x = F.layer_norm(x, (self.embedding_dim,))
        
        # 应用自适应缩放和偏移
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


class AdaLayerNormZero(nn.Module):
    """
    零初始化的自适应层归一化 (AdaNormZero)
    
    与 AdaLayerNorm 类似，但使用零初始化，训练更稳定。
    返回额外的门控参数用于控制注意力和前馈网络。
    """
    
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # 时间步嵌入
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        
        # 归一化层（零初始化）
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 6)
        
        # 零初始化线性层
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, class_labels: Optional[torch.Tensor] = None, hidden_dtype=None):
        """
        前向传播
        
        Args:
            x: 输入特征，形状 (B, T, C)
            timestep: 时间步，形状 (B,)
            class_labels: 类别标签（可选）
            hidden_dtype: 隐藏状态的数据类型
            
        Returns:
            norm_hidden_states: 归一化后的特征
            gate_msa: 自注意力的门控参数
            shift_mlp: 前馈网络的偏移参数
            scale_mlp: 前馈网络的缩放参数
            gate_mlp: 前馈网络的门控参数
        """
        # 获取时间步嵌入
        emb = self.emb(timestep)
        emb = self.silu(emb)
        emb = self.linear(emb)
        
        # 分离所有参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        
        # LayerNorm
        norm_hidden_states = F.layer_norm(x, (self.embedding_dim,))
        
        # 应用自适应缩放和偏移（用于自注意力）
        norm_hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ============================================================================
# 模块 4: 前馈网络实现
# ============================================================================

class FeedForward(nn.Module):
    """
    前馈神经网络层 (Feed-Forward Network)
    
    Transformer 架构中的标准前馈层，用于非线性特征变换。
    结构: 输入 -> 激活函数 -> Dropout -> 线性投影 -> (可选) Dropout -> 输出
    
    支持的激活函数:
        - "gelu": 标准 GELU
        - "gelu-approximate": 近似 GELU
        - "geglu": 门控 GELU
        - "geglu-approximate": 近似门控 GELU
        - "snakebeta": SnakeBeta 周期性激活
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
        """
        初始化前馈网络层
        
        Args:
            dim: 输入特征维度
            dim_out: 输出特征维度（None 时等于 dim）
            mult: 隐藏层维度倍数（inner_dim = dim * mult）
            dropout: Dropout 概率
            activation_fn: 激活函数类型
            final_dropout: 是否在输出前应用额外的 Dropout
        """
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # 根据配置选择激活函数
        if activation_fn == "gelu":
            self.act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            self.act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            self.act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            self.act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            self.act_fn = SnakeBeta(dim, inner_dim)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # 构建网络模块
        self.dropout1 = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim_out)
        self.dropout2 = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(self, hidden_states):
        """
        前向传播
        
        Args:
            hidden_states: 输入特征，形状 (B, T, C)
            
        Returns:
            变换后的特征，形状 (B, T, dim_out)
        """
        # 激活函数（包含输入投影）
        hidden_states = self.act_fn(hidden_states)
        # Dropout
        hidden_states = self.dropout1(hidden_states)
        # 输出投影
        hidden_states = self.proj_out(hidden_states)
        # 可选的最终 Dropout
        hidden_states = self.dropout2(hidden_states)
        
        return hidden_states


# ============================================================================
# 模块 5: Transformer 基础块实现
# ============================================================================

class BasicTransformerBlock(nn.Module):
    """
    Transformer 基础块
    
    Transformer 架构的核心组件，包含三个主要子模块：
    1. 自注意力 (Self-Attention): 捕获序列内部的时间依赖关系
    2. 交叉注意力 (Cross-Attention): 融合编码器输出（如文本特征）
    3. 前馈网络 (Feed-Forward): 进行非线性特征变换
    
    每个子模块都遵循 Pre-Norm 架构：
    - 先进行 Layer Normalization
    - 然后执行主要计算
    - 最后通过残差连接相加
    
    特殊功能:
        - 支持 AdaLayerNorm: 根据扩散时间步自适应归一化
        - 支持分块前馈: 可节省内存处理长序列
        - 支持多种注意力模式: 仅自注意力、仅交叉注意力、双重自注意力
    
    在 Matcha-TTS 中的作用:
        - 处理 mel-spectrogram 的时间序列特征
        - 融合文本编码器的输出
        - 支持 Flow Matching 的时间步条件化
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
        """
        初始化 Transformer 块
        
        Args:
            dim: 特征维度
            num_attention_heads: 多头注意力的头数
            attention_head_dim: 每个注意力头的维度
            dropout: Dropout 概率
            cross_attention_dim: 交叉注意力的编码器特征维度
            activation_fn: 前馈网络使用的激活函数
            num_embeds_ada_norm: AdaNorm 的嵌入数量（用于扩散模型）
            attention_bias: 注意力层是否使用偏置
            only_cross_attention: 是否只使用交叉注意力
            double_self_attention: 是否使用双重自注意力
            upcast_attention: 是否使用高精度注意力计算
            norm_elementwise_affine: LayerNorm 是否使用可学习的仿射参数
            norm_type: 归一化类型
            final_dropout: 前馈网络是否使用最终 Dropout
        """
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # 确定使用的归一化类型
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        # 验证归一化配置
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined."
            )

        # ====================================================================
        # 子模块 1: 自注意力 (Self-Attention)
        # ====================================================================
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

        # ====================================================================
        # 子模块 2: 交叉注意力 (Cross-Attention) - 可选
        # ====================================================================
        if cross_attention_dim is not None or double_self_attention:
            # 注意：AdaNormZero 只用于第一个注意力块
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
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # ====================================================================
        # 子模块 3: 前馈网络 (Feed-Forward Network)
        # ====================================================================
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim, 
            dropout=dropout, 
            activation_fn=activation_fn, 
            final_dropout=final_dropout
        )

        # 分块前馈的配置（用于内存优化）
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        """
        设置前馈网络的分块大小（用于内存优化）
        
        Args:
            chunk_size: 每个块的大小（None 表示不分块）
            dim: 分块的维度索引
        """
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
        """
        前向传播：依次通过三个子模块
        
        数据流:
        1. 自注意力: hidden_states -> norm1 -> attn1 -> + hidden_states
        2. 交叉注意力: hidden_states -> norm2 -> attn2 -> + hidden_states (可选)
        3. 前馈网络: hidden_states -> norm3 -> ff -> + hidden_states
        
        Args:
            hidden_states: 输入特征，形状 (B, T, C)
            attention_mask: 自注意力的掩码，形状 (B, T)
            encoder_hidden_states: 编码器输出（用于交叉注意力）
            encoder_attention_mask: 编码器掩码
            timestep: 扩散时间步（用于 AdaNorm）
            cross_attention_kwargs: 交叉注意力的额外参数
            class_labels: 类别标签（用于 AdaNormZero）
            
        Returns:
            处理后的特征，形状 (B, T, C)
        """
        # ====================================================================
        # 步骤 1: 自注意力 (Self-Attention)
        # ====================================================================
        # Pre-Norm: 先归一化
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 准备交叉注意力参数
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 执行注意力计算
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        
        # AdaNormZero: 应用门控机制
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output     
        
        # 残差连接
        hidden_states = attn_output + hidden_states

        # ====================================================================
        # 步骤 2: 交叉注意力 (Cross-Attention) - 可选
        # ====================================================================
        if self.attn2 is not None:
            # Pre-Norm: 先归一化
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # 执行交叉注意力（融合编码器特征）
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            
            # 残差连接
            hidden_states = attn_output + hidden_states

        # ====================================================================
        # 步骤 3: 前馈网络 (Feed-Forward Network)
        # ====================================================================
        # Pre-Norm: 先归一化
        norm_hidden_states = self.norm3(hidden_states)

        # AdaNormZero: 应用缩放和偏移
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # 执行前馈网络（支持分块以节省内存）
        if self._chunk_size is not None:
            # 分块处理
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} "
                    f"has to be divisible by chunk size: {self._chunk_size}."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            # 正常处理
            ff_output = self.ff(norm_hidden_states)

        # AdaNormZero: 应用门控机制
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        # 残差连接
        hidden_states = ff_output + hidden_states

        return hidden_states
