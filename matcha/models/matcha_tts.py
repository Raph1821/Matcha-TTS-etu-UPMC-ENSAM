import datetime as dt
import math
import random

import torch

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 
    def __init__(
        self,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
        n_feats=None,
        encoder=None,
        decoder=None,
        cfm=None,
        data_statistics=None,
        out_size=None,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
        # 简化参数接口（用于直接调用）
        out_channels=None,
        hidden_channels=None,
    ):
        super().__init__()

        # 判断使用哪种初始化方式：配置对象方式 or 简化参数方式
        if encoder is not None and decoder is not None and cfm is not None:
            # 原版配置对象方式
            self._init_from_config(
                n_vocab, n_spks, spk_emb_dim, n_feats, encoder, decoder, cfm,
                data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
            )
        else:
            # 简化参数方式
            if out_channels is None or hidden_channels is None:
                raise ValueError(
                    "Either provide (encoder, decoder, cfm) config objects, "
                    "or provide (out_channels, hidden_channels) for simplified initialization"
                )
            self._init_from_simple_params(
                n_vocab, out_channels, hidden_channels, n_spks, spk_emb_dim,
                data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
            )

    def _init_from_config(
        self, n_vocab, n_spks, spk_emb_dim, n_feats, encoder, decoder, cfm,
        data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
    ):
        """原版配置对象初始化方式"""
        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)

    def _init_from_simple_params(
        self, n_vocab, out_channels, hidden_channels, n_spks=1, spk_emb_dim=128,
        data_statistics=None, out_size=None, optimizer=None, scheduler=None,
        prior_loss=True, use_precomputed_durations=False
    ):
        """简化参数初始化方式（复现版本，不完全复制）"""
        # 保存超参数，但不包括 optimizer 和 scheduler（它们会在 configure_optimizers 中处理）
        self.save_hyperparameters(logger=False, ignore=['optimizer', 'scheduler'])
        
        # 保存优化器和调度器配置（如果提供）
        self.scheduler_config = scheduler

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = out_channels  # n_feats 就是 out_channels
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        # 创建简化的配置对象
        from types import SimpleNamespace
        
        # Encoder 配置
        encoder_params = SimpleNamespace(
            n_feats=out_channels,
            n_channels=hidden_channels,
            filter_channels=768,  # 默认值
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            prenet=True,
        )
        
        duration_predictor_params = SimpleNamespace(
            filter_channels_dp=256,
            kernel_size=3,
            p_dropout=0.1,
        )
        
        encoder_config = SimpleNamespace(
            encoder_type="transformer",
            encoder_params=encoder_params,
            duration_predictor_params=duration_predictor_params,
        )
        
        # Decoder 配置
        decoder_params = {
            "channels": (256, 256),
            "dropout": 0.05,
            "attention_head_dim": 64,
            "n_blocks": 1,
            "num_mid_blocks": 2,
            "num_heads": 4,
            "act_fn": "snake",
            "down_block_type": "transformer",
            "mid_block_type": "transformer",
            "up_block_type": "transformer",
        }
        
        # CFM 配置
        cfm_params = SimpleNamespace(
            solver="euler",
            sigma_min=1e-4,
        )
        
        # 初始化组件
        self.encoder = TextEncoder(
            encoder_config.encoder_type,
            encoder_config.encoder_params,
            encoder_config.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * out_channels,
            out_channel=out_channels,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # 数据统计（如果没有提供，使用默认值）
        if data_statistics is None:
            data_statistics = {"mel_mean": 0.0, "mel_std": 1.0}
        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn
