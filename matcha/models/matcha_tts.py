"""
Module principal Matcha-TTS

Implémentation complète du modèle Matcha-TTS avec support de deux méthodes d'initialisation:
- Initialisation par objets de configuration (compatible avec Hydra)
- Initialisation par paramètres simplifiés (pour usage direct)
"""

import datetime as dt
import math
import random

import torch

import matcha.utils.monotonic_align as monotonic_align
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import ConditionalFlowMatching
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


class MatchaTTS(BaseLightningClass):
    """
    Modèle Matcha-TTS complet
    
    Architecture:
    - Encodeur de texte avec prédiction de durée
    - Décodeur basé sur Flow Matching (CFM)
    - Alignement monotone (MAS) pour synchronisation texte-audio
    - Trois types de pertes: durée, prior, flow matching
    """
    
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
        # Interface de paramètres simplifiés (pour appel direct)
        out_channels=None,
        hidden_channels=None,
    ):
        super().__init__()

        # Détecter quelle méthode d'initialisation utiliser
        if encoder is not None and decoder is not None and cfm is not None:
            # Méthode originale avec objets de configuration
            self._init_from_config(
                n_vocab, n_spks, spk_emb_dim, n_feats, encoder, decoder, cfm,
                data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
            )
        else:
            # Méthode avec paramètres simplifiés
            if out_channels is None or hidden_channels is None:
                raise ValueError(
                    "Soit fournir les objets de configuration (encoder, decoder, cfm), "
                    "soit fournir (out_channels, hidden_channels) pour l'initialisation simplifiée"
                )
            self._init_from_simple_params(
                n_vocab, out_channels, hidden_channels, n_spks, spk_emb_dim,
                data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
            )

    def _init_from_config(
        self, n_vocab, n_spks, spk_emb_dim, n_feats, encoder, decoder, cfm,
        data_statistics, out_size, optimizer, scheduler, prior_loss, use_precomputed_durations
    ):
        """Initialisation avec objets de configuration (méthode originale)"""
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

        self.decoder = ConditionalFlowMatching(
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
        """Initialisation avec paramètres simplifiés (version de reproduction)"""
        # Sauvegarder les hyperparamètres, exclure optimizer et scheduler (gérés dans configure_optimizers)
        self.save_hyperparameters(logger=False, ignore=['optimizer', 'scheduler'])
        
        # Sauvegarder la configuration du scheduler (si fournie)
        self.scheduler_config = scheduler

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = out_channels  # n_feats correspond à out_channels
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        # Créer des objets de configuration simplifiés
        from types import SimpleNamespace
        
        
        
        # Configuration du décodeur
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
        
        # Configuration CFM
        cfm_params = SimpleNamespace(
            solver="euler",
            sigma_min=1e-4,
        )
        
        # Initialiser les composants
        self.encoder = TextEncoder(
            n_vocab,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            filter_channels=768,  # Valeur par défaut
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            duration_filter_channels= 256,
            duration_kernel_size = 3,
            duration_p_dropout= 0.1
        )

        self.decoder = ConditionalFlowMatching(
            in_channels=2 * out_channels,
            out_channel=out_channels,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # Statistiques des données (utiliser des valeurs par défaut si non fournies)
        if data_statistics is None:
            data_statistics = {"mel_mean": 0.0, "mel_std": 1.0}
        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """
        Génère un spectrogramme mel à partir du texte.
        
        Retourne:
            1. Sorties de l'encodeur
            2. Sorties du décodeur
            3. Alignement généré

        Args:
            x: batch de textes, convertis en tenseur avec des IDs d'embedding de phonèmes.
                shape: (batch_size, max_text_length)
            x_lengths: longueurs des textes dans le batch.
                shape: (batch_size,)
            n_timesteps: nombre de pas à utiliser pour la diffusion inverse dans le décodeur.
            temperature: contrôle la variance de la distribution terminale.
            spks: IDs des locuteurs.
                shape: (batch_size,)
            length_scale: contrôle le rythme de la parole.
                Augmenter la valeur pour ralentir la parole générée et vice versa.

        Returns:
            dict: {
                "encoder_outputs": spectrogramme mel moyen généré par l'encodeur,
                    shape: (batch_size, n_feats, max_mel_length)
                "decoder_outputs": spectrogramme mel raffiné amélioré par le CFM,
                    shape: (batch_size, n_feats, max_mel_length)
                "attn": carte d'alignement entre texte et spectrogramme mel,
                    shape: (batch_size, max_text_length, max_mel_length)
                "mel": spectrogramme mel dénormalisé,
                    shape: (batch_size, n_feats, max_mel_length)
                "mel_lengths": longueurs des spectrogrammes mel,
                    shape: (batch_size,)
                "rtf": facteur temps réel,
                    type: float
            }
        """
        # Pour le calcul RTF
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Obtenir l'embedding du locuteur
            spks = self.spk_emb(spks.long())

        # Obtenir les sorties de l'encodeur `mu_x` et les durées de tokens en log `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Utiliser les durées obtenues `w` pour construire la carte d'alignement `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Aligner le texte encodé et obtenir mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Générer un échantillon en suivant le flux de probabilité
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
        Calcule 3 pertes:
            1. Perte de durée: perte entre les durées de tokens prédites et celles extraites par MAS.
            2. Perte prior: perte entre le spectrogramme mel et les sorties de l'encodeur.
            3. Perte de flow matching: perte entre le spectrogramme mel et les sorties du décodeur.

        Args:
            x: batch de textes, convertis en tenseur avec des IDs d'embedding de phonèmes.
                shape: (batch_size, max_text_length)
            x_lengths: longueurs des textes dans le batch.
                shape: (batch_size,)
            y: batch de spectrogrammes mel correspondants.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths: longueurs des spectrogrammes mel dans le batch.
                shape: (batch_size,)
            out_size: longueur (à la fréquence d'échantillonnage du mel) du segment à couper, sur lequel le décodeur sera entraîné.
                Doit être divisible par 2^{nombre de sous-échantillonnages UNet}. Nécessaire pour augmenter la taille du batch.
            spks: IDs des locuteurs.
                shape: (batch_size,)
            durations: durées précalculées (optionnel)
        """
        if self.n_spks > 1:
            # Obtenir l'embedding du locuteur
            spks = self.spk_emb(spks)

        # Obtenir les sorties de l'encodeur `mu_x` et les durées de tokens en log `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Utiliser MAS pour trouver l'alignement le plus probable `attn` entre texte et spectrogramme mel
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Calculer la perte entre les durées en log prédites et celles obtenues de MAS
        # Référencée comme prior loss dans le papier
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Couper un petit segment du spectrogramme mel pour augmenter la taille du batch
        #   - "Hack" pris de Grad-TTS, dans le cas de Grad-TTS, on ne peut pas entraîner avec batch size 32 sur une GPU 24GB sans cela
        #   - Pas besoin de ce hack pour Matcha-TTS, mais cela fonctionne aussi avec
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

        # Aligner le texte encodé avec le spectrogramme mel et obtenir le segment mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Calculer la perte du décodeur
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn
