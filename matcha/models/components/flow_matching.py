"""
Module de Flow Matching Conditionnel

Implémentation de l'algorithme de flow matching basé sur la résolution d'ODE
pour générer des spectrogrammes mel à partir de bruit.
"""

from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    """Classe de base pour le Flow Matching Conditionnel"""
    
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.feature_dim = n_feats
        self.num_speakers = n_spks
        self.speaker_emb_dim = spk_emb_dim
        self.ode_solver_type = cfm_params.solver
        self.min_noise_std = getattr(cfm_params, "sigma_min", 1e-4)
        self.velocity_estimator = None

    def _initialize_noise(self, shape, temperature=1.0, device=None):
        """Initialise le bruit aléatoire"""
        if device is None:
            device = shape.device if hasattr(shape, 'device') else torch.device('cpu')
        return torch.randn_like(shape) * temperature

    def _create_time_steps(self, num_steps, device):
        """Crée la séquence de pas de temps"""
        return torch.linspace(0, 1, num_steps + 1, device=device)

    @torch.inference_mode()
    def forward(self, encoder_output, output_mask, num_timesteps, temperature=1.0, speaker_emb=None, condition=None):
        """
        Inférence: génère un spectrogramme mel à partir du bruit
        
        Args:
            encoder_output: sortie de l'encodeur, shape (batch_size, n_feats, mel_timesteps)
            output_mask: masque de sortie, shape (batch_size, 1, mel_timesteps)
            num_timesteps: nombre de pas de diffusion
            temperature: facteur d'échelle de température
            speaker_emb: embedding du locuteur, shape (batch_size, spk_emb_dim)
            condition: réservé pour usage futur
            
        Returns:
            Échantillon généré, shape (batch_size, n_feats, mel_timesteps)
        """
        initial_noise = self._initialize_noise(encoder_output, temperature)
        time_sequence = self._create_time_steps(num_timesteps, encoder_output.device)
        return self._solve_ode_euler(
            initial_state=initial_noise,
            time_sequence=time_sequence,
            encoder_output=encoder_output,
            mask=output_mask,
            speaker_emb=speaker_emb,
            condition=condition
        )

    def _solve_ode_euler(self, initial_state, time_sequence, encoder_output, mask, speaker_emb, condition):
        """Résout l'ODE avec la méthode d'Euler à pas fixe"""
        current_state = initial_state
        current_time = time_sequence[0]
        time_step = time_sequence[1] - time_sequence[0]
        intermediate_states = []

        for step_idx in range(1, len(time_sequence)):
            velocity_field = self.velocity_estimator(
                current_state, mask, encoder_output, current_time, speaker_emb, condition
            )
            current_state = current_state + time_step * velocity_field
            current_time = current_time + time_step
            intermediate_states.append(current_state)
            
            if step_idx < len(time_sequence) - 1:
                time_step = time_sequence[step_idx + 1] - current_time

        return intermediate_states[-1]

    def _sample_random_time(self, batch_size, device, dtype):
        """Échantillonne un temps aléatoire t ∈ [0, 1]"""
        return torch.rand([batch_size, 1, 1], device=device, dtype=dtype)

    def _build_conditional_path(self, noise, target, time):
        """Construit le chemin conditionnel y_t = (1 - (1 - σ_min) * t) * z + t * x_1"""
        noise_coeff = 1 - (1 - self.min_noise_std) * time
        target_coeff = time
        return noise_coeff * noise + target_coeff * target

    def _compute_velocity_target(self, target, noise):
        """Calcule la cible du champ de vitesse u = x_1 - (1 - σ_min) * z"""
        return target - (1 - self.min_noise_std) * noise

    def compute_loss(self, target_sample, target_mask, encoder_output, speaker_emb=None, condition=None):
        """
        Calcule la perte de flow matching conditionnel
        
        Args:
            target_sample: échantillon cible, shape (batch_size, n_feats, mel_timesteps)
            target_mask: masque cible, shape (batch_size, 1, mel_timesteps)
            encoder_output: sortie de l'encodeur, shape (batch_size, n_feats, mel_timesteps)
            speaker_emb: embedding du locuteur, shape (batch_size, spk_emb_dim)
            
        Returns:
            loss: perte de flow matching (scalaire)
            path_point: point sur le chemin y_t
        """
        batch_size = encoder_output.shape[0]
        random_time = self._sample_random_time(batch_size, encoder_output.device, encoder_output.dtype)
        initial_noise = torch.randn_like(target_sample)
        path_point = self._build_conditional_path(initial_noise, target_sample, random_time)
        velocity_target = self._compute_velocity_target(target_sample, initial_noise)
        predicted_velocity = self.velocity_estimator(
            path_point, target_mask, encoder_output, random_time.squeeze(), speaker_emb
        )
        loss = F.mse_loss(predicted_velocity, velocity_target, reduction="sum") / (
            torch.sum(target_mask) * velocity_target.shape[1]
        )
        return loss, path_point


class CFM(BASECFM):
    """Implémentation du Flow Matching Conditionnel"""
    
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        estimator_input_channels = in_channels
        if n_spks > 1:
            estimator_input_channels += spk_emb_dim
        
        self.velocity_estimator = Decoder(
            in_channels=estimator_input_channels,
            out_channels=out_channel,
            **decoder_params
        )
