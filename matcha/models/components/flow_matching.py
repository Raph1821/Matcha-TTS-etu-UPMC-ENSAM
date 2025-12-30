from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder


class BaseConditionalFlowMatching(nn.Module, ABC):
    """
    Classe de base pour le Flow Matching (équivalent de BASECFM).
    
    Contient les algorithmes de génération (ODE solver) et de calcul de loss.
    Le decoder (estimator) doit être défini dans les classes filles.
    """
    
    def __init__(self, 
                 n_feats, 
                 cfm_params):
        """
        Args:
            n_feats (int): Nombre de features mel (utilisé dans matcha_tts.py)
            cfm_params: Paramètres CFM (doit contenir solver, sigma_min)
        """
        super().__init__()
        self.n_feats = n_feats
        self.solver = cfm_params.solver if hasattr(cfm_params, 'solver') else 'euler'
        self.sigma_min = cfm_params.sigma_min if hasattr(cfm_params, 'sigma_min') else 1e-4
        
        # Le decoder(estimator) sera défini dans les classes filles
        self.estimator = None
    
    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, cond=None):
        """
        Génère un mel-spectrogramme à partir du bruit (INFERENCE).
        
        Args:
            mu (torch.Tensor): Condition (sortie encoder), shape (B, C, T)
            mask (torch.Tensor): Output Masque, shape (B, 1, T)
            n_timesteps (int): Nombre de pas de diffusion (plus = meilleur mais plus lent)
            temperature (float): Contrôle la variance (1.0 = normal, >1 = plus varié)
            cond: Not used but kept for future purposes
        
        Returns:
            sample: torch.Tensor: Mel-spectrogramme généré, shape (batch_size, n_feats, mel_timesteps)
        """
        # 1. Commencer avec du bruit gaussien pur
        z = torch.randn_like(mu) * temperature
        
        # 2. Créer les timesteps de 0 à 1
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        
        # 3. Résoudre l'ODE avec la méthode d'Euler
        return self.solve_ode_euler(z, t_span, mu, mask, cond)
    
    def solve_ode_euler(self, x, t_span, mu, mask, cond):
        """
        Résout l'ODE avec la méthode d'Euler.
        
        L'ODE à résoudre: dx/dt = v(x, t, mu)
        où v est le champ de vitesse prédit par le decoder.
        
        Args:
            x: État initial (bruit aléatoire), shape (B, C, T)
            t_span: Timesteps [0, dt, 2dt, ..., 1], shape (n_timesteps+1,)
            mu (torch.Tensor): Condition (encoder output), shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): Output Masque, shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        
        Returns:
            torch.Tensor: État final (mel-spectrogramme)
        """
        t = t_span[0]  # t = 0
        dt = t_span[1] - t_span[0]  # Pas de temps initial
        batch_size = x.shape[0]
        
        # Intégration d'Euler: x_{t+dt} = x_t + dt * v(x_t, t)
        for step in range(1, len(t_span)):
            # Prédire le champ de vitesse au temps t
            # Decoder.forward 期望 t 的形状为 (batch_size,)，所以需要将标量转换为张量
            t_tensor = torch.full((batch_size,), t, device=x.device, dtype=x.dtype)
            dphi_dt = self.estimator(x, mask, mu, t_tensor, cond)
            velocity = dphi_dt

            # Mise à jour d'Euler
            x = x + dt * velocity
            
            # Avancer le temps
            t = t + dt
            
            # Recalculer dt pour le prochain pas (peut varier)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        
        return x
    
    def compute_loss(self, x1, mask, mu, cond=None):
        """
        Calcule la loss de diffusion du flow matching (ENTRAÎNEMENT).
        
        Utilise la formulation Conditional Flow Matching (CFM):
        - Échantillonne un timestep aléatoire t ~ Uniform[0, 1]
        - Interpole entre bruit z et target: phi_t = (1-(1-self.sigma_min)*t)*z + t*x1
        - Prédit le champ de vitesse: u_target = x1 - (1 - self.sigma_min) * z)
        - Minimise: ||decoder(phi_t, t) - u||²
        
        Args:
            x1 (torch.Tensor): target, Mel-spectrogramme cible, shape (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): Target Masque, shape (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): Condition (encoder output), shape (batch_size, n_feats, mel_timesteps)
            cond: Not used but kept for future purposes
        
        Returns:
            loss (torch.Tensor): Scalaire, la loss moyenne
            y (torch.Tensor): État interpolé (pour debug/visualisation)
        """
        batch_size = mu.shape[0]
        
        # 1. Échantillonner un timestep aléatoire pour chaque exemple
        t = torch.rand([batch_size, 1, 1], device=mu.device, dtype=mu.dtype)
        
        # 2. Échantillonner du bruit gaussien p(x_0)
        z = torch.randn_like(x1)
        
        # 3. Interpoler entre bruit (t=0) et target (t=1)
        #    phi_t = (1 - (1-sigma_min)*t) * z + t * x1
        #    Le sigma_min évite d'avoir exactement du bruit pur à t=0
        phi_t = (1 - (1 - self.sigma_min) * t) * z + t * x1
        
        # 4. Le champ de vitesse cible (direction optimale)
        #    u = d(phi_t)/dt = x1 - (1-sigma_min)*z
        u_target = x1 - (1 - self.sigma_min) * z
        
        # 5. Prédire le champ de vitesse avec le estimator
        u_pred = self.estimator(phi_t, mask, mu, t.squeeze(), cond)
        
        # 6. Calculer la MSE loss (normalisée par le masque)
        loss = F.mse_loss(u_pred, u_target, reduction="sum")
        loss = loss / (torch.sum(mask) * u_target.shape[1])
        
        return loss, phi_t


class ConditionalFlowMatching(BaseConditionalFlowMatching):
    """
    Implémentation complète du Flow Matching (équivalent de CFM).
    
    Hérite de BaseFlowMatching et crée le Decoder automatiquement.
    Compatible avec matcha_tts.py qui utilise CFM.
    """
    
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params):
        """
        Args:
            in_channels (int): Nombre de canaux d'entrée (= n_feats dans l'original)
            out_channel (int): Nombre de canaux de sortie
            cfm_params: Objet contenant solver, sigma_min, etc.
            decoder_params (dict): Paramètres pour le Decoder (channels, dropout, etc.)
        """
        # Initialiser la classe de base avec n_feats et cfm_params
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params
        )
        
        # Créer le Decoder (estimator) - le réseau qui prédit le champ de vitesse
        self.estimator = Decoder(
            in_channels=in_channels,
            out_channels=out_channel,
            **decoder_params
        )


# Alias pour compatibilité avec matcha_tts.py
CFM = ConditionalFlowMatching