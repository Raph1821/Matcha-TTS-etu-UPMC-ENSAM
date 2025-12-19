import numpy as np 
from scipy.io.wavfile import read
import torch
from librosa.filters import mel as librosa_mel_fn



MAX_WAV_VALUE = 32768.0



def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


# Réduire l'écart entre les sons de très faible intensité et les sons de forte intensité
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Compresse l'amplitude pour le modèle"""
    return torch.log(torch.clamp(x, min=clip_val) * C)

# Augmenter l'écart entre les sons de très faible intensité et les sons de forte intensité
def spectral_normalize_torch(magnitudes):
    """Applique la normalisation dynamique sur les magnitudes"""
    return dynamic_range_compression_torch(magnitudes)






class MelSpectrogram:
    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        
        # Pré-calcul de la base Mel (sur CPU par défaut)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = torch.from_numpy(mel).float()
        self.hann_window = torch.hann_window(win_size)
    
    def _apply_stft(self, y):
        """Étape 1 : Transformée de Fourier à court terme"""
        # Padding réflexif pour éviter les artefacts aux bords
        pad_size = int((self.n_fft - self.hop_size) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)
        
        spec = torch.stft(
            y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=self.hann_window.to(y.device), center=self.center,
            pad_mode="reflect", normalized=False, onesided=True, return_complex=True
        )
        # Calcul de la magnitude spectrale
        return torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    
    def __call__(self, y):
        """Étape 2 : Projection Mel et Normalisation"""
        # 1. Obtenir les magnitudes
        magnitudes = self._apply_stft(y)
        
        # 2. Projection sur l'échelle de Mel
        mel_output = torch.matmul(self.mel_basis.to(y.device), magnitudes)
        
        # 3. Normalisation finale (Log-compression)
        return spectral_normalize_torch(mel_output)



def load_and_process_audio(file_path, mel_processor):
    """Charge le wav et le transforme en Mel en une ligne"""
    sampling_rate, data = read(file_path) #
    # Conversion en tenseur et normalisation de l'amplitude 16-bit
    y = torch.FloatTensor(data.astype(np.float32)) / MAX_WAV_VALUE
    y = y.unsqueeze(0) # Ajout de la dimension batch
    
    return mel_processor(y)