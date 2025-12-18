import torch
import matplotlib.pyplot as plt
from audio_process import MelSpectrogram, load_and_process_audio
from scipy.io.wavfile import read
import utils as utils

# 1. Initialisation avec les paramètres de Matcha-TTS
mel_proc = MelSpectrogram(
    n_fft=1024, num_mels=80, sampling_rate=22050, 
    hop_size=256, win_size=1024, fmin=0, fmax=8000
)

# 2. Chargement d'un fichier réel de votre dossier data
path = "data\LJSpeech-1.1\wavs\LJ001-0010.wav"

# 3. Traitement audio pour obtenir le spectrogramme Mel
mel_spectrogram = load_and_process_audio(path, mel_proc)

# 4. Affichage du spectrogramme Mel
utils.plot_spectrogram(mel_spectrogram, title="Mel Spectrogram of LJ001-0010.wav")

