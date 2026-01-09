import os
import torch
from torch.utils.data import Dataset
from matcha.text_to_ID.cleaners import english_cleaners
from matcha.text_to_ID.symbols import symbols
from matcha.utils.audio_process import MelSpectrogram, load_and_process_audio

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path):
        # On lit le fichier train.txt ou val.txt (format: chemin_wav|texte)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f.readlines()]
        
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.mel_proc = MelSpectrogram(n_fft=1024, num_mels=80, sampling_rate=22050, 
                                       hop_size=256, win_size=1024, fmin=0, fmax=8000)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_path, raw_text = self.metadata[idx]

        # 1. Traitement du Texte
        clean_text = english_cleaners(raw_text)
        text_ids = torch.tensor([self.symbol_to_id[s] for s in clean_text if s in self.symbol_to_id], dtype=torch.long)

        # 2. Traitement Audio (wav_path est déjà le chemin complet grâce à ljspeech.py)
        mel = load_and_process_audio(wav_path, self.mel_proc).detach()

        return {
            "x": text_ids, 
            "x_lengths": torch.tensor(len(text_ids)),
            "y": mel.squeeze(0), 
            "y_lengths": torch.tensor(mel.shape[-1])
        }
