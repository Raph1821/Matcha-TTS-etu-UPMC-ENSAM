import os
import torch
from torch.utils.data import Dataset
# On importe la nouvelle fonction text_to_sequence
from matcha.text_to_ID.text_to_sequence import text_to_sequence
from matcha.text_to_ID.cleaners import english_cleaners
from matcha.utils.audio_process import MelSpectrogram, load_and_process_audio

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f.readlines()]
        
        # Le dictionnaire symbol_to_id est maintenant géré dans text_to_sequence
        
        self.mel_proc = MelSpectrogram(n_fft=1024, num_mels=80, sampling_rate=22050, 
                                       hop_size=256, win_size=1024, fmin=0, fmax=8000)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_path, raw_text = self.metadata[idx]

        # 1. Nettoyage basique (optionnel si phonemizer gère bien le brut)
        clean_text = english_cleaners(raw_text)
        
        # 2. Phonémisation et Conversion en IDs
        #  le texte devient des phonèmes puis des IDs
        sequence = text_to_sequence(clean_text, ["english_cleaners"])
        text_ids = torch.tensor(sequence, dtype=torch.long)

        # 3. Traitement Audio
        mel = load_and_process_audio(wav_path, self.mel_proc)

        return {
            "x": text_ids, 
            "x_lengths": torch.tensor(len(text_ids)),
            "y": mel.squeeze(0), 
            "y_lengths": torch.tensor(mel.shape[-1])
        }
            "y_lengths": torch.tensor(mel.shape[-1])
        }
