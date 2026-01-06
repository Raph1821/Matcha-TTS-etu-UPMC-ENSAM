import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 
import os 
from pathlib import Path

from matcha.data_management.ljspeechDataset import LJSpeechDataset
from matcha.utils.data_download.ljspeech import process_csv

class LJSpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        train_txt = self.data_dir / "train.txt"
        val_txt = self.data_dir / "val.txt"
        
        if not train_txt.exists() or not val_txt.exists():
            metadata_found = False
            
            if (self.data_dir / "metadata.csv").exists():
                metadata_found = True
            else:
                for subdir in self.data_dir.iterdir():
                    if subdir.is_dir() and "ljspeech" in subdir.name.lower():
                        if (subdir / "metadata.csv").exists():
                            metadata_found = True
                            break
            
            if metadata_found:
                print(f"Génération de train.txt et val.txt depuis metadata.csv dans {self.data_dir}...")
                try:
                    process_csv(self.data_dir, output_dir=self.data_dir)
                    if not train_txt.exists() or not val_txt.exists():
                        raise RuntimeError(
                            f"Les fichiers train.txt et val.txt n'ont pas été générés dans {self.data_dir}. "
                            f"Vérifiez les permissions d'écriture."
                        )
                    print(f"✓ train.txt et val.txt générés avec succès dans {self.data_dir}")
                except Exception as e:
                    raise RuntimeError(
                        f"Erreur lors de la génération de train.txt/val.txt: {e}\n"
                        f"Vérifiez que metadata.csv existe et est accessible dans {self.data_dir}"
                    ) from e
            else:
                raise FileNotFoundError(
                    f"Les fichiers train.txt et val.txt n'existent pas dans {self.data_dir}.\n"
                    f"metadata.csv introuvable dans {self.data_dir} ou ses sous-répertoires.\n"
                    f"Veuillez d'abord télécharger le dataset LJSpeech et exécuter:\n"
                    f"  python -m matcha.utils.data_download.ljspeech {self.data_dir}\n"
                    f"Ou assurez-vous que metadata.csv existe dans {self.data_dir} ou un sous-répertoire contenant 'ljspeech'"
                )
        
        self.train_ds = LJSpeechDataset(str(train_txt))
        self.val_ds = LJSpeechDataset(str(val_txt))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            collate_fn=self.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            collate_fn=self.collate
        )
    
    def collate(self, batch):
        """
        Organise les données en batch en ajoutant du padding.
        """
        # Extraire les éléments du batch
        # x: IDs de texte, y: Spectrogrammes de Mel
        x = [item["x"] for item in batch]
        y = [item["y"] for item in batch]
        x_lengths = torch.tensor([item["x_lengths"] for item in batch])
        y_lengths = torch.tensor([item["y_lengths"] for item in batch])

        # Padding du texte (x)
        # batch_first=True donne une forme [Batch, Longueur_max]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)

        # Padding de l'audio (y)
        # y est [Canaux_Mel, Temps]. On doit padder sur la dimension Temps (index 1).
        # On transpose pour utiliser pad_sequence, puis on revient à la forme initiale.
        y_padded = pad_sequence([item.transpose(0, 1) for item in y], 
                                batch_first=True, padding_value=0).transpose(1, 2)

        return {
            "x": x_padded,
            "x_lengths": x_lengths,
            "y": y_padded,
            "y_lengths": y_lengths
        }
    

    def get_data_statistics(self):
        """
        Calcule la moyenne et l'écart-type des spectrogrammes Mel sur le set d'entraînement.
        """
        
        # On s'assure que le dataset est chargé
        if not hasattr(self, 'train_ds'):
            self.setup()

        mel_sum = 0.0
        mel_sq_sum = 0.0
        total_frames = 0
        
        # Pour éviter de charger tout en RAM, on itère sur une version DataLoader simple
        # ou directement sur le dataset si possible.
        temp_loader = DataLoader(
            self.train_ds, batch_size=32, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate
        )

        for batch in temp_loader:
            # batch['y'] shape: [B, n_feats, T]
            mel = batch['y']
            mask = batch['x_lengths'] # Attention: x_lengths est la longueur texte, il nous faut un masque audio
            # On réutilise y_lengths qui est dans le collate
            y_lengths = batch['y_lengths']
            
            # On ne prend en compte que les parties non-paddées
            for i in range(mel.shape[0]):
                length = y_lengths[i]
                valid_mel = mel[i, :, :length] # [n_feats, length]
                
                mel_sum += valid_mel.sum().item()
                mel_sq_sum += (valid_mel ** 2).sum().item()
                total_frames += (length * valid_mel.shape[0]).item() # n_feats * time

        mel_mean = mel_sum / total_frames
        mel_std = (mel_sq_sum / total_frames - mel_mean ** 2) ** 0.5
    
        return {"mel_mean": mel_mean, "mel_std": mel_std}