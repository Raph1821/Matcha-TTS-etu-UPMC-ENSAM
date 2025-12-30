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
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Le répertoire de données {self.data_dir} n'existe pas.\n"
                f"Veuillez d'abord télécharger le dataset LJSpeech:\n"
                f"  python -m matcha.utils.data_download.ljspeech {self.data_dir}"
            )
        
        metadata_path = None
        if (self.data_dir / "metadata.csv").exists():
            metadata_path = self.data_dir
        elif self.data_dir.is_dir():
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir() and "ljspeech" in subdir.name.lower():
                    if (subdir / "metadata.csv").exists():
                        metadata_path = subdir
                        break
        
        if metadata_path is None:
            raise FileNotFoundError(
                f"metadata.csv introuvable dans {self.data_dir} ou ses sous-répertoires.\n"
                f"Veuillez d'abord télécharger le dataset LJSpeech:\n"
                f"  python -m matcha.utils.data_download.ljspeech {self.data_dir}"
            )
        
        train_txt = metadata_path / "train.txt"
        val_txt = metadata_path / "val.txt"
        
        if not train_txt.exists() or not val_txt.exists():
            print(f"Génération de train.txt et val.txt depuis metadata.csv dans {metadata_path}...")
            try:
                process_csv(self.data_dir, output_dir=metadata_path)
                if not train_txt.exists() or not val_txt.exists():
                    raise RuntimeError(
                        f"Les fichiers train.txt et val.txt n'ont pas été générés dans {metadata_path}. "
                        f"Vérifiez les permissions d'écriture."
                    )
                print(f"✓ train.txt et val.txt générés avec succès dans {metadata_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Erreur lors de la génération de train.txt/val.txt: {e}\n"
                    f"Vérifiez que metadata.csv existe et est accessible dans {self.data_dir}"
                ) from e
        
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