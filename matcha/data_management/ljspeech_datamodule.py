import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 
import os 

from matcha.data_management.ljspeechDataset import LJSpeechDataset

class LJSpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        self.train_ds = LJSpeechDataset(os.path.join(self.data_dir, "train.txt"))
        self.val_ds = LJSpeechDataset(os.path.join(self.data_dir, "val.txt"))

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
        # 1. Extraire les éléments du batch
        # x: IDs de texte, y: Spectrogrammes de Mel
        x = [item["x"] for item in batch]
        y = [item["y"] for item in batch]
        x_lengths = torch.tensor([item["x_lengths"] for item in batch])
        y_lengths = torch.tensor([item["y_lengths"] for item in batch])

        # 2. Padding du texte (x)
        # batch_first=True donne une forme [Batch, Longueur_max]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)

        # 3. Padding de l'audio (y)
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