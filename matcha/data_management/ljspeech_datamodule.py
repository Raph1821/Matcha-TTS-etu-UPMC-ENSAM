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
        print(f"\n{'='*60}")
        print(f"LJSpeechDataModule.setup() - Initialisation")
        print(f"{'='*60}")
        print(f"data_dir: {self.data_dir}")
        print(f"data_dir existe: {self.data_dir.exists()}")
        print(f"data_dir est un répertoire: {self.data_dir.is_dir() if self.data_dir.exists() else 'N/A'}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Le répertoire de données {self.data_dir} n'existe pas.\n"
                f"Veuillez d'abord télécharger le dataset LJSpeech:\n"
                f"  python -m matcha.utils.data_download.ljspeech {self.data_dir}"
            )
        
        print(f"\nRecherche de metadata.csv...")
        print(f"  - Vérification dans {self.data_dir}/metadata.csv: {(self.data_dir / 'metadata.csv').exists()}")
        
        metadata_path = None
        if (self.data_dir / "metadata.csv").exists():
            metadata_path = self.data_dir
            print(f"  ✓ metadata.csv trouvé directement dans {self.data_dir}")
        elif self.data_dir.is_dir():
            print(f"  - Recherche dans les sous-répertoires de {self.data_dir}...")
            subdirs_found = []
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    subdirs_found.append(subdir.name)
                    if "ljspeech" in subdir.name.lower():
                        csv_exists = (subdir / "metadata.csv").exists()
                        print(f"    - {subdir.name}/: contient 'ljspeech'={('ljspeech' in subdir.name.lower())}, metadata.csv existe={csv_exists}")
                        if csv_exists:
                            metadata_path = subdir
                            print(f"  ✓ metadata.csv trouvé dans {subdir}")
                            break
            if not metadata_path:
                print(f"  Sous-répertoires trouvés: {subdirs_found}")
        
        if metadata_path is None:
            print(f"\n❌ ERREUR: metadata.csv introuvable")
            raise FileNotFoundError(
                f"metadata.csv introuvable dans {self.data_dir} ou ses sous-répertoires.\n"
                f"Veuillez d'abord télécharger le dataset LJSpeech:\n"
                f"  python -m matcha.utils.data_download.ljspeech {self.data_dir}"
            )
        
        print(f"\nRépertoire de metadata.csv: {metadata_path}")
        print(f"  - metadata.csv: {(metadata_path / 'metadata.csv').exists()}")
        print(f"  - wavs/: {(metadata_path / 'wavs').exists()}")
        
        train_txt = metadata_path / "train.txt"
        val_txt = metadata_path / "val.txt"
        
        print(f"\nVérification des fichiers train.txt et val.txt...")
        print(f"  - train.txt: {train_txt} (existe: {train_txt.exists()})")
        print(f"  - val.txt: {val_txt} (existe: {val_txt.exists()})")
        
        if not train_txt.exists() or not val_txt.exists():
            print(f"\n{'='*60}")
            print(f"Génération de train.txt et val.txt")
            print(f"{'='*60}")
            print(f"  Répertoire source (ljpath): {self.data_dir}")
            print(f"  Répertoire de sortie (output_dir): {metadata_path}")
            try:
                process_csv(self.data_dir, output_dir=metadata_path)
                
                print(f"\nVérification après génération...")
                print(f"  - train.txt: {train_txt} (existe: {train_txt.exists()})")
                if train_txt.exists():
                    print(f"    Taille: {train_txt.stat().st_size} octets")
                print(f"  - val.txt: {val_txt} (existe: {val_txt.exists()})")
                if val_txt.exists():
                    print(f"    Taille: {val_txt.stat().st_size} octets")
                
                if not train_txt.exists() or not val_txt.exists():
                    print(f"\n❌ ERREUR: Les fichiers n'ont pas été générés")
                    print(f"  Permissions du répertoire {metadata_path}:")
                    print(f"    - Lecture: {os.access(metadata_path, os.R_OK)}")
                    print(f"    - Écriture: {os.access(metadata_path, os.W_OK)}")
                    raise RuntimeError(
                        f"Les fichiers train.txt et val.txt n'ont pas été générés dans {metadata_path}. "
                        f"Vérifiez les permissions d'écriture."
                    )
                print(f"\n✓ train.txt et val.txt générés avec succès dans {metadata_path}")
            except Exception as e:
                print(f"\n❌ ERREUR lors de la génération:")
                print(f"  Type: {type(e).__name__}")
                print(f"  Message: {str(e)}")
                import traceback
                print(f"  Traceback:")
                traceback.print_exc()
                raise RuntimeError(
                    f"Erreur lors de la génération de train.txt/val.txt: {e}\n"
                    f"Vérifiez que metadata.csv existe et est accessible dans {self.data_dir}"
                ) from e
        else:
            print(f"✓ train.txt et val.txt existent déjà")
        
        print(f"\nChargement des datasets...")
        print(f"  - train.txt: {train_txt}")
        print(f"  - val.txt: {val_txt}")
        self.train_ds = LJSpeechDataset(str(train_txt))
        self.val_ds = LJSpeechDataset(str(val_txt))
        print(f"✓ Datasets chargés: {len(self.train_ds)} échantillons train, {len(self.val_ds)} échantillons val")
        print(f"{'='*60}\n")

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