# taken from https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/utils.py
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
# Licence: BSD 2-Clause
# pylint: disable=C0123

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Union
import torch

_LG = logging.getLogger(__name__)


def _extract_tar(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if isinstance(from_path, Path):
        from_path = str(from_path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, "r:*") as tar:
        members = tar.getmembers()
        files = []
        for file_ in members:
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    _LG.info("%s already extracted.", file_path)
                    if not overwrite:
                        continue
            tar.extract(file_, to_path)
        return files


def _extract_zip(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if type(from_path) is Path:
        from_path = str(Path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        files = zfile.namelist()
        for file_ in files:
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                _LG.info("%s already extracted.", file_path)
                if not overwrite:
                    continue
            zfile.extract(file_, to_path)
    return files


def download_pretrained_model(
    url: str, 
    destination: str, 
    force_download: bool = False
):
    """
    Télécharge un modèle pré-entraîné depuis une URL (GitHub Release, etc.)
    """
    if os.path.exists(destination) and not force_download:
        print(f" Le modèle existe déjà à : {destination}")
        return

    print(f"⬇  Téléchargement du modèle depuis {url}...")
    try:
        # Crée le dossier parent si nécessaire
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Télécharge avec barre de progression
        torch.hub.download_url_to_file(url, destination, progress=True)
        print(" Téléchargement terminé !")
    except Exception as e:
        print(f" Erreur lors du téléchargement : {e}")
        # Nettoie le fichier partiel si échec
        if os.path.exists(destination):
            os.remove(destination)
