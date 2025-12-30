#!/usr/bin/env python

import argparse
import random
import sys
import tempfile
from pathlib import Path

from torch.hub import download_url_to_file

from matcha.utils.data_download.utils import _extract_tar

#- Téléchargent les données depuis une URL au format tar ou zip 
#- Extraient les archives dans un répertoire de sortie val ou train 
#- Génèrent des fichiers .txt au format <chemin_wav>|<transcription>.
#- Utilisent argparse pour organiser les répertoires de sortie 


URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

INFO_PAGE = "https://keithito.com/LJ-Speech-Dataset/"

LICENCE = "Public domain (LibriVox copyright disclaimer)"

CITATION = """
@misc{ljspeech17,
  author       = {Keith Ito and Linda Johnson},
  title        = {The LJ Speech Dataset},
  howpublished = {\\url{https://keithito.com/LJ-Speech-Dataset/}},
  year         = 2017
}
"""


def decision():
    return random.random() < 0.98


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data",
        help="Place to store the converted data (subdirectory LJSpeech-1.1 will be created)",
    )

    return parser.parse_args()


def process_csv(ljpath: Path, output_dir: Path = None):
    print(f"Recherche de metadata.csv dans {ljpath}...")
    
    basepath = None
    if (ljpath / "metadata.csv").exists():
        basepath = ljpath
        print(f"✓ metadata.csv trouvé dans {ljpath}")
    else:
        for subdir in ljpath.iterdir():
            if subdir.is_dir() and "ljspeech" in subdir.name.lower():
                if (subdir / "metadata.csv").exists():
                    basepath = subdir
                    print(f"✓ metadata.csv trouvé dans {subdir}")
                    break
    
    if basepath is None:
        raise FileNotFoundError(
            f"metadata.csv introuvable dans {ljpath} ou ses sous-répertoires. "
            f"Vérifiez que le dataset LJSpeech est correctement téléchargé."
        )
    
    csvpath = basepath / "metadata.csv"
    wavpath = basepath / "wavs"
    
    if output_dir is None:
        output_dir = ljpath
    
    print(f"Génération de train.txt et val.txt depuis {csvpath}...")
    print(f"  Fichiers de sortie: {output_dir / 'train.txt'}, {output_dir / 'val.txt'}")
    
    with (
        open(csvpath, encoding="utf-8") as csvf,
        open(output_dir / "train.txt", "w", encoding="utf-8") as tf,
        open(output_dir / "val.txt", "w", encoding="utf-8") as vf,
    ):
        lines = csvf.readlines()
        total = len(lines)
        train_count = 0
        val_count = 0
        
        for i, line in enumerate(lines, 1):
            if i % 1000 == 0 or i == total:
                print(f"  Traitement: {i}/{total} lignes ({100*i//total}%)", end='\r')
                sys.stdout.flush()
            
            line = line.strip()
            parts = line.split("|")
            wavfile = str(wavpath / f"{parts[0]}.wav")
            if decision():
                tf.write(f"{wavfile}|{parts[1]}\n")
                train_count += 1
            else:
                vf.write(f"{wavfile}|{parts[1]}\n")
                val_count += 1
        
        print(f"\n✓ Génération terminée: {train_count} échantillons train, {val_count} échantillons val")


def main():
    args = get_args()

    print("=" * 60)
    print("Téléchargement et préparation du dataset LJSpeech")
    print("=" * 60)
    
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()
            print(f"✓ Répertoire de sauvegarde créé: {save_dir}")

    outpath = Path(args.output_dir)
    if not outpath.is_dir():
        outpath.mkdir()
        print(f"✓ Répertoire de sortie créé: {outpath}")

    if save_dir:
        tarname = URL.rsplit("/", maxsplit=1)[-1]
        tarfile = save_dir / tarname
        if not tarfile.exists():
            print(f"\n📥 Téléchargement de {tarname}...")
            print(f"   URL: {URL}")
            print(f"   Destination: {tarfile}")
            download_url_to_file(URL, str(tarfile), progress=True)
            print(f"✓ Téléchargement terminé: {tarfile.stat().st_size / (1024**3):.2f} GB")
        else:
            print(f"\n✓ Fichier déjà téléchargé: {tarfile}")
        
        print(f"\n📦 Extraction de l'archive vers {outpath}...")
        print("   (Cela peut prendre plusieurs minutes, veuillez patienter...)")
        _extract_tar(tarfile, outpath)
        print("✓ Extraction terminée")
        
        print(f"\n📝 Génération des fichiers train.txt et val.txt...")
        process_csv(outpath)
    else:
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=True) as zf:
            print(f"\n📥 Téléchargement temporaire de {URL}...")
            download_url_to_file(URL, zf.name, progress=True)
            print(f"✓ Téléchargement terminé")
            
            print(f"\n📦 Extraction de l'archive vers {outpath}...")
            print("   (Cela peut prendre plusieurs minutes, veuillez patienter...)")
            _extract_tar(zf.name, outpath)
            print("✓ Extraction terminée")
            
            print(f"\n📝 Génération des fichiers train.txt et val.txt...")
            process_csv(outpath)
    
    print("\n" + "=" * 60)
    print("✓ Préparation du dataset terminée avec succès!")
    print(f"  Données disponibles dans: {outpath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
