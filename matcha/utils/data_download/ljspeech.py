#!/usr/bin/env python

import argparse
import os
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
    print(f"\nprocess_csv() appelé avec:")
    print(f"  ljpath: {ljpath} (type: {type(ljpath)})")
    print(f"  output_dir: {output_dir} (type: {type(output_dir)})")
    
    if not isinstance(ljpath, Path):
        ljpath = Path(ljpath)
        print(f"  ljpath converti en Path: {ljpath}")
    
    if not ljpath.exists():
        print(f"  ❌ ERREUR: Le répertoire {ljpath} n'existe pas")
        raise FileNotFoundError(f"Le répertoire {ljpath} n'existe pas.")
    
    print(f"  ljpath existe: {ljpath.exists()}")
    print(f"  ljpath est un répertoire: {ljpath.is_dir()}")
    
    print(f"\nRecherche de metadata.csv dans {ljpath}...")
    
    basepath = None
    if (ljpath / "metadata.csv").exists():
        basepath = ljpath
        print(f"  ✓ metadata.csv trouvé directement dans {ljpath}")
    else:
        print(f"  metadata.csv non trouvé dans {ljpath}, recherche dans les sous-répertoires...")
        if ljpath.is_dir():
            for subdir in ljpath.iterdir():
                if subdir.is_dir() and "ljspeech" in subdir.name.lower():
                    print(f"    - Vérification de {subdir}...")
                    if (subdir / "metadata.csv").exists():
                        basepath = subdir
                        print(f"  ✓ metadata.csv trouvé dans {subdir}")
                        break
                    else:
                        print(f"      metadata.csv non trouvé dans {subdir}")
    
    if basepath is None:
        print(f"  ❌ ERREUR: metadata.csv introuvable")
        raise FileNotFoundError(
            f"metadata.csv introuvable dans {ljpath} ou ses sous-répertoires. "
            f"Vérifiez que le dataset LJSpeech est correctement téléchargé."
        )
    
    csvpath = basepath / "metadata.csv"
    wavpath = basepath / "wavs"
    
    print(f"\nChemins déterminés:")
    print(f"  basepath: {basepath}")
    print(f"  csvpath: {csvpath} (existe: {csvpath.exists()})")
    print(f"  wavpath: {wavpath} (existe: {wavpath.exists()})")
    
    if output_dir is None:
        output_dir = ljpath
        print(f"  output_dir non spécifié, utilisation de ljpath: {output_dir}")
    else:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
            print(f"  output_dir converti en Path: {output_dir}")
        print(f"  output_dir: {output_dir}")
        print(f"  output_dir existe: {output_dir.exists()}")
        if not output_dir.exists():
            print(f"  Création du répertoire {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  output_dir créé/vérifié: {output_dir.exists()}")
        print(f"  Permissions: lecture={os.access(output_dir, os.R_OK)}, écriture={os.access(output_dir, os.W_OK)}")
    
    train_txt_path = output_dir / "train.txt"
    val_txt_path = output_dir / "val.txt"
    
    print(f"\nGénération de train.txt et val.txt...")
    print(f"  Source: {csvpath}")
    print(f"  Destination train.txt: {train_txt_path}")
    print(f"  Destination val.txt: {val_txt_path}")
    
    try:
        with (
            open(csvpath, encoding="utf-8") as csvf,
            open(train_txt_path, "w", encoding="utf-8") as tf,
            open(val_txt_path, "w", encoding="utf-8") as vf,
        ):
            print(f"  Fichiers ouverts avec succès")
            lines = csvf.readlines()
            total = len(lines)
            print(f"  Total de lignes dans metadata.csv: {total}")
            train_count = 0
            val_count = 0
            
            for i, line in enumerate(lines, 1):
                if i % 1000 == 0 or i == total:
                    print(f"  Traitement: {i}/{total} lignes ({100*i//total}%)", end='\r')
                    sys.stdout.flush()
                
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    print(f"  ⚠ Ligne {i} ignorée (format invalide): {line[:50]}")
                    continue
                wavfile = str(wavpath / f"{parts[0]}.wav")
                if decision():
                    tf.write(f"{wavfile}|{parts[1]}\n")
                    train_count += 1
                else:
                    vf.write(f"{wavfile}|{parts[1]}\n")
                    val_count += 1
            
            print(f"\n  Génération terminée: {train_count} échantillons train, {val_count} échantillons val")
        
        print(f"\nVérification des fichiers générés:")
        print(f"  train.txt: {train_txt_path} (existe: {train_txt_path.exists()})")
        if train_txt_path.exists():
            print(f"    Taille: {train_txt_path.stat().st_size} octets")
        print(f"  val.txt: {val_txt_path} (existe: {val_txt_path.exists()})")
        if val_txt_path.exists():
            print(f"    Taille: {val_txt_path.stat().st_size} octets")
        
    except Exception as e:
        print(f"\n  ❌ ERREUR lors de l'écriture des fichiers:")
        print(f"    Type: {type(e).__name__}")
        print(f"    Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


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
        print("   (Cela peut prendre plusieurs minutes, veuillez patienter...)\n")
        _extract_tar(tarfile, outpath)
        print("\n✓ Extraction terminée")
    else:
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=True) as zf:
            print(f"\n📥 Téléchargement temporaire de {URL}...")
            download_url_to_file(URL, zf.name, progress=True)
            print(f"✓ Téléchargement terminé")
            
            print(f"\n📦 Extraction de l'archive vers {outpath}...")
            print("   (Cela peut prendre plusieurs minutes, veuillez patienter...)\n")
            _extract_tar(zf.name, outpath)
            print("\n✓ Extraction terminée")
    
    print("\n" + "=" * 60)
    print("✓ Téléchargement et extraction terminés avec succès!")
    print(f"  Données disponibles dans: {outpath}")
    print(f"  Note: Les fichiers train.txt et val.txt seront générés automatiquement")
    print(f"        lors de l'utilisation de LJSpeechDataModule.")
    print("=" * 60)


if __name__ == "__main__":
    main()
