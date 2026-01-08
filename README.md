<div align="center">

# Matcha-TTS: A fast TTS architecture with conditional flow matching

### Mathis Lecry, Paul-Marie Demars, Yucheng DAI, Minh Nhut NGUYEN

<div align="left">

## 📋 Table des matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Flux de Données Complet](#flux-de-données-complet)
- [Scripts disponibles](#scripts-disponibles)
- [Structure du projet](#structure-du-projet)
- [Architecture du code](#architecture-et-structure-du-code)
- [Tests et notebooks](#tests-et-notebooks)
- [Checkpoints et Logs](#checkpoints-et-logs)


## Installation

### 1. Créer l'environnement conda

Dans le terminal (cmd ou PowerShell), se placer dans le dossier du projet :

```bash
cd path/to/Matcha-TTS-etu
```

Créer et activer l'environnement :

```bash
conda create -n matcha-tts python=3.10
conda activate matcha-tts 
```

### 2. Installer les dépendances

Assurez-vous d'être dans le dossier racine du projet (où se trouve `requirements.txt`) :

```bash
pip install -r requirements.txt
```

### 3. (Optionnel) Compiler les modules Cython

Pour de meilleures performances (recommandé) :
Le module `monotonic_align` est critique pour les performances. Il calcule l'alignement optimal entre le texte et l'audio.

**Sans Cython :** Version Python pure (lente)  
**Avec Cython :** Version compilée en C (10-100x plus rapide)

```bash
pip install Cython
python compiler_cython.py
```

Cela compile le module d'alignement monotone (`monotonic_align`) en C pour accélérer l'entraînement.
Voir [README_CYTHON](guide_book/README_CYTHON.md) pour plus de détails.

### 4. Préparer les données

Le dataset LJSpeech doit être téléchargé et placé dans un dossier `data/` à **la racine du projet** :

```
Matcha-TTS-etu/
├── train.py
├── README.md
├── matcha/
└── data/                  ← À créer
    └── LJSpeech-1.1/      ← Dataset à télécharger
        ├── metadata.csv
        └── wavs/
```

**Téléchargement du dataset :**

Option 1 - Script automatique :
```bash
# Utiliser le script de téléchargement (si disponible)
python -m matcha.utils.data_download.ljspeech
```

Option 2 - Manuel :
1. Télécharger depuis : https://keithito.com/LJ-Speech-Dataset/
2. Extraire l'archive
3. Créer le dossier `data/` à la racine du projet
4. Placer le dossier `LJSpeech-1.1/` dans `data/`


## Utilisation

### Entraînement complet

```bash
# 1. Lancer l'entraînement
python train.py --checkpoint path/to/checkpoint.ckpt    # chemin des checkpoints
```
OU
```bash
# 2. Forcer un nouveau départ
python train.py --no-resume
```

### Génération audio

```bash
# Méthode 1 : Griffin-Lim (rapide mais qualité moyenne)
python generate.py

# Méthode 2 : HiFi-GAN (meilleure qualité)
python generate_HifiGan.py
```

### Analyse des résultats

```bash
# Générer les graphiques de métriques
python analyze_training.py

# Visualiser avec TensorBoard
tensorboard --logdir lightning_logs
```
## Flux de Données Complet

```
1. Texte brut : "Hello world"
   ↓
2. [text_to_sequence] → Tokens : [34, 12, 45, ...]
   ↓
3. [TextEncoder] → Vecteurs h : [batch, n_tokens, 192]
   ↓
4. [Duration Predictor] → Durées : [5, 3, 7, ...]
   ↓
5. [Upsampling] → h aligné : [batch, 192, T_audio]
   ↓
6. [Decoder + Flow Matching] → Mel : [batch, 80, T_audio]
   ↓
7. [Vocoder Griffin-Lim/HiFi-GAN] → Audio : waveform
   ↓
8. Fichier WAV sauvegardé
```

## Scripts disponibles

[scripts_disponibles](guide_book/scripts_disponibles.md)

## Structure du projet

[project_structure](guide_book/project_structure.md)

## Architecture et Structure du Code

[code_architecture](guide_book/code_architecture.md)

## Tests et Notebooks

[tests_&_notebooks](guide_book/tests_&_notebooks.md)

## Checkpoints et Logs

[ckpts_&_logs](guide_book/ckpts_&_logs.md)

---

## 👥 Contributeurs

- Mathis Lecry
- Paul-Marie Demars
- Yucheng DAI
- Minh Nhut NGUYEN