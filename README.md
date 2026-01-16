<div align="center">

# Matcha-TTS: A fast TTS architecture with conditional flow matching

### Mathis Lecry, Paul-Marie Demars, Yucheng DAI, Minh Nhut NGUYEN

<div align="left">


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
ou sur le terminal Jupyter de l'école :
```bash
pip install -e .
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

Cela installe et compile le module d'alignement monotone (`monotonic_align`) en C pour accélérer l'entraînement.

### 4. Préparer les données

Le dataset LJSpeech doit être téléchargé et placé dans le dossier `data/` à **la racine du projet** :

```
Matcha-TTS-etu/
├── train.py
├── README.md
├── matcha/
└── data/                  ← À créer
    └── LJSpeech-1.1/      
        ├── metadata.csv
        └── wavs/
```

**Téléchargement du dataset :**

Pour télécharger le dataset on utilise le script de téléchargement automatique ljspeech.py récupéreé sur l'article. Il effectue automatiquement le traitement des donné, ainsi que la séparation du set Train et Val:

```bash
python -m matcha.utils.data_download.ljspeech.py
```

## Utilisation

### Entraînement complet

```bash
python train.py 
```
OU
```bash
python train.py --no-resume
```

### Génération audio

```bash
# Méthode 1 : Griffin-Lim (première méthode testée pour debugguer notre code)
python generate.py

# Méthode 2 : HiFi-GAN (Qualité audio finale)
python generate_HifiGan.py
```

Ou on peut utiliser le notebook demo_matcha.ipynb pour pouvoir lancer rapidement notre code et voir notre pipeline.

### Analyse des résultats

```bash
# Générer les graphiques de métriques étudiés
python analyze_training.py

# Visualiser avec TensorBoard
tensorboard --logdir lightning_logs
```

