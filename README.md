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
python train.py 
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

### `train.py` - Entraînement

Lance l'entraînement du modèle Matcha-TTS.

**Utilisation basique :**
```bash
python train.py
```

**Options :**
```bash
python train.py --checkpoint path/to/checkpoint.ckpt  # Reprendre depuis un checkpoint
python train.py --no-resume                           # Forcer le démarrage depuis zéro
```

**Fonctionnalités :**
- Détection automatique du dernier checkpoint
- Reprise d'entraînement avec état complet (epoch, step, optimiseur)
- Sauvegarde automatique des 3 meilleurs modèles + le dernier
- Logs TensorBoard dans `lightning_logs/`
- Gradient clipping et accumulation

**Configuration :**
- Max epochs : 1000
- Batch size : 16
- Precision : 32-bit
- GPU : 1 device
- Accumulation : 2 batches

### `generate.py` - Génération audio (Griffin-Lim)

Génère de l'audio à partir de texte en utilisant Griffin-Lim pour la reconstruction.

**Configuration dans le script :**
```python
CHECKPOINT_PATH = None  # Auto-détection du dernier checkpoint
OUTPUT_FOLDER = "generated_audio"
TEXTE_A_DIRE = "Hello, I am your Matcha Text to Speech model."
```

**Utilisation :**
```bash
python generate.py
```

**Sorties :**
- `generated_audio/test_matcha.wav` : Fichier audio
- `generated_audio/mel_spectrogram.png` : Visualisation

**Processus :**
1. Charge le checkpoint entraîné
2. Convertit le texte en tokens
3. Génère le spectrogramme Mel via Flow Matching
4. Reconstruit l'audio avec InverseMelScale + Griffin-Lim

### `generate_HifiGan.py` - Génération audio (HiFi-GAN)

Version alternative utilisant le vocoder HiFi-GAN pour une meilleure qualité audio.

```bash
python generate_HifiGan.py
```

### `analyze_training.py` - Analyse des logs

Extrait et visualise les métriques d'entraînement depuis TensorBoard.

**Utilisation :**
```bash
python analyze_training.py
```

**Sorties dans `training_analysis/` :**
- `all_metrics.csv` : Toutes les métriques en format tableau
- `loss_train.png` : Évolution de la loss d'entraînement
- `loss_val.png` : Évolution de la loss de validation
- `learning_rate.png` : Évolution du learning rate
- `comparison_train_val.png` : Comparaison train/val
- Autres métriques disponibles

### `compiler_cython.py` - Compilation Cython

Compile les modules Cython pour optimiser les performances.

```bash
python compiler_cython.py
```

Compile spécifiquement `matcha/utils/monotonic_align/core.pyx` qui est utilisé pour l'alignement texte-audio.

### `test_monotonic_align_speed.py` - Test de performance

Compare la vitesse entre la version Python et Cython de l'alignement monotone.

```bash
python test_monotonic_align_speed.py
```

### `test_pipeline.ipynb` - Test du pipeline

Notebook Jupyter pour tester le pipeline complet de bout en bout.

## Structure du projet

```
Matcha-TTS-etu/
│
├── 📄 Scripts principaux
│   ├── train.py                      # Entraînement du modèle
│   ├── generate.py                   # Génération audio (Griffin-Lim)
│   ├── generate_HifiGan.py          # Génération audio (HiFi-GAN)
│   ├── analyze_training.py          # Analyse des métriques d'entraînement
│   ├── compiler_cython.py           # Compilation Cython
│   └── test_monotonic_align_speed.py # Test de performance
│
├── 📄 Configuration
│   ├── requirements.txt             # Dépendances Python
│   ├── setup.py                     # Installation du package
│   ├── checkpts/config.json         # Configuration du modèle
│   └── .gitignore
│
├── 📄 Documentation
│   ├── README.md                    # Ce fichier
│   ├── CHANGELOG.md                 # Historique des modifications
│   └── guide_book/                  # Guides et documentation détaillée
│       ├── ARCHITECTURE_PROTOCOL.md # Protocole d'architecture détaillé
│       └── README_CYTHON.md         # Documentation Cython
│
├── 📁 matcha/                       # Package principal
│   ├── models/                      # Modèles neuronaux
│   │   ├── matcha_tts.py           # Classe principale MatchaTTS
│   │   ├── baselightningmodule.py  # Module de base Lightning
│   │   └── components/             # Composants du modèle
│   │       ├── text_encoder.py     # Encodeur de texte (Transformer)
│   │       ├── decoder.py          # Décodeur U-Net
│   │       ├── flow_matching.py    # Algorithme Flow Matching
│   │       └── transformer.py      # Blocs Transformer
│   │
│   ├── data_management/            # Gestion des données
│   │   ├── ljspeechDataset.py     # Dataset PyTorch
│   │   └── ljspeech_datamodule.py # DataModule Lightning
│   │
│   ├── text_to_ID/                # Traitement du texte
│   │   ├── text_to_sequence.py    # Conversion texte → tokens
│   │   ├── cleaners.py            # Nettoyage du texte
│   │   ├── symbols.py             # Vocabulaire
│   │   ├── numbers.py             # Conversion nombres → texte
│   │   ├── cmudict.py             # Dictionnaire phonétique
│   │   └── cmudict-0.7b           # Données CMU
│   │
│   ├── utils/                     # Utilitaires
│   │   ├── audio_process.py       # Traitement audio
│   │   ├── model.py               # Utilitaires modèle
│   │   ├── utils.py               # Fonctions diverses
│   │   ├── monotonic_align/       # Alignement monotone (Cython)
│   │   └── data_download/         # Téléchargement données
│   │
│   ├── tests_text/                # Tests unitaires
│   └── hifigan/                   # Vocoder HiFi-GAN
│
├── 📁 hifi_gan/                    # Vocoder HiFi-GAN alternatif
│   ├── models.py
│   ├── env.py
│   └── utils.py
│
├── 📁 notebooks/                   # Notebooks Jupyter
│   ├── test_audio_to_Mel.ipynb
│   └── test_text.ipynb
│
└── 📁 lightning_logs/              # Logs et checkpoints
    └── version_X/checkpoints/      # Modèles sauvegardés (.ckpt)
```

## Architecture et Structure du Code

### 1. Le Cœur du Modèle : `matcha/models/matcha_tts.py`

Ce fichier contient la classe principale **`MatchaTTS`**. C'est le "cerveau" du projet qui hérite de **`LightningModule`** (PyTorch Lightning).

**Son rôle :** Il assemble les briques fondamentales.

**Ce qu'il contient :**
- **Text Encoder :** Convertit le texte en vecteurs contextuels
- **Decoder (U-Net) :** Implémente le Flow Matching pour transformer le bruit en spectrogramme
- **Fonction de perte (Loss) :** Calcule l'erreur de prédiction du champ de vecteurs
- **Optimiseur :** AdamW avec learning rate scheduler

**Méthodes clés :**
- `forward()` : Passe avant pour l'entraînement
- `synthesise()` : Génération audio (inférence)
- `training_step()` / `validation_step()` : Gestion Lightning

### 2. La Gestion des Données : `matcha/data_management/`

Ce dossier prépare le "carburant" du modèle.

**`ljspeechDataset.py` (Le Dataset) :**
- Lit les fichiers audio `.wav` et transcriptions `.txt`
- Transforme l'audio en **Mel-Spectrogramme** (80 bins)
- Nettoie et tokenise le texte
- Applique la normalisation

**`ljspeech_datamodule.py` (Le DataModule) :**
- Organise les données en batches
- Divise en Train (90%) / Val (5%) / Test (5%)
- Gère le parallélisme avec `num_workers`
- Configure pin_memory et persistent_workers

### 3. Les Composants : `matcha/models/components/`

**`text_encoder.py` - Encodage linguistique**
- Embedding des tokens de texte
- Transformer avec attention multi-têtes
- Prédiction des durées phonétiques
- Upsampling vers la dimension temporelle

**`decoder.py` - Décodeur U-Net**
- Architecture U-Net avec skip connections
- Conditionné par le temps (timestep embedding)
- Prédit le champ de vecteurs pour le Flow Matching
- Utilise des blocs Conformer/Transformer

**`flow_matching.py` - Flow Matching**
- Implémente l'ODE conditionnelle
- Résolution par méthode d'Euler
- Transforme bruit → spectrogramme Mel
- Contrôle par température et steps

**`transformer.py` - Blocs Transformer**
- Multi-Head Attention
- Feed-Forward Networks
- Layer Normalization
- Positional Encoding

### 4. Le Traitement du Texte : `matcha/text_to_ID/`

**Pipeline de conversion :**
```
Texte brut → Cleaning → Normalisation → Phonémisation → Tokens
```

- **`cleaners.py`** : Minuscules, suppression accents, normalisation
- **`numbers.py`** : "123" → "one hundred twenty three"
- **`cmudict.py`** : Dictionnaire phonétique anglais
- **`text_to_sequence.py`** : Orchestration complète
- **`symbols.py`** : Vocabulaire (lettres, phonèmes, ponctuation)

### 5. Les Utilitaires : `matcha/utils/`

**`audio_process.py` - Traitement audio**
- STFT (Short-Time Fourier Transform)
- Conversion vers Mel-spectrogram
- Normalisation / Dénormalisation
- Paramètres : n_fft=1024, hop_length=256, n_mels=80

**`monotonic_align/` - Alignement optimisé**
- Version Cython ultra-rapide
- Aligne le texte avec les frames audio
- Utilisé pendant l'entraînement

## Tests et Notebooks

### Notebooks disponibles

1. **`test_text.ipynb`** : Test du pipeline de traitement texte
   - Nettoyage
   - Phonémisation
   - Tokenisation

2. **`test_audio_to_Mel.ipynb`** : Test de conversion audio
   - Chargement WAV
   - STFT
   - Mel-spectrogram

3. **`test_pipeline.ipynb`** : Test du pipeline complet
   - Chargement données
   - Forward pass
   - Génération

### Tests unitaires

```bash
# Tests dans matcha/tests_text/
python -m pytest matcha/tests_text/
```

## Checkpoints et Logs

### Structure des logs

```
lightning_logs/
├── version_0/          # Premier entraînement
├── version_1/          # Deuxième entraînement
└── version_N/          # N-ième entraînement
    ├── checkpoints/
    │   ├── best-epoch=XX-loss=Y.YYY.ckpt  # Meilleurs modèles (top 3)
    │   └── last-epoch=XX-step=YYYY.ckpt   # Dernier checkpoint
    ├── events.out.tfevents.xxxxx          # TensorBoard
    └── hparams.yaml                       # Hyperparamètres
```

### Métriques sauvegardées

- `loss/train` : Loss d'entraînement
- `loss/val` : Loss de validation
- `learning_rate` : Taux d'apprentissage
- `epoch` : Numéro d'époque
- Custom metrics si ajoutées

## Notes importantes

- **GPU recommandé** : L'entraînement sur CPU est très lent
- **Mémoire** : Minimum 8GB de RAM, 4GB de VRAM GPU
- **Dataset** : LJSpeech (~2.5GB) recommandé pour débuter
- **Temps d'entraînement** : Plusieurs heures à jours selon GPU
- **Qualité audio** : Griffin-Lim = rapide mais qualité moyenne, HiFi-GAN = meilleure qualité

---

## 👥 Contributeurs

- Mathis Lecry
- Paul-Marie Demars
- Yucheng DAI
- Minh Nhut NGUYEN