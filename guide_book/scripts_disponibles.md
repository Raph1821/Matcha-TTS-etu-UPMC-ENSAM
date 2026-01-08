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