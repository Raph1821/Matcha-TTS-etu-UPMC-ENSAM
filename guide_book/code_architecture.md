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