# Journal des modifications - Version de reproduction Matcha-TTS

Ce document détaille toutes les modifications apportées depuis la version originale de Matcha-TTS jusqu'à la version de reproduction actuelle.

## 📋 Vue d'ensemble des modifications

L'objectif principal de ces modifications est de **reproduire** (et non copier) l'architecture complète de Matcha-TTS originale, de corriger les composants clés manquants dans la version de développement, tout en maintenant la flexibilité et la robustesse du code.

## 🆕 I. Fichiers ajoutés

### 1. `matcha/utils/monotonic_align/__init__.py`
**Fonction** : Interface Python du module MAS (Monotonic Alignment Search)

**Caractéristiques** :
- Fournit la fonction `maximum_path()` pour l'alignement texte-audio
- Contient une implémentation de repli Python (lorsque l'extension Cython n'est pas disponible)
- Gère automatiquement les échecs d'importation Cython

**Méthode d'implémentation** : Version de reproduction avec gestion d'erreurs détaillée et mécanisme de repli

---

### 2. `matcha/utils/monotonic_align/core.pyx`
**Fonction** : Implémentation optimisée Cython de l'algorithme MAS

**Caractéristiques** :
- Utilise Cython pour accélérer les calculs d'alignement
- Contient des chaînes de documentation et commentaires détaillés
- Format et structure du code ajustés (reproduction plutôt que copie)

**Fonctions clés** :
- `maximum_path_each()` : Calcule le chemin d'alignement monotone pour un seul échantillon
- `maximum_path_c()` : Traitement parallèle par lots de plusieurs échantillons

---

### 3. `matcha/utils/model.py` (重构版本)

#### Améliorations de refactorisation
- ✅ Refactorise toutes les implémentations de fonctions, améliore les noms de variables et la structure du code
- ✅ Ajoute la documentation en français
- ✅ Maintient la compatibilité ascendante (via des alias de fonctions)
- ✅ Améliore la lisibilité et la maintenabilité du code

**Fonctions principales (nouveaux noms)** :
- `create_sequence_mask()` : Crée un masque de séquence
- `build_alignment_path()` : Génère un chemin d'alignement basé sur la durée
- `compute_duration_loss()` : Calcule la perte de durée
- `apply_normalization()` / `apply_denormalization()` : Normalisation et dénormalisation des données
- `adjust_length_for_downsampling()` : Ajuste la longueur pour la compatibilité avec le sous-échantillonnage

**Compatibilité ascendante** :
- Conserve les anciens noms de fonctions comme alias (`sequence_mask`, `generate_path`, `duration_loss`, `normalize`, `denormalize`, `fix_len_compatibility`)
- Le code existant peut être utilisé sans modification

---

### 4. `matcha/utils/pylogger.py`
**Fonction** : Module utilitaire de journalisation

**Caractéristiques** :
- Support de la journalisation pour l'entraînement multi-GPU
- Contient un repli lorsque pytorch_lightning n'est pas disponible
- Utilise le décorateur `rank_zero_only` pour éviter la duplication des logs

#### Améliorations de refactorisation récentes
- ✅ **Optimisation de la structure du code** : Divise la logique d'application du décorateur en fonctions indépendantes
  - `_create_noop_decorator()` : Crée un décorateur no-op (lorsque rank_zero_only n'est pas disponible)
  - `_apply_rank_zero_filter()` : Applique le filtre rank-zero à toutes les méthodes de journalisation
- ✅ **Amélioration de la méthode d'import** : Supporte deux chemins d'import (`pytorch_lightning` et `lightning`) avec try-except imbriqués
- ✅ **Amélioration des noms de variables** : `name` → `logger_name`, `logger` → `logger_instance`, améliore la lisibilité du code
- ✅ **Amélioration de la maintenabilité** : Structure du code plus claire, plus facile à comprendre et à maintenir

---

### 5. `matcha/models/baselightningmodule.py`
**Fonction** : Classe de base PyTorch Lightning

**Caractéristiques** :
- Fournit un flux d'entraînement/validation générique
- Supporte deux méthodes : objets de configuration originaux et paramètres simplifiés
- Gère automatiquement la configuration de l'optimiseur et du planificateur de taux d'apprentissage
- Contient des fonctionnalités de visualisation à la fin de la validation

**Méthodes clés** :
- `update_data_statistics()` : Met à jour les statistiques des données
- `configure_optimizers()` : Configure l'optimiseur (supporte deux méthodes)
- `get_losses()` : Obtient le dictionnaire des pertes
- `training_step()` / `validation_step()` : Étapes d'entraînement et de validation

---

### 6. `matcha/models/components/transformer.py`
**Fonction** : Implémentation de BasicTransformerBlock (pour l'U-Net du Decoder)

**Caractéristiques** :
- Supporte BasicTransformerBlock de la bibliothèque diffusers
- Contient une implémentation de repli complète (lorsque diffusers n'est pas disponible)
- Supporte plusieurs fonctions d'activation : GELU, GEGLU, SnakeBeta, etc.
- Corrige le traitement de la fonction d'activation "snake"

**Classes clés** :
- `SnakeBeta` : Variante de la fonction d'activation Snake
- `FeedForward` : Couche de réseau feed-forward
- `BasicTransformerBlock` : Bloc Transformer (avec support de repli)

---

## 🔄 II. Fichiers complètement réécrits

### 1. `matcha/models/components/text_encoder.py`

#### Problèmes de la version originale
- ❌ Utilisait le format seq-first `[B, T, C]`, nécessitant de fréquentes transpositions
- ❌ Manquait DurationPredictor (impossible de prédire la durée des phonèmes)
- ❌ Manquait RotaryPositionalEmbeddings (RoPE)
- ❌ Manquait prenet (réseau de prétraitement)
- ❌ Utilisait MultiHeadAttention standard basé sur Linear

#### Améliorations de la version de reproduction
- ✅ Utilise le format Conv1d `[B, C, T]` (cohérent avec la version originale)
- ✅ Ajoute `DurationPredictor` : Prédit la durée de chaque phonème
- ✅ Ajoute `RotaryPositionalEmbeddings (RoPE)` : Encodage de position rotatif
- ✅ Ajoute `prenet` : Réseau de prétraitement (ConvReluNorm)
- ✅ Utilise `MultiHeadAttention` basé sur Conv1d (avec support RoPE)
- ✅ Implémente les classes complètes `Encoder` et `TextEncoder`

#### Nouveaux composants
- `LayerNorm` : Normalisation de couche personnalisée
- `ConvReluNorm` : Bloc Convolution + ReLU + Normalisation
- `DurationPredictor` : Prédicteur de durée
- `RotaryPositionalEmbeddings` : Encodage de position RoPE
- `MultiHeadAttention` : Attention multi-têtes basée sur Conv1d (avec support RoPE)
- `FFN` : Réseau feed-forward
- `Encoder` : Empilement d'encodeurs Transformer
- `TextEncoder` : Encodeur de texte complet

**Valeur de retour** : `(mu, logw, x_mask)` - Trois valeurs (corrige le problème de valeur de retour de la version de développement originale)

---

### 2. `matcha/models/components/decoder.py`

#### Problèmes de la version originale
- ❌ Seulement un empilement simple de DecoderBlock
- ❌ Pas de structure U-Net
- ❌ Manquait les connexions skip
- ❌ Architecture trop simplifiée

#### Améliorations de la version de reproduction
- ✅ Implémente l'architecture U-Net complète (down blocks → mid blocks → up blocks)
- ✅ Ajoute `ResnetBlock1D`, `Block1D`, `Downsample1D`, `Upsample1D`
- ✅ Ajoute `TimestepEmbedding`, `SinusoidalPosEmb`
- ✅ Ajoute les connexions skip (corrige les problèmes de correspondance de taille)
- ✅ Supporte `ConformerWrapper` (optionnel)
- ✅ Corrige le traitement de la taille du masque lors du sous-échantillonnage/sur-échantillonnage

#### Nouveaux composants
- `SinusoidalPosEmb` : Encodage de position sinusoïdal
- `Block1D` : Bloc de convolution 1D
- `ResnetBlock1D` : Bloc résiduel (avec embedding temporel)
- `Downsample1D` / `Upsample1D` : Couches de sous-échantillonnage et sur-échantillonnage
- `TimestepEmbedding` : Embedding de pas de temps
- `ConformerWrapper` : Enveloppe de bloc Conformer (optionnel)
- `Decoder` : Décodeur U-Net complet

**Corrections clés** :
- Corrige les problèmes de correspondance de taille des connexions skip
- Améliore la logique de traitement du masque lors du sous-échantillonnage/sur-échantillonnage
- Ajoute des mécanismes de vérification et d'ajustement automatique de la taille

---

### 3. `matcha/models/components/flow_matching.py`

#### Problèmes de la version originale
- ❌ Seulement une classe de méthodes statiques
- ❌ Manquait l'implémentation CFM complète
- ❌ Pas de méthode `solve_euler`

#### Améliorations de la version de reproduction
- ✅ Implémente la classe de base `BASECFM` complète
- ✅ Implémente la classe `CFM` (Flow Matching complet)
- ✅ Ajoute la méthode `_solve_ode_euler()` (utilisée pour l'inférence, méthode privée)
- ✅ Ajoute la méthode `compute_loss()` (calcul de perte complet)
- ✅ Intègre Decoder comme estimateur
- ✅ **Compatibilité arrière** : `compute_loss()` supporte deux API (ancienne `x1/mask/mu` et nouvelle `target_sample/target_mask/encoder_output`)
- ✅ **Optimisation de la structure du code** : Divise la logique en méthodes auxiliaires (`_initialize_noise`, `_create_time_steps`, `_sample_random_time`, `_build_conditional_path`, `_compute_velocity_target`)

#### Corrections récentes (amélioration de la compatibilité arrière)
- ✅ **Correction du TypeError lors de l'entraînement** : Résout le problème où `compute_loss()` recevait un argument de mot-clé inattendu `x1`
- ✅ **Support double API** : La méthode `compute_loss()` accepte maintenant à la fois les anciens noms de paramètres (`x1`, `mask`, `mu`, `spks`, `cond`) et les nouveaux (`target_sample`, `target_mask`, `encoder_output`, `speaker_emb`, `condition`)
- ✅ **Mapping automatique des paramètres** : Si les nouveaux paramètres sont `None`, récupère automatiquement les valeurs des anciens paramètres, garantissant que le code existant fonctionne sans modification
- ✅ **Amélioration de la gestion d'erreurs** : Si aucune des deux API ne fournit les paramètres nécessaires, lance un message d'erreur clair

**Méthodes clés** :
- `forward()` : Diffusion avant (utilisée lors de l'inférence)
- `_solve_ode_euler()` : Solveur Euler (résolution ODE, méthode privée)
- `compute_loss()` : Calcule la perte Flow Matching (support des anciennes et nouvelles API)

**Méthodes auxiliaires** :
- `_initialize_noise()` : Initialise le bruit aléatoire
- `_create_time_steps()` : Crée la séquence de pas de temps
- `_sample_random_time()` : Échantillonne un temps aléatoire
- `_build_conditional_path()` : Construit le chemin conditionnel
- `_compute_velocity_target()` : Calcule la cible du champ de vitesse

---

### 4. `matcha/models/components/text_encoder.py` (重构版本)

#### Améliorations de refactorisation
- ✅ Restructure le code, améliore les noms de variables pour la lisibilité
- ✅ Ajoute des méthodes auxiliaires, améliore l'organisation du code
- ✅ Ajoute la documentation en français
- ✅ Maintient la compatibilité fonctionnelle et la cohérence de l'API

**Améliorations principales** :
- Noms de variables : `channels` → `feature_dim`/`channel_dim`, `n_heads` → `num_heads`, `p_dropout` → `dropout_rate`
- Refactorisation des méthodes : `attention` → `_compute_attention`, `_neg_half` → `_apply_neg_half_transform`
- Organisation du code : divise les méthodes complexes, ajoute une décomposition claire des étapes
- Documentation : ajoute des chaînes de documentation en français, reste concise

---

### 5. `matcha/models/matcha_tts.py`

#### Problèmes de la version originale
- ❌ Seulement Flow Matching Loss
- ❌ Manquait Duration Loss
- ❌ Manquait Prior Loss
- ❌ Sur-échantillonnage direct, pas de mécanisme d'alignement

#### Améliorations de la version de reproduction
- ✅ Ajoute trois fonctions de perte :
  - **Duration Loss** : Durée prédite vs durée alignée par MAS
  - **Prior Loss** : Différence entre mel et sortie de l'encodeur
  - **Flow Matching Loss** : Perte de prédiction du champ de vitesse
- ✅ Ajoute le mécanisme d'alignement MAS (Monotonic Alignment Search)
- ✅ Ajoute la méthode `synthesise()` (flux d'inférence complet)
- ✅ Ajoute la méthode `forward()` (flux d'entraînement complet)
- ✅ **Supporte deux méthodes d'initialisation** :
  - Méthode d'objet de configuration originale (compatible Hydra)
  - Méthode de paramètres simplifiés (`n_vocab`, `out_channels`, `hidden_channels`)

**Fonctionnalités clés** :
- Détection automatique de la méthode d'initialisation (objet de configuration vs paramètres simplifiés)
- Flux d'entraînement complet (inclut alignement et trois pertes)
- Flux d'inférence complet (inclut prédicteur de durée et alignement)

---

## ✏️ III. Fichiers modifiés

### 1. `matcha/utils/utils.py`

**Nouvelles fonctionnalités** :
- `plot_tensor()` : Convertit un tenseur en tableau d'images (pour l'enregistrement des logs)
- `save_figure_to_numpy()` : Convertit une figure matplotlib en tableau numpy

**Améliorations** :
- Compatible avec différentes versions de matplotlib
- Amélioration de la gestion des erreurs
- Ajout de commentaires détaillés en français

#### Améliorations de fonctionnalités récentes
- ✅ **Amélioration de `save_figure_to_numpy()`** :
  - Version originale utilisait `np.fromstring()` déprécié, supportait uniquement l'ancienne version de matplotlib
  - Version actuelle utilise `np.frombuffer()` (méthode recommandée)
  - Compatible avec les anciennes et nouvelles versions de matplotlib (gestion try-except)
  - Supporte la conversion RGBA vers RGB (les nouvelles versions de matplotlib utilisent buffer_rgba)
- ✅ **Amélioration de `plot_tensor()`** :
  - Version originale acceptait uniquement les tableaux numpy, sans vérification de type
  - Version actuelle supporte `torch.Tensor` et `numpy.ndarray`
  - Traitement automatique de la dimension batch (prend automatiquement le premier échantillon si `ndim == 3`)
  - Ajoute la vérification et validation des erreurs (lance une exception claire si `ndim != 2`)
- ✅ **Amélioration de la robustesse** : Meilleure gestion des erreurs, support de types plus large, support de versions plus compatible

---

### 2. `matcha/utils/__init__.py`

**Modifications** :
- Ajoute l'export de `pylogger`

---

### 3. `matcha/__init__.py`

**Modifications** :
- Ajoute l'export de `utils`

---

### 4. `train.py`

**Nouvelles fonctionnalités** :
- ✅ Supporte la reprise de l'entraînement depuis un checkpoint
- ✅ Recherche automatique du dernier checkpoint
- ✅ Support des arguments en ligne de commande
- ✅ Sauvegarde automatique des meilleurs modèles et du dernier modèle
- ✅ Gestion d'erreurs complète (démarrage automatique depuis le début si le checkpoint n'existe pas)

**Optimisations de configuration d'entraînement** :
- Découpage de gradient (`gradient_clip_val=1.0`) : prévient l'explosion du gradient
- Accumulation de gradient (`accumulate_grad_batches=2`) : augmente efficacement la taille du batch
- Optimisation du chargement des données : `pin_memory=False` et `persistent_workers=False`, évite les erreurs de réinitialisation de connexion en environnement multi-processus
- Stratégie de points de contrôle : surveillance de la perte de validation, sauvegarde des 3 meilleurs modèles et du dernier modèle

**Nouvelles fonctions** :
- `find_latest_checkpoint()` : Trouve le dernier fichier checkpoint (recherche récursive et tri par date de modification)

**Arguments en ligne de commande** :
- `--checkpoint` : Spécifie le chemin du checkpoint
- `--no-resume` : Force le démarrage depuis le début

**Exemples d'utilisation** :
```bash
# Recherche automatique du dernier checkpoint
python train.py

# Spécifier un checkpoint
python train.py --checkpoint path/to/checkpoint.ckpt

# Depuis le début
python train.py --no-resume
```

---

### 5. `generate.py`

**Corrections** :
- ✅ Corrige l'erreur de déballage de la valeur de retour de `TextEncoder` (de 2 à 3 valeurs)
- ✅ Utilise la méthode `model.synthesise()` (flux d'inférence complet)
- ✅ Améliore la logique de traitement du mel spectrogramme

---

### 6. `.gitignore`

**Règles d'ignorance incluses** (assure que ces fichiers ne seront pas soumis) :
- `data/` - Fichiers de base de données
- `lightning_logs/` - Fichiers checkpoint
- `generated_audio/` - Audio généré
- `*.ckpt` - Tous les fichiers checkpoint

---

### 7. `matcha/data_management/ljspeech_datamodule.py`

**Optimisation du chargement des données** :
- ✅ **Correction de l'erreur de sortie Ctrl+C** : Résout le `ConnectionResetError` qui se produit lors de la sortie de l'entraînement avec `Ctrl+C`
- ✅ **Cause racine** : `pin_memory=True` crée un thread en arrière-plan `_pin_memory_loop` en environnement multi-processus, et lorsque le processus principal est interrompu, ce thread tente de lire la queue, causant une erreur de réinitialisation de connexion
- ✅ **Solution** :
  - Ajoute les paramètres `pin_memory=False` et `persistent_workers=False` dans `LJSpeechDataModule.__init__()` (valeurs par défaut)
  - Applique ces paramètres dans `DataLoader`
  - Définit explicitement ces paramètres à `False` dans `train.py`
- ✅ **Impact** : Évite les erreurs de réinitialisation de connexion en environnement multi-processus, sortie d'entraînement plus propre, impact de performance minimal pour les données de type spectrogramme mel (environ 5-15%)

---

## 🔍 IV. Différences principales entre la version de reproduction et la version originale

### 1. Intégrité de l'architecture ✅
- **Version originale** : Architecture Matcha-TTS complète
- **Version de reproduction** : Corrige tous les composants clés manquants, l'architecture est alignée avec la version originale

---

### 2. Méthode d'implémentation du code 🔄
- **Version originale** : Utilise le système de configuration Hydra, passe les paramètres via des fichiers de configuration
- **Version de reproduction** :
  - ✅ Supporte la méthode d'objet de configuration originale (entièrement compatible)
  - ✅ Supporte la méthode de paramètres simplifiés (`n_vocab`, `out_channels`, `hidden_channels`)
  - ✅ Détection automatique de la méthode à utiliser

---

### 3. Gestion des dépendances 🛡️
- **Version originale** : Suppose que toutes les dépendances sont disponibles
- **Version de reproduction** :
  - ✅ Ajoute des mécanismes de repli (lorsque diffusers, conformer ne sont pas disponibles)
  - ✅ Ajoute un repli Python (lorsque Cython n'est pas disponible)
  - ✅ Gestion d'erreurs plus robuste

---

### 4. Corrections d'erreurs 🐛
- **Version originale** : Déjà testée et validée
- **Version de reproduction** :
  - ✅ Corrige les problèmes de correspondance de taille des connexions skip
  - ✅ Corrige le traitement du masque lors du sous-échantillonnage/sur-échantillonnage
  - ✅ Corrige le traitement de la fonction d'activation "snake"
  - ✅ Corrige les problèmes de configuration de l'optimiseur
  - ✅ Corrige les problèmes de valeur de retour de TextEncoder

---

### 5. Style de code 📝
- **Version originale** : Style d'ingénierie utilisant Hydra + fichiers de configuration
- **Version de reproduction** :
  - ✅ Maintient la logique centrale cohérente
  - ✅ Structure du code ajustée (reproduction plutôt que copie)
  - ✅ Ajoute des commentaires en français
  - ✅ Améliore la gestion des erreurs et l'expérience utilisateur

---

### 6. Perte d'entraînement 📊
- **Version originale** : Trois pertes (Duration Loss, Prior Loss, Flow Matching Loss)
- **Version de reproduction** : Mêmes trois pertes, logique d'implémentation cohérente

---

### 7. Mécanisme d'alignement 🔗
- **Version originale** : Utilise MAS (Monotonic Alignment Search)
- **Version de reproduction** : Utilise également MAS, inclut un repli Python

---

### 8. Récupération de checkpoint 💾
- **Version originale** : Nécessite de spécifier manuellement le chemin du checkpoint
- **Version de reproduction** :
  - ✅ Recherche automatique du dernier checkpoint
  - ✅ Support des arguments en ligne de commande
  - ✅ Gestion d'erreurs complète (démarrage automatique depuis le début s'il n'existe pas)

---

## 📊 V. Statistiques des modifications

### Fichiers ajoutés : 6
1. `matcha/utils/monotonic_align/__init__.py`
2. `matcha/utils/monotonic_align/core.pyx`
3. `matcha/utils/model.py`
4. `matcha/utils/pylogger.py`
5. `matcha/models/baselightningmodule.py`
6. `matcha/models/components/transformer.py`

---

### Complètement réécrits : 4
1. `matcha/models/components/text_encoder.py`
2. `matcha/models/components/decoder.py`
3. `matcha/models/components/flow_matching.py`
4. `matcha/models/matcha_tts.py`

---

### Fichiers modifiés : 7
1. `matcha/utils/utils.py`
2. `matcha/utils/__init__.py`
3. `matcha/__init__.py`
4. `train.py`
5. `generate.py`
6. `.gitignore` (règles nécessaires confirmées incluses)
7. `matcha/data_management/ljspeech_datamodule.py`

---

## 🎯 VI. Résumé des améliorations clés

### 1. Correction de l'architecture TextEncoder
- Passage du format seq-first simplifié au format Conv1d complet
- Ajout de composants clés : DurationPredictor, RoPE, prenet, etc.
- Utilisation de MultiHeadAttention basé sur Conv1d

---

### 2. Correction de l'architecture Decoder
- Passage d'un simple empilement de blocs à l'architecture U-Net complète
- Ajout de connexions skip
- Correction des problèmes de correspondance de taille

---

### 3. Implémentation Flow Matching
- Passage d'une classe de méthodes statiques à la classe CFM complète
- Ajout de la méthode solve_euler pour l'inférence

---

### 4. Intégrité de la perte d'entraînement
- Ajout de Duration Loss et Prior Loss
- Implémentation complète des trois fonctions de perte

---

### 5. Mécanisme d'alignement
- Implémentation de MAS (Monotonic Alignment Search)
- Remplacement de la méthode de sur-échantillonnage simple

---

### 6. Flexibilité et robustesse
- Supporte deux méthodes d'initialisation
- Ajout de mécanismes de repli
- Amélioration de la gestion des erreurs

---

### 7. Expérience utilisateur
- Supporte la récupération automatique de checkpoint
- Ajout d'arguments en ligne de commande
- Amélioration de la sortie des logs

---

## 📝 VII. Instructions d'utilisation

### Entraîner le modèle

```bash
# Reprendre automatiquement depuis le dernier checkpoint (s'il existe)
python train.py

# Spécifier un checkpoint
python train.py --checkpoint lightning_logs/version_X/checkpoints/xxx.ckpt

# Entraîner depuis le début
python train.py --no-resume
```
---

### Générer de l'audio

```bash
python generate.py
```

---

### Soumission Git

Assurez-vous que `.gitignore` est correctement configuré, les fichiers/répertoires suivants ne seront pas soumis :
- `data/` - Base de données
- `lightning_logs/` - Checkpoint
- `generated_audio/` - Audio généré
- `*.ckpt` - Fichiers checkpoint

---

## ✅ VIII. Liste de vérification

- [x] TextEncoder retourne trois valeurs (mu, logw, x_mask)
- [x] Decoder utilise l'architecture U-Net complète
- [x] Flow Matching a une implémentation CFM complète
- [x] Utilise trois fonctions de perte lors de l'entraînement
- [x] Utilise MAS pour l'alignement
- [x] Supporte la reprise de l'entraînement depuis un checkpoint
- [x] Supporte deux méthodes d'initialisation
- [x] Toutes les dépendances ont des mécanismes de repli
- [x] Le code est une reproduction plutôt qu'une copie complète

---

## 🔗 IX. Références

- Matcha-TTS original : [Dépôt GitHub](https://github.com/infinity-engines/Matcha-TTS)
- Dépôt de ce projet : [Matcha-TTS-etu-UPMC-ENSAM](https://github.com/Raph1821/Matcha-TTS-etu-UPMC-ENSAM)

---

**Dernière mise à jour** : 2025-01-XX
**Version** : Version de reproduction v1.0
