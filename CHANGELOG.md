# 修改日志 - Matcha-TTS 复现版本

本文档详细记录了从原始 Matcha-TTS 版本到当前复现版本的所有修改。

---

# Journal des modifications - Version de reproduction Matcha-TTS

Ce document détaille toutes les modifications apportées depuis la version originale de Matcha-TTS jusqu'à la version de reproduction actuelle.

---

## 📋 修改概览

本次修改的主要目标是**复现**（而非复制）原始 Matcha-TTS 的完整架构，修复开发版本中缺失的关键组件，同时保持代码的灵活性和健壮性。

---

## 📋 Vue d'ensemble des modifications

L'objectif principal de ces modifications est de **reproduire** (et non copier) l'architecture complète de Matcha-TTS originale, de corriger les composants clés manquants dans la version de développement, tout en maintenant la flexibilité et la robustesse du code.

---

## 🆕 I. 新增文件

---

## 🆕 I. Fichiers ajoutés

### 1. `matcha/utils/monotonic_align/__init__.py`
**功能**：MAS（Monotonic Alignment Search）模块的 Python 接口

**特点**：
- 提供 `maximum_path()` 函数用于文本-音频对齐
- 包含纯 Python 回退实现（当 Cython 扩展不可用时）
- 自动处理 Cython 导入失败的情况

**实现方式**：复现版本，包含详细的错误处理和回退机制

---

### 1. `matcha/utils/monotonic_align/__init__.py`
**Fonction** : Interface Python du module MAS (Monotonic Alignment Search)

**Caractéristiques** :
- Fournit la fonction `maximum_path()` pour l'alignement texte-audio
- Contient une implémentation de repli Python (lorsque l'extension Cython n'est pas disponible)
- Gère automatiquement les échecs d'importation Cython

**Méthode d'implémentation** : Version de reproduction avec gestion d'erreurs détaillée et mécanisme de repli

---

### 2. `matcha/utils/monotonic_align/core.pyx`
**功能**：MAS 算法的 Cython 优化实现

**特点**：
- 使用 Cython 加速对齐计算
- 包含详细的文档字符串和注释
- 代码格式和结构经过调整（复现而非复制）

**关键函数**：
- `maximum_path_each()`：计算单个样本的单调对齐路径
- `maximum_path_c()`：批量并行处理多个样本

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

### 3. `matcha/utils/model.py`
**功能**：与模型相关的工具函数集合

**包含的函数**：
- `sequence_mask()`：创建序列掩码
- `generate_path()`：基于时长生成对齐路径
- `duration_loss()`：计算时长损失
- `normalize()` / `denormalize()`：数据归一化和反归一化
- `fix_len_compatibility()`：修复长度兼容性（用于 U-Net 下采样）

**实现方式**：从原始版本复现，保持逻辑一致性

---

### 3. `matcha/utils/model.py`
**Fonction** : Collection de fonctions utilitaires liées au modèle

**Fonctions incluses** :
- `sequence_mask()` : Crée un masque de séquence
- `generate_path()` : Génère un chemin d'alignement basé sur la durée
- `duration_loss()` : Calcule la perte de durée
- `normalize()` / `denormalize()` : Normalisation et dénormalisation des données
- `fix_len_compatibility()` : Corrige la compatibilité de longueur (pour le sous-échantillonnage U-Net)

**Méthode d'implémentation** : Reproduit depuis la version originale, en conservant la logique cohérente

---

### 4. `matcha/utils/pylogger.py`
**功能**：日志记录工具模块

**特点**：
- 支持多 GPU 训练的日志记录
- 包含当 pytorch_lightning 不可用时的回退
- 使用 `rank_zero_only` 装饰器避免日志重复

---

### 4. `matcha/utils/pylogger.py`
**Fonction** : Module utilitaire de journalisation

**Caractéristiques** :
- Support de la journalisation pour l'entraînement multi-GPU
- Contient un repli lorsque pytorch_lightning n'est pas disponible
- Utilise le décorateur `rank_zero_only` pour éviter la duplication des logs

---

### 5. `matcha/models/baselightningmodule.py`
**功能**：PyTorch Lightning 基类

**特点**：
- 提供通用的训练/验证流程
- 支持两种方法：原始配置对象和简化参数
- 自动处理优化器和学习率调度器的配置
- 包含验证结束时的可视化功能

**关键方法**：
- `update_data_statistics()`：更新数据统计信息
- `configure_optimizers()`：配置优化器（支持两种方法）
- `get_losses()`：获取损失字典
- `training_step()` / `validation_step()`：训练和验证步骤

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
**功能**：BasicTransformerBlock 实现（用于 Decoder 的 U-Net）

**特点**：
- 支持来自 diffusers 库的 BasicTransformerBlock
- 包含完整的回退实现（当 diffusers 不可用时）
- 支持多种激活函数：GELU、GEGLU、SnakeBeta 等
- 修复了 "snake" 激活函数的处理

**关键类**：
- `SnakeBeta`：Snake 激活函数的变体
- `FeedForward`：前馈网络层
- `BasicTransformerBlock`：Transformer 块（带回退支持）

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

## 🔄 II. 完全重写的文件

---

## 🔄 II. Fichiers complètement réécrits

### 1. `matcha/models/components/text_encoder.py`

#### 原版本的问题
- ❌ 使用 seq-first 格式 `[B, T, C]`，需要频繁转置
- ❌ 缺少 DurationPredictor（无法预测音素时长）
- ❌ 缺少 RotaryPositionalEmbeddings（RoPE）
- ❌ 缺少 prenet（预处理网络）
- ❌ 使用基于 Linear 的标准 MultiHeadAttention

#### 复现版本的改进
- ✅ 使用 Conv1d 格式 `[B, C, T]`（与原始版本一致）
- ✅ 添加 `DurationPredictor`：预测每个音素的时长
- ✅ 添加 `RotaryPositionalEmbeddings (RoPE)`：旋转位置编码
- ✅ 添加 `prenet`：预处理网络（ConvReluNorm）
- ✅ 使用基于 Conv1d 的 `MultiHeadAttention`（支持 RoPE）
- ✅ 实现完整的 `Encoder` 和 `TextEncoder` 类

#### 新增组件
- `LayerNorm`：自定义层归一化
- `ConvReluNorm`：卷积 + ReLU + 归一化块
- `DurationPredictor`：时长预测器
- `RotaryPositionalEmbeddings`：RoPE 位置编码
- `MultiHeadAttention`：基于 Conv1d 的多头注意力（支持 RoPE）
- `FFN`：前馈网络
- `Encoder`：Transformer 编码器堆叠
- `TextEncoder`：完整的文本编码器

**返回值**：`(mu, logw, x_mask)` - 三个值（修复了原始开发版本的返回值问题）

---

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

#### 原版本的问题
- ❌ 只是简单的 DecoderBlock 堆叠
- ❌ 没有 U-Net 结构
- ❌ 缺少跳跃连接
- ❌ 架构过于简化

#### 复现版本的改进
- ✅ 实现完整的 U-Net 架构（down blocks → mid blocks → up blocks）
- ✅ 添加 `ResnetBlock1D`、`Block1D`、`Downsample1D`、`Upsample1D`
- ✅ 添加 `TimestepEmbedding`、`SinusoidalPosEmb`
- ✅ 添加跳跃连接（修复了尺寸匹配问题）
- ✅ 支持 `ConformerWrapper`（可选）
- ✅ 修复了下采样/上采样时掩码尺寸的处理

#### 新增组件
- `SinusoidalPosEmb`：正弦位置编码
- `Block1D`：1D 卷积块
- `ResnetBlock1D`：残差块（带时间嵌入）
- `Downsample1D` / `Upsample1D`：下采样和上采样层
- `TimestepEmbedding`：时间步嵌入
- `ConformerWrapper`：Conformer 块包装器（可选）
- `Decoder`：完整的 U-Net 解码器

**关键修复**：
- 修复了跳跃连接的尺寸匹配问题
- 改进了下采样/上采样时的掩码处理逻辑
- 添加了尺寸检查和自动调整机制

---

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

#### 原版本的问题
- ❌ 只是一个静态方法类
- ❌ 缺少完整的 CFM 实现
- ❌ 没有 `solve_euler` 方法

#### 复现版本的改进
- ✅ 实现完整的 `BASECFM` 基类
- ✅ 实现 `CFM` 类（完整的 Flow Matching）
- ✅ 添加 `solve_euler()` 方法（用于推理）
- ✅ 添加 `compute_loss()` 方法（完整的损失计算）
- ✅ 将 Decoder 集成为估计器

**关键方法**：
- `forward()`：前向扩散（用于推理）
- `solve_euler()`：Euler 求解器（ODE 求解）
- `compute_loss()`：计算 Flow Matching 损失

---

#### Problèmes de la version originale
- ❌ Seulement une classe de méthodes statiques
- ❌ Manquait l'implémentation CFM complète
- ❌ Pas de méthode `solve_euler`

#### Améliorations de la version de reproduction
- ✅ Implémente la classe de base `BASECFM` complète
- ✅ Implémente la classe `CFM` (Flow Matching complet)
- ✅ Ajoute la méthode `solve_euler()` (utilisée pour l'inférence)
- ✅ Ajoute la méthode `compute_loss()` (calcul de perte complet)
- ✅ Intègre Decoder comme estimateur

**Méthodes clés** :
- `forward()` : Diffusion avant (utilisée lors de l'inférence)
- `solve_euler()` : Solveur Euler (résolution ODE)
- `compute_loss()` : Calcule la perte Flow Matching

---

### 4. `matcha/models/components/text_encoder.py` (重构版本)

#### 重构改进
- ✅ 重构代码结构，改进变量命名以提高可读性
- ✅ 添加辅助方法，改进代码组织方式
- ✅ 添加法语文档注释
- ✅ 保持功能兼容性和 API 接口一致性

**主要改进**：
- 变量命名：`channels` → `feature_dim`/`channel_dim`，`n_heads` → `num_heads`，`p_dropout` → `dropout_rate`
- 方法重构：`attention` → `_compute_attention`，`_neg_half` → `_apply_neg_half_transform`
- 代码组织：拆分复杂方法，添加清晰的步骤分解
- 文档：添加法语文档字符串，保持简洁

---

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

#### 原版本的问题
- ❌ 只有 Flow Matching Loss
- ❌ 缺少 Duration Loss
- ❌ 缺少 Prior Loss
- ❌ 直接上采样，没有对齐机制

#### 复现版本的改进
- ✅ 添加三种损失函数：
  - **Duration Loss**：预测时长 vs MAS 对齐时长
  - **Prior Loss**：mel 与编码器输出的差异
  - **Flow Matching Loss**：速度场预测损失
- ✅ 添加 MAS（Monotonic Alignment Search）对齐机制
- ✅ 添加 `synthesise()` 方法（完整的推理流程）
- ✅ 添加 `forward()` 方法（完整的训练流程）
- ✅ **支持两种初始化方法**：
  - 原始配置对象方法（兼容 Hydra）
  - 简化参数方法（`n_vocab`、`out_channels`、`hidden_channels`）

**关键功能**：
- 自动检测初始化方法（配置对象 vs 简化参数）
- 完整的训练流程（包括对齐和三种损失）
- 完整的推理流程（包括时长预测和对齐）

---

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

## ✏️ III. 修改的文件

---

## ✏️ III. Fichiers modifiés

### 1. `matcha/utils/utils.py`

**新增功能**：
- `plot_tensor()`：将张量转换为图像数组（用于日志记录）
- `save_figure_to_numpy()`：将 matplotlib 图形转换为 numpy 数组

**改进**：
- 兼容不同版本的 matplotlib
- 改进了错误处理
- 添加了详细的中文注释

---

**Nouvelles fonctionnalités** :
- `plot_tensor()` : Convertit un tenseur en tableau d'images (pour l'enregistrement des logs)
- `save_figure_to_numpy()` : Convertit une figure matplotlib en tableau numpy

**Améliorations** :
- Compatible avec différentes versions de matplotlib
- Amélioration de la gestion des erreurs
- Ajout de commentaires détaillés en français

---

### 2. `matcha/utils/__init__.py`

**修改**：
- 添加 `pylogger` 的导出

---

**Modifications** :
- Ajoute l'export de `pylogger`

---

### 3. `matcha/__init__.py`

**修改**：
- 添加 `utils` 的导出

---

**Modifications** :
- Ajoute l'export de `utils`

---

### 4. `train.py`

**新功能**：
- ✅ 支持从 checkpoint 恢复训练
- ✅ 自动查找最新 checkpoint
- ✅ 支持命令行参数
- ✅ 自动保存最佳模型和最新模型
- ✅ 完整的错误处理（如果 checkpoint 不存在则自动从头开始）

**新增函数**：
- `find_latest_checkpoint()`：查找最新的 checkpoint 文件

**命令行参数**：
- `--checkpoint`：指定 checkpoint 路径
- `--no-resume`：强制从头开始

**使用示例**：
```bash
# 自动查找最新 checkpoint
python train.py

# 指定 checkpoint
python train.py --checkpoint path/to/checkpoint.ckpt

# 从头开始
python train.py --no-resume
```

---

**Nouvelles fonctionnalités** :
- ✅ Supporte la reprise de l'entraînement depuis un checkpoint
- ✅ Recherche automatique du dernier checkpoint
- ✅ Support des arguments en ligne de commande
- ✅ Sauvegarde automatique des meilleurs modèles et du dernier modèle
- ✅ Gestion d'erreurs complète (démarrage automatique depuis le début si le checkpoint n'existe pas)

**Nouvelles fonctions** :
- `find_latest_checkpoint()` : Trouve le dernier fichier checkpoint

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

**修复**：
- ✅ 修复了 `TextEncoder` 返回值解包错误（从 2 个值到 3 个值）
- ✅ 使用 `model.synthesise()` 方法（完整的推理流程）
- ✅ 改进了 mel 频谱图处理逻辑

---

**Corrections** :
- ✅ Corrige l'erreur de déballage de la valeur de retour de `TextEncoder` (de 2 à 3 valeurs)
- ✅ Utilise la méthode `model.synthesise()` (flux d'inférence complet)
- ✅ Améliore la logique de traitement du mel spectrogramme

---

### 6. `.gitignore`

**包含的忽略规则**（确保这些文件不会被提交）：
- `data/` - 数据库文件
- `lightning_logs/` - Checkpoint 文件
- `generated_audio/` - 生成的音频
- `*.ckpt` - 所有 checkpoint 文件

---

**Règles d'ignorance incluses** (assure que ces fichiers ne seront pas soumis) :
- `data/` - Fichiers de base de données
- `lightning_logs/` - Fichiers checkpoint
- `generated_audio/` - Audio généré
- `*.ckpt` - Tous les fichiers checkpoint

---

## 🔍 IV. 复现版本与原始版本的主要区别

---

## 🔍 IV. Différences principales entre la version de reproduction et la version originale

### 1. 架构完整性 ✅
- **原始版本**：完整的 Matcha-TTS 架构
- **复现版本**：修复了所有缺失的关键组件，架构与原始版本对齐

---

### 1. Intégrité de l'architecture ✅
- **Version originale** : Architecture Matcha-TTS complète
- **Version de reproduction** : Corrige tous les composants clés manquants, l'architecture est alignée avec la version originale

---

### 2. 代码实现方式 🔄
- **原始版本**：使用 Hydra 配置系统，通过配置文件传递参数
- **复现版本**：
  - ✅ 支持原始配置对象方法（完全兼容）
  - ✅ 支持简化参数方法（`n_vocab`、`out_channels`、`hidden_channels`）
  - ✅ 自动检测使用哪种方法

---

### 2. Méthode d'implémentation du code 🔄
- **Version originale** : Utilise le système de configuration Hydra, passe les paramètres via des fichiers de configuration
- **Version de reproduction** :
  - ✅ Supporte la méthode d'objet de configuration originale (entièrement compatible)
  - ✅ Supporte la méthode de paramètres simplifiés (`n_vocab`, `out_channels`, `hidden_channels`)
  - ✅ Détection automatique de la méthode à utiliser

---

### 3. 依赖管理 🛡️
- **原始版本**：假设所有依赖都可用
- **复现版本**：
  - ✅ 添加回退机制（当 diffusers、conformer 不可用时）
  - ✅ 添加 Python 回退（当 Cython 不可用时）
  - ✅ 更健壮的错误处理

---

### 3. Gestion des dépendances 🛡️
- **Version originale** : Suppose que toutes les dépendances sont disponibles
- **Version de reproduction** :
  - ✅ Ajoute des mécanismes de repli (lorsque diffusers, conformer ne sont pas disponibles)
  - ✅ Ajoute un repli Python (lorsque Cython n'est pas disponible)
  - ✅ Gestion d'erreurs plus robuste

---

### 4. 错误修复 🐛
- **原始版本**：已经过测试和验证
- **复现版本**：
  - ✅ 修复了跳跃连接的尺寸匹配问题
  - ✅ 修复了下采样/上采样时的掩码处理
  - ✅ 修复了 "snake" 激活函数的处理
  - ✅ 修复了优化器配置问题
  - ✅ 修复了 TextEncoder 返回值问题

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

### 5. 代码风格 📝
- **原始版本**：使用 Hydra + 配置文件的工程风格
- **复现版本**：
  - ✅ 保持核心逻辑一致
  - ✅ 调整代码结构（复现而非复制）
  - ✅ 添加中文注释
  - ✅ 改进错误处理和用户体验

---

### 5. Style de code 📝
- **Version originale** : Style d'ingénierie utilisant Hydra + fichiers de configuration
- **Version de reproduction** :
  - ✅ Maintient la logique centrale cohérente
  - ✅ Structure du code ajustée (reproduction plutôt que copie)
  - ✅ Ajoute des commentaires en français
  - ✅ Améliore la gestion des erreurs et l'expérience utilisateur

---

### 6. 训练损失 📊
- **原始版本**：三种损失（Duration Loss、Prior Loss、Flow Matching Loss）
- **复现版本**：相同的三种损失，实现逻辑一致

---

### 6. Perte d'entraînement 📊
- **Version originale** : Trois pertes (Duration Loss, Prior Loss, Flow Matching Loss)
- **Version de reproduction** : Mêmes trois pertes, logique d'implémentation cohérente

---

### 7. 对齐机制 🔗
- **原始版本**：使用 MAS（Monotonic Alignment Search）
- **复现版本**：同样使用 MAS，包含 Python 回退

---

### 7. Mécanisme d'alignement 🔗
- **Version originale** : Utilise MAS (Monotonic Alignment Search)
- **Version de reproduction** : Utilise également MAS, inclut un repli Python

---

### 8. Checkpoint 恢复 💾
- **原始版本**：需要手动指定 checkpoint 路径
- **复现版本**：
  - ✅ 自动查找最新 checkpoint
  - ✅ 支持命令行参数
  - ✅ 完整的错误处理（如果不存在则自动从头开始）

---

### 8. Récupération de checkpoint 💾
- **Version originale** : Nécessite de spécifier manuellement le chemin du checkpoint
- **Version de reproduction** :
  - ✅ Recherche automatique du dernier checkpoint
  - ✅ Support des arguments en ligne de commande
  - ✅ Gestion d'erreurs complète (démarrage automatique depuis le début s'il n'existe pas)

---

## 📊 V. 修改统计

---

## 📊 V. Statistiques des modifications

### 新增文件：6 个
1. `matcha/utils/monotonic_align/__init__.py`
2. `matcha/utils/monotonic_align/core.pyx`
3. `matcha/utils/model.py`
4. `matcha/utils/pylogger.py`
5. `matcha/models/baselightningmodule.py`
6. `matcha/models/components/transformer.py`

---

### Fichiers ajoutés : 6
1. `matcha/utils/monotonic_align/__init__.py`
2. `matcha/utils/monotonic_align/core.pyx`
3. `matcha/utils/model.py`
4. `matcha/utils/pylogger.py`
5. `matcha/models/baselightningmodule.py`
6. `matcha/models/components/transformer.py`

---

### 完全重写：4 个
1. `matcha/models/components/text_encoder.py`
2. `matcha/models/components/decoder.py`
3. `matcha/models/components/flow_matching.py`
4. `matcha/models/matcha_tts.py`

---

### Complètement réécrits : 4
1. `matcha/models/components/text_encoder.py`
2. `matcha/models/components/decoder.py`
3. `matcha/models/components/flow_matching.py`
4. `matcha/models/matcha_tts.py`

---

### 修改的文件：6 个
1. `matcha/utils/utils.py`
2. `matcha/utils/__init__.py`
3. `matcha/__init__.py`
4. `train.py`
5. `generate.py`
6. `.gitignore`（已确认包含必要的规则）

---

### Fichiers modifiés : 6
1. `matcha/utils/utils.py`
2. `matcha/utils/__init__.py`
3. `matcha/__init__.py`
4. `train.py`
5. `generate.py`
6. `.gitignore` (règles nécessaires confirmées incluses)

---

## 🎯 VI. 关键改进总结

---

## 🎯 VI. Résumé des améliorations clés

### 1. TextEncoder 架构修复
- 从简化的 seq-first 格式转换为完整的 Conv1d 格式
- 添加关键组件：DurationPredictor、RoPE、prenet 等
- 使用基于 Conv1d 的 MultiHeadAttention

---

### 1. Correction de l'architecture TextEncoder
- Passage du format seq-first simplifié au format Conv1d complet
- Ajout de composants clés : DurationPredictor, RoPE, prenet, etc.
- Utilisation de MultiHeadAttention basé sur Conv1d

---

### 2. Decoder 架构修复
- 从简单的块堆叠转换为完整的 U-Net 架构
- 添加跳跃连接
- 修复尺寸匹配问题

---

### 2. Correction de l'architecture Decoder
- Passage d'un simple empilement de blocs à l'architecture U-Net complète
- Ajout de connexions skip
- Correction des problèmes de correspondance de taille

---

### 3. Flow Matching 实现
- 从静态方法类转换为完整的 CFM 类
- 添加用于推理的 solve_euler 方法

---

### 3. Implémentation Flow Matching
- Passage d'une classe de méthodes statiques à la classe CFM complète
- Ajout de la méthode solve_euler pour l'inférence

---

### 4. 训练损失完整性
- 添加 Duration Loss 和 Prior Loss
- 完整实现三种损失函数

---

### 4. Intégrité de la perte d'entraînement
- Ajout de Duration Loss et Prior Loss
- Implémentation complète des trois fonctions de perte

---

### 5. 对齐机制
- 实现 MAS（Monotonic Alignment Search）
- 替换简单的上采样方法

---

### 5. Mécanisme d'alignement
- Implémentation de MAS (Monotonic Alignment Search)
- Remplacement de la méthode de sur-échantillonnage simple

---

### 6. 灵活性和健壮性
- 支持两种初始化方法
- 添加回退机制
- 改进错误处理

---

### 6. Flexibilité et robustesse
- Supporte deux méthodes d'initialisation
- Ajout de mécanismes de repli
- Amélioration de la gestion des erreurs

---

### 7. 用户体验
- 支持自动恢复 checkpoint
- 添加命令行参数
- 改进日志输出

---

### 7. Expérience utilisateur
- Supporte la récupération automatique de checkpoint
- Ajout d'arguments en ligne de commande
- Amélioration de la sortie des logs

---

## 📝 VII. 使用说明

---

## 📝 VII. Instructions d'utilisation

### 训练模型

```bash
# 自动从最新 checkpoint 恢复（如果存在）
python train.py

# 指定 checkpoint
python train.py --checkpoint lightning_logs/version_X/checkpoints/xxx.ckpt

# 从头开始训练
python train.py --no-resume
```

---

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

### 生成音频

```bash
python generate.py
```

---

### Générer de l'audio

```bash
python generate.py
```

---

### Git 提交

确保 `.gitignore` 配置正确，以下文件/目录不会被提交：
- `data/` - 数据库
- `lightning_logs/` - Checkpoint
- `generated_audio/` - 生成的音频
- `*.ckpt` - Checkpoint 文件

---

### Soumission Git

Assurez-vous que `.gitignore` est correctement configuré, les fichiers/répertoires suivants ne seront pas soumis :
- `data/` - Base de données
- `lightning_logs/` - Checkpoint
- `generated_audio/` - Audio généré
- `*.ckpt` - Fichiers checkpoint

---

## ✅ VIII. 检查清单

---

## ✅ VIII. Liste de vérification

- [x] TextEncoder 返回三个值（mu, logw, x_mask）
- [x] Decoder 使用完整的 U-Net 架构
- [x] Flow Matching 有完整的 CFM 实现
- [x] 训练时使用三种损失函数
- [x] 使用 MAS 进行对齐
- [x] 支持从 checkpoint 恢复训练
- [x] 支持两种初始化方法
- [x] 所有依赖都有回退机制
- [x] 代码是复现而非完全复制

---

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

## 🔗 IX. 参考资料

---

## 🔗 IX. Références

- 原始 Matcha-TTS：[GitHub 仓库](https://github.com/infinity-engines/Matcha-TTS)
- 本项目仓库：[Matcha-TTS-etu-UPMC-ENSAM](https://github.com/Raph1821/Matcha-TTS-etu-UPMC-ENSAM)

---

- Matcha-TTS original : [Dépôt GitHub](https://github.com/infinity-engines/Matcha-TTS)
- Dépôt de ce projet : [Matcha-TTS-etu-UPMC-ENSAM](https://github.com/Raph1821/Matcha-TTS-etu-UPMC-ENSAM)

---

**最后更新**：2025-01-XX
**版本**：复现版本 v1.0

---

**Dernière mise à jour** : 2025-01-XX
**Version** : Version de reproduction v1.0
