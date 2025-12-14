# ARCHITECTURE_PROTOCOL.md

## 1. Objectif du document

Ce document définit un **contrat d’architecture** pour le cœur du modèle acoustique Matcha-TTS,
tel qu’il est réimplémenté dans le cadre du projet académique
**Matcha-TTS-etu-UPMC-ENSAM**.

L’objectif est de :
- clarifier le rôle exact de chaque module,
- définir les **interfaces attendues** (entrées / sorties),
- garantir la **cohérence structurelle** entre les composants,
- permettre une réécriture indépendante du code source original,
  tout en respectant la logique décrite dans l’article scientifique.

Ce document sert de **référence commune** pour tous les membres de l’équipe
avant toute implémentation, refactorisation ou entraînement.

---

## 2. Périmètre de l’architecture

Le périmètre couvert par ce protocole correspond **exclusivement au modèle acoustique**,
c’est-à-dire à la transformation :

> texte / phonèmes → représentations acoustiques latentes (mels)

Les fichiers concernés sont localisés dans :
    matcha/models/components/


et incluent :

- `text_encoder.py`
- `transformer.py`
- `decoder.py`
- `flow_matching.py`

Ces fichiers constituent **le cœur algorithmique du modèle Matcha-TTS**,
directement lié aux contributions méthodologiques de l’article
(Conditional Flow Matching, ODE decoder, alignement implicite).

### Hors périmètre explicite

Les éléments suivants ne sont **pas** couverts par ce protocole :

- infrastructure d’entraînement (PyTorch Lightning, Trainer, optimisateurs),
- scripts CLI et inférence,
- vocoder (HiFi-GAN),
- chargement des données, métriques, callbacks,
- export, déploiement ou accélération matérielle.

Ces parties peuvent être modifiées, remplacées ou simplifiées
sans impacter la validité du présent protocole architectural.

---
## 3. Vue d’ensemble du pipeline conceptuel

### Pipeline de données (vue séquentielle)

| Étape | Module | Description |
|------|-------|------------|
| 1 | Texte brut | Entrée textuelle |
| 2 | Phonétisation | Conversion en tokens linguistiques |
| 3 | Text Encoder | Encodage contextuel du texte |
| 4 | Représentation *(h)* | Représentations continues |
| 5 | Prédiction de durées | Projection + durée par token |
| 6 | Upsampling | Alignement texte ↔ temps |
| 7 | Condition acoustique | Paramètres *(μ / mu)* |
| 8 | Flow Matching | ODE conditionnelle |
| 9 | Représentation latente | Mel-spectrogramme |


Principe général :

Matcha-TTS ne génère pas directement l’audio, ni même le spectrogramme
de manière autorégressive.

Le modèle apprend un champ de vecteurs conditionnel permettant de transformer
un bruit initial en une représentation acoustique cohérente, conditionnée
par le texte, via la résolution d’une équation différentielle ordinaire (ODE).

Chaque bloc du pipeline est conçu pour être :
- différentiable,
- modulaire,
- indépendant des scripts d’entraînement et de déploiement.

Cette séparation permet une réimplémentation fidèle de l’architecture
décrite dans l’article, sans dépendance directe au code source original.
---
## 4. Rôle et responsabilités des modules (`components/`)

Le dossier `matcha/models/components/` regroupe les **briques fonctionnelles
fondamentales** du modèle acoustique Matcha-TTS.  
Chaque fichier correspond à un **sous-problème bien identifié** du pipeline,
avec des responsabilités clairement séparées.

### 4.1 `text_encoder.py` — Encodage linguistique

**Responsabilité principale :**  
Transformer une séquence de tokens linguistiques (phonèmes, symboles)
en une représentation continue exploitable par le modèle acoustique.

**Entrées :**
- Séquence de tokens linguistiques (IDs entiers)
- Masques de longueur (optionnels)

**Sorties :**
- Représentations textuelles continues `h`
- Prédictions de durées (au niveau token)

**Rôle dans le pipeline :**
- Fournit la base sémantique et prosodique du signal
- Conditionne toutes les étapes acoustiques suivantes

---

### 4.2 `transformer.py` — Bloc de modélisation séquentielle

**Responsabilité principale :**  
Modéliser les dépendances temporelles et contextuelles
dans des séquences 1D (texte ou acoustique).

**Entrées :**
- Séquences continues (texte ou acoustique)
- Encodages temporels / embeddings de position ou de temps

**Sorties :**
- Séquences enrichies contextuellement, de même dimension

**Rôle dans le pipeline :**
- Sert de backbone commun (self-attention + feed-forward)
- Utilisé à la fois dans l’encodeur texte et dans le réseau de flow

---

### 4.3 `decoder.py` — Construction de la condition acoustique μ

**Responsabilité principale :**  
Aligner les représentations textuelles sur l’axe temporel
et produire la condition acoustique μ (mu).

**Entrées :**
- Représentations textuelles continues `h`
- Durées prédites par le text encoder

**Sorties :**
- Condition acoustique temporelle μ
- Masques temporels alignés

**Rôle dans le pipeline :**
- Réalise l’upsampling (réplication temporelle)
- Fait le lien explicite entre texte et temps acoustique

---

### 4.4 `flow_matching.py` — Modélisation acoustique par ODE

**Responsabilité principale :**  
Apprendre un champ de vecteurs conditionnel permettant
de transformer un bruit initial en représentation acoustique cohérente.

**Entrées :**
- État acoustique courant `x_t`
- Temps continu `t`
- Condition acoustique μ

**Sorties :**
- Vitesse `v_t(x | μ, t)` pour l’intégration de l’ODE

**Rôle dans le pipeline :**
- Cœur génératif du modèle acoustique
- Implémente le principe de Flow Matching décrit dans l’article

---

### 4.5 Principe d’isolement des modules

Chaque module respecte les règles suivantes :
- interfaces claires (entrées / sorties explicites),
- absence de dépendances circulaires,
- interchangeabilité possible (implémentations alternatives),
- testabilité indépendante.

Cette séparation garantit la lisibilité du modèle,
la reproductibilité scientifique
et la possibilité de réimplémentation complète sans dépendre du code original.
 ---
## 5. Contraintes d’implémentation et conventions de développement

Cette section définit les **règles techniques et conceptuelles**
à respecter lors de la réécriture des modules du dossier
`matcha/models/components/`.

L’objectif est d’assurer :
- la cohérence architecturale,
- la comparabilité avec l’article original,
- la maintenabilité du code dans un cadre académique.

---

### 5.1 Conventions générales de code

Tous les modules doivent respecter les principes suivants :

- implémentation en **PyTorch pur**,
- typage implicite clair (dimensions documentées),
- aucune dépendance au code officiel Matcha-TTS,
- usage explicite des tenseurs (pas de logique cachée),
- noms de classes et fonctions descriptifs.

Chaque fichier doit pouvoir être :
- lu indépendamment,
- testé indépendamment,
- remplacé indépendamment.

---

### 5.2 Interfaces et signatures

Les interfaces entre modules sont **contractuelles** :

- les formes des tenseurs doivent être clairement définies,
- aucune transformation implicite entre modules,
- les masques (padding, longueurs) sont passés explicitement.

Toute modification d’interface doit être :
- documentée,
- répercutée dans les modules dépendants,
- validée par des tests unitaires simples.

---

### 5.3 Contraintes sur le modèle Transformer

Pour `transformer.py`, les contraintes minimales sont :

- architecture Transformer 1D (self-attention),
- compatibilité avec des séquences de longueur variable,
- support des embeddings de position ou de temps,
- séparation claire entre :
  - attention,
  - feed-forward,
  - normalisation,
  - résidus.

Aucune hypothèse spécifique au texte ou à l’audio
ne doit être codée dans ce module.

---

### 5.4 Contraintes numériques et différentiabilité

Tous les calculs doivent être :

- entièrement différentiables,
- compatibles avec l’entraînement GPU,
- stables numériquement (normalisation, clipping si nécessaire).

Les opérations suivantes sont interdites :
- boucles Python dépendant du temps acoustique,
- opérations non différentiables dans le cœur du modèle.

---

### 5.5 Entraînement et intégration

Les modules `components/` :

- **ne gèrent pas l’entraînement**,
- **ne définissent pas d’optimiseur**,
- **ne chargent pas de données**.

Ils sont conçus pour être appelés par :
- `matcha_tts.py` (assemblage du modèle),
- une classe d’entraînement (ex. Lightning).

Toute logique liée à :
- l’entraînement,
- l’évaluation,
- le déploiement,
doit rester hors de ce dossier.

---

### 5.6 Philosophie pédagogique

Cette réimplémentation vise avant tout :

- la compréhension complète du modèle,
- la traçabilité scientifique,
- la capacité à expliquer chaque bloc mathématiquement.

La performance finale est **secondaire**
par rapport à la clarté, la modularité
et la fidélité conceptuelle à l’article original.
