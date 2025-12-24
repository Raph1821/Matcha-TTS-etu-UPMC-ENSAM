<div align="center">

# Matcha-TTS: A fast TTS architecture with conditional flow matching

### Mathis Lecry, Paul-Marie Demars, Yucheng, Minh Nhut NGUYEN

<div align="left">

# Dans le prompt cmd conda, se placer dans le dossier ou le code se trouve. 

Création de l'environnement: 

```
conda create -n matcha_tts_etu python=3.9
conda activate matcha_tts_etu 
```

# Assurez-vous d'être dans le dossier racine du projet (où se trouve requirements.txt)
```
pip install -r requirements.txt
```


## 📂 Architecture et Structure du Code

Voici comment les différents fichiers et dossiers interagissent pour permettre l'entraînement et la génération de voix.

### 1. Le Cœur du Modèle : `matcha/models/matcha_tts.py`
Ce fichier contient la classe principale **`MatchaTTS`**. C'est le "cerveau" du projet qui hérite de **`LightningModule`** (PyTorch Lightning).

* **Son rôle :** Il assemble les briques fondamentales.
* **Ce qu'il contient :**
    * **Text Encoder :** Convertit le texte en vecteurs.
    * **Decoder (U-Net) :** C'est ici que se fait le *Flow Matching*. Il prédit le champ de vecteurs pour transformer le bruit en spectrogramme.
    * **Fonction de perte (Loss) :** Il calcule l'écart entre la prédiction et la réalité pour permettre au modèle d'apprendre.
    * **Optimiseur :** Il définit comment les poids du réseau sont mis à jour (via AdamW généralement).

> **Lien :** C'est ce fichier qui est instancié par **`train.py`** pour être entraîné, et par **`generate.py`** pour créer de l'audio.

### 2. La Gestion des Données : `matcha/data_management/`
Ce dossier prépare le "carburant" du modèle. Il s'assure que les données (texte et audio) arrivent correctement formatées dans le réseau.

* **`ljspeechDataset.py` (L'ouvrier) :**
    * Il lit les fichiers physiques (fichiers **`.wav`** et transcriptions **`.txt`**).
    * Il transforme l'audio en **Mel-Spectrogramme** (la représentation visuelle du son que le modèle apprend à imiter).
    * Il nettoie et tokenise le texte.

* **`ljspeech_datamodule.py` (Le logisticien) :**
    * Il utilise la classe **`Dataset`** ci-dessus.
    * Il organise les données en lots (batches) pour ne pas saturer la mémoire.
    * Il divise les données en trois groupes : **Train** (entraînement), **Val** (validation) et **Test**.

> **Lien :** Ce module est appelé par **`train.py`** pour fournir les données au modèle **`MatchaTTS`** boucle après boucle.

### 3. L'Entraînement : `train.py`
C'est le script principal pour lancer l'apprentissage. Il joue le rôle de chef d'orchestre.

* **Son fonctionnement :**
    1.  Il charge la configuration (hyperparamètres).
    2.  Il instancie le **DataModule** (pour récupérer les données).
    3.  Il instancie le modèle **MatchaTTS**.
    4.  Il crée un **`Trainer`** (via PyTorch Lightning) qui gère la boucle d'entraînement, les sauvegardes automatiques (**`checkpoints`**) et les logs.
    5.  Il lance **`trainer.fit()`**.

> **Résultat :** À la fin (ou pendant) l'exécution de ce fichier, des fichiers **`.ckpt`** (checkpoints) sont créés dans le dossier **`lightning_logs/`**. Ce sont les sauvegardes de l'intelligence du modèle.

### 4. La Génération (Inférence) : `generate.py`
C'est le script final qui utilise ce qui a été appris pour parler.

* **Son fonctionnement :**
    1.  Il charge un fichier **`.ckpt`** (généré par **`train.py`**) pour restaurer le modèle **`MatchaTTS`** entraîné.
    2.  Il prend un texte en entrée.
    3.  Il utilise le **Flow Matching** (via le décodeur du modèle) pour générer un Mel-Spectrogramme.
    4.  **Vocoder :** Il envoie ce spectrogramme dans un Vocoder (ex: SpeechGAN ou HiFi-GAN) pour le transformer en fichier audio **`.wav`** écoutable.

### 🔄 Résumé du Flux de Données

1.  **Données brutes** (Wav/Txt)
    ⬇️ *(lues par)*
2.  **DataManagement** (**`ljspeech_datamodule.py`**)
    ⬇️ *(envoyées par batchs à)*
3.  **Entraînement** (**`train.py`** qui pilote **`matcha_tts.py`**)
    ⬇️ *(produit un)*
4.  **Checkpoint** (**`.ckpt`**)
    ⬇️ *(chargé par)*
5.  **Génération** (**`generate.py`**) $\rightarrow$ 🎵 **Audio Final**