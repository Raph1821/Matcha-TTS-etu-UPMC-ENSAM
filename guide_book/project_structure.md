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