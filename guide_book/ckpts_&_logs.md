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