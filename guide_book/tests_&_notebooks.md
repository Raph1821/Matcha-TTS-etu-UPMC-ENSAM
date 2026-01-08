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