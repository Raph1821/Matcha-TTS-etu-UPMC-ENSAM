# Compilation Cython - Guide rapide

## Pourquoi compiler ?

Le fichier `core.pyx` est du **code source Cython**, pas du code Python exécutable. Il doit être compilé en fichier binaire (`.so` sur Linux, `.pyd` sur Windows) avant d'être utilisé.

**Analogie** : C'est comme un fichier `.c` en C - vous devez le compiler avant de l'exécuter.

## Solution rapide

### Méthode 1 : Script de compilation (recommandé)

```bash
# 1. Installer Cython si nécessaire
pip install Cython

# 2. Compiler
python compiler_cython.py
```

### Méthode 2 : Utiliser setup.py

```bash
# 1. Installer Cython
pip install Cython

# 2. Installer le projet (compilera automatiquement)
pip install -e .
```

### Méthode 3 : Compilation manuelle

```bash
# 1. Installer Cython
pip install Cython

# 2. Compiler avec Cython directement
cython matcha/utils/monotonic_align/core.pyx

# 3. Compiler avec gcc (Linux) ou cl (Windows)
# (Plus complexe, non recommandé)
```

## Vérification

Après compilation, vous devriez voir un fichier `.so` (Linux) ou `.pyd` (Windows) dans :
```
matcha/utils/monotonic_align/core.cpython-*.so
```

Testez avec :
```bash
python test_monotonic_align_speed.py
```

Si vous voyez "✅ Version Cython disponible", c'est bon !

## Si la compilation échoue

Le code utilise automatiquement une version Python fallback (plus lente mais fonctionnelle). Vous pouvez continuer à utiliser le projet sans Cython.
