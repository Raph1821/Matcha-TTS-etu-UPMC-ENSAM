"""
Test de performance: Comparaison Cython vs Python pour Monotonic Alignment Search
"""
import time
import numpy as np
import torch
from matcha.utils.monotonic_align import maximum_path

# Vérifier si Cython est disponible
try:
    from matcha.utils.monotonic_align.core import compute_batch_alignments
    CYTHON_DISPONIBLE = True
except ImportError:
    CYTHON_DISPONIBLE = False


def creer_donnees_test(batch_size, text_length, audio_length):
    """Crée des données de test"""
    log_prior = torch.randn(batch_size, text_length, audio_length)
    mask = torch.ones(batch_size, text_length, audio_length)
    for i in range(batch_size):
        t_x = int(text_length * (0.7 + 0.3 * np.random.random()))
        t_y = int(audio_length * (0.7 + 0.3 * np.random.random()))
        mask[i, t_x:, :] = 0
        mask[i, :, t_y:] = 0
    return log_prior, mask


def version_python_fallback(paths, values, t_xs, t_ys, max_neg_val=-1e9):
    """Implémentation Python pure"""
    b = values.shape[0]
    for i in range(b):
        t_x, t_y = int(t_xs[i]), int(t_ys[i])
        value = values[i, :t_x, :t_y]
        path = paths[i, :t_x, :t_y]
        
        # Forward pass
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[x, y-1]
                if x == 0:
                    v_prev = 0. if y == 0 else max_neg_val
                else:
                    v_prev = value[x-1, y-1]
                value[x, y] = max(v_cur, v_prev) + value[x, y]
        
        # Backward pass
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[index, y] = 1
            if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
                index = index - 1


def maximum_path_python(value, mask):
    """Version Python de maximum_path"""
    value = value * mask
    device = value.device
    dtype = value.dtype
    value_np = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value_np).astype(np.int32)
    mask_np = mask.data.cpu().numpy()
    
    t_x_max = mask_np.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(2)[:, 0].astype(np.int32)
    version_python_fallback(path, value_np, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def tester_vitesse(batch_size, text_length, audio_length, nb_runs=20):
    """Test de vitesse pour une configuration donnée"""
    log_prior, mask = creer_donnees_test(batch_size, text_length, audio_length)
    
    # Test version actuelle (Cython ou Python)
    temps_actuel = []
    for _ in range(nb_runs):
        debut = time.time()
        maximum_path(log_prior.clone(), mask.clone())
        temps_actuel.append(time.time() - debut)
    
    temps_moyen_actuel = np.mean(temps_actuel) * 1000  # en ms
    
    # Test version Python si Cython disponible
    if CYTHON_DISPONIBLE:
        temps_python = []
        for _ in range(nb_runs):
            debut = time.time()
            maximum_path_python(log_prior.clone(), mask.clone())
            temps_python.append(time.time() - debut)
        
        temps_moyen_python = np.mean(temps_python) * 1000  # en ms
        acceleration = temps_moyen_python / temps_moyen_actuel
        
        return {
            "batch": batch_size,
            "text": text_length,
            "audio": audio_length,
            "cython_ms": f"{temps_moyen_actuel:.2f}",
            "python_ms": f"{temps_moyen_python:.2f}",
            "acceleration": f"{acceleration:.1f}x"
        }
    else:
        return {
            "batch": batch_size,
            "text": text_length,
            "audio": audio_length,
            "python_ms": f"{temps_moyen_actuel:.2f}",
            "cython": "N/A"
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Test de performance: Monotonic Alignment Search")
    print("=" * 70)
    
    if CYTHON_DISPONIBLE:
        print("✅ Version Cython disponible\n")
    else:
        print("⚠️  Version Cython non disponible (utilisation du fallback Python)\n")
    
    # Configurations de test
    configs = [
        (8, 50, 200),
        (16, 100, 500),
        (32, 150, 800),
    ]
    
    resultats = []
    for batch, text, audio in configs:
        resultat = tester_vitesse(batch, text, audio)
        resultats.append(resultat)
    
    # Affichage des résultats
    print(f"{'Batch':<6} {'Text':<6} {'Audio':<6} {'Cython (ms)':<12} {'Python (ms)':<12} {'Accélération':<12}")
    print("-" * 70)
    
    for r in resultats:
        if CYTHON_DISPONIBLE:
            print(f"{r['batch']:<6} {r['text']:<6} {r['audio']:<6} {r['cython_ms']:<12} {r['python_ms']:<12} {r['acceleration']:<12}")
        else:
            print(f"{r['batch']:<6} {r['text']:<6} {r['audio']:<6} {'N/A':<12} {r['python_ms']:<12} {'N/A':<12}")
    
    print("=" * 70)
