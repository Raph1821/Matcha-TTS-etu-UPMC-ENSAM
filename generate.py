import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matcha.models.matcha_tts import MatchaTTS
from matcha.text_to_ID.text_to_sequence import text_to_sequence

# --- CONFIGURATION ---
CHECKPOINT_PATH = None  # Laissera le script trouver le dernier automatiquement
OUTPUT_FOLDER = "generated_audio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXTE_A_DIRE = "Hello, this is a test of the Matcha TTS model."

def get_latest_checkpoint(logs_dir="lightning_logs"):
    """Trouve automatiquement le dernier fichier .ckpt"""
    import glob
    # Cherche récursivement tous les .ckpt
    files = glob.glob(f"{logs_dir}/**/*.ckpt", recursive=True)
    if not files:
        raise FileNotFoundError("Aucun checkpoint trouvé ! As-tu lancé l'entraînement ?")
    # Trie par date de modification (le plus récent en dernier)
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ Checkpoint trouvé : {latest_file}")
    return latest_file

def simple_euler_ode_solver(model, mu, n_steps=10):
    """
    Le cœur du Flow Matching : transforme le bruit en son pas à pas.
    """
    # 1. On part d'un bruit blanc (t=0)
    # mu shape: [1, 80, T]
    z = torch.randn_like(mu, device=DEVICE)
    
    # 2. On avance dans le temps de 0 à 1
    dt = 1.0 / n_steps
    
    print(f"🔄 Génération en {n_steps} étapes...")
    
    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.tensor([t_val], device=DEVICE)
        
        # Le décodeur prédit la vitesse (le vecteur direction)
        # On n'a pas besoin de masque ici car on génère tout
        v_pred = model.decoder(z, t, mu, mask=None)
        
        # Euler step : nouvelle position = ancienne + vitesse * temps
        z = z + v_pred * dt
        
    return z # C'est notre spectrogramme généré (y_hat)

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Chargement du modèle
    ckpt = CHECKPOINT_PATH if CHECKPOINT_PATH else get_latest_checkpoint()
    print("⏳ Chargement du modèle...")
    
    # On charge le modèle et ses hyperparamètres
    model = MatchaTTS.load_from_checkpoint(ckpt)
    model.to(DEVICE)
    model.eval() # Mode évaluation (désactive le dropout)

    # 2. Préparation du texte
    print(f"📖 Texte : '{TEXTE_A_DIRE}'")
    sequence = text_to_sequence(TEXTE_A_DIRE, ["english_cleaners"]) # Ou basic_cleaners
    x = torch.tensor([sequence], dtype=torch.long, device=DEVICE)
    x_lengths = torch.tensor([len(sequence)], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        # 3. Encodage (Texte -> Mu)
        mu, _ = model.encoder(x, x_lengths)
        
        # --- ALIGNEMENT MANUEL (IMPORTANT) ---
        # Comme on a utilisé l'interpolation à l'entraînement, on doit le refaire ici.
        # On décide arbitrairement qu'un caractère dure X temps.
        # Facteur 5 = chaque lettre dure un peu de temps. Ajuste si ça parle trop vite/lentement.
        duree_audio_cible = mu.shape[-1] * 5 
        mu = torch.nn.functional.interpolate(mu, size=duree_audio_cible, mode='nearest')
        
        # 4. Flow Matching (Génération du Spectrogramme)
        spectrogram = simple_euler_ode_solver(model, mu, n_steps=50)

    # 5. Conversion Spectrogramme -> Audio (Griffin-Lim)
    # C'est une méthode mathématique pour reconstruire le son sans Vocoder entraîné
    # 5. Conversion Spectrogramme -> Audio (Inverse Mel + Griffin-Lim)
    print("🔊 Conversion en audio (InvMel -> Griffin-Lim)...")
    
    # A. Création de la transformation Inverse Mel (Pour passer de 80 -> 513 canaux)
    # On doit utiliser les mêmes paramètres que ceux utilisés pour créer le dataset LJSpeech
    inv_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1,  # = 513 bins de fréquence
        n_mels=80,
        sample_rate=22050,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney' 
    ).to(DEVICE)

    # B. Configuration de Griffin-Lim (Pour passer de Spectrogramme -> Onde)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024, 
        n_iter=32, 
        hop_length=256,
        win_length=1024,
        power=1.0
    ).to(DEVICE)
    
    # C. Exécution du Pipeline
    # 1. Le modèle sort des log-mels, on repasse en échelle normale avec exp()
    mel_spectrogram = torch.exp(spectrogram)
    
    # 2. On "décompresse" : Mel (80) -> Linéaire (513)
    linear_spectrogram = inv_mel_scale(mel_spectrogram)
    
    # 3. On reconstruit la phase et l'onde sonore
    waveform = griffin_lim(linear_spectrogram)

    # 6. Sauvegarde
    save_path = os.path.join(OUTPUT_FOLDER, "test_matcha.wav")
    torchaudio.save(save_path, waveform.cpu(), sample_rate=22050)
    print(f"✨ Audio sauvegardé dans : {save_path}")

    # (Optionnel) Afficher le spectrogramme
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.squeeze().cpu().numpy(), origin='lower', aspect='auto')
    plt.title("Spectrogramme Généré")
    plt.colorbar()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "spectrogram.png"))
    print("📊 Spectrogramme sauvegardé.")

if __name__ == "__main__":
    main()