import sys
import json
sys.path.append('./hifi_gan') 
# Maintenant on peut importer les classes comme si elles étaient installées
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator as HiFiGAN
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
TEXTE_A_DIRE = "Hi guys, it is maybe time to work a little bit for the project, we are a group. Love you !"

# --- CONFIGURATION HIFI-GAN ---
HIFIGAN_CONFIG = './checkpts/config.json'      # Mettez le vrai nom de votre json
HIFIGAN_CHECKPT = './checkpts/generator_v1' # Mettez le vrai nom de votre .pt

def get_latest_checkpoint(logs_dir="lightning_logs"):
    """Trouve automatiquement le dernier fichier .ckpt pour avoir les derniers poids."""
    import glob
    # Cherche récursivement tous les .ckpt
    files = glob.glob(f"{logs_dir}/**/*.ckpt", recursive=True)
    if not files:
        raise FileNotFoundError("Aucun checkpoint trouvé ! As-tu lancé l'entraînement ?")
    # Trie par date de modification (le plus récent en dernier)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Checkpoint trouvé : {latest_file}")
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
    
    print(f"Génération en {n_steps} étapes...")
    
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
    print("Chargement du modèle...")
    
    # On charge le modèle et ses hyperparamètres
    model = MatchaTTS.load_from_checkpoint(ckpt)
    model.to(DEVICE)
    model.eval() # Mode évaluation (désactive le dropout)

    # --- CHARGEMENT DE HIFI-GAN ---
    print("Chargement de HiFi-GAN...")
    
    # 1. On lit la configuration
    with open(HIFIGAN_CONFIG) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config) # Utilisation de la classe AttrDict de env.py
    
    # 2. On initialise le générateur
    vocoder = HiFiGAN(h).to(DEVICE)
    
    # 3. On charge les poids
    # Note: Parfois le checkpoint contient tout un dictionnaire, parfois juste les poids.
    # Le code officiel fait state_dict['generator']
    state_dict_g = torch.load(HIFIGAN_CHECKPT, map_location=DEVICE)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    vocoder.eval()
    vocoder.remove_weight_norm() # Nettoyage pour l'inférence
    print("HiFi-GAN chargé !")
    # ------------------------------

    # 2. Préparation du texte
    print(f"Texte : '{TEXTE_A_DIRE}'")
    sequence = text_to_sequence(TEXTE_A_DIRE, ["english_cleaners"]) # Ou basic_cleaners
    x = torch.tensor([sequence], dtype=torch.long, device=DEVICE)
    x_lengths = torch.tensor([len(sequence)], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        # 3. Utilisation de la méthode synthesise du modèle pour le processus d'inférence complet
        # Cela gère automatiquement l'encodage, l'alignement et la génération
        output = model.synthesise(
            x=x,
            x_lengths=x_lengths,
            n_timesteps=50,
            temperature=1.0,
            length_scale=1.0
        )
        if "mel" in output:
            mel = output["mel"]
        else:
        # Récupération du mel spectrogramme généré
            mel = output["decoder_outputs"]  # C'est le mel déjà dénormalisé

    print("Synthèse audio avec HiFi-GAN...")
    with torch.no_grad():
        # Le modèle attend un tenseur de forme [Batch, Canaux, Temps]
        # output['mel'] est déjà [1, 80, T], c'est parfait.
        
        # Génération
        audio = vocoder(mel)
        
        # Le résultat est [1, 1, T_audio], on veut juste [T_audio] ou [1, T_audio] pour sauvegarder
        audio = audio.squeeze()
        
        # Normalisation pour éviter la saturation (clipping)
        audio = audio.clamp(-1, 1)
        
        # Conversion en CPU pour la sauvegarde
        audio_cpu = audio.cpu()

    # 6. Sauvegarde
    save_path = os.path.join(OUTPUT_FOLDER, "matcha_hifigan_result.wav")
    
    # ATTENTION : Utilisez la fréquence d'échantillonnage de HiFi-GAN (souvent 22050 ou 24000)
    # Elle est stockée dans la config 'h.sampling_rate'
    torchaudio.save(save_path, audio_cpu.unsqueeze(0), sample_rate=h.sampling_rate)
    print(f"Audio sauvegardé dans : {save_path}")

    # Optionnel: Afficher le spectrogramme
    plot_data = mel.squeeze().cpu().numpy()
    
    print(f"Statistiques du spectrogramme mel:")
    print(f"   Min: {plot_data.min():.4f}, Max: {plot_data.max():.4f}")
    print(f"   Moyenne: {plot_data.mean():.4f}, Écart-type: {plot_data.std():.4f}")
    
    # Si les valeurs sont négatives, c'est en espace log, il faut exp
    if plot_data.min() < 0:
        print("   Valeurs négatives détectées, application de la transformation exp...")
        plot_data = np.exp(plot_data)
        print(f"   Après exp - Min: {plot_data.min():.4f}, Max: {plot_data.max():.4f}")
    
    # Utiliser l'échelle dB pour améliorer le contraste - couramment utilisé dans les articles
    # dB = 20 * log10(valeur), éviter log(0)
    eps = 1e-10
    plot_data_db = 20 * np.log10(plot_data + eps)
    
    # Définir une plage dB raisonnable (typiquement -80dB à 0dB ou plus)
    db_min = np.percentile(plot_data_db, 1)
    db_max = np.percentile(plot_data_db, 99)
    # Si db_max est trop petit, utiliser le max réel
    if db_max < -10:
        db_max = plot_data_db.max()
    
    print(f"   Plage dB: {db_min:.2f} dB à {db_max:.2f} dB")
    
    # Sauvegarder le spectrogramme mel (en utilisant l'échelle dB)
    plt.figure(figsize=(12, 6))
    img = plt.imshow(plot_data_db, origin='lower', aspect='auto', cmap='viridis',
                     vmin=db_min, vmax=db_max, interpolation='bilinear')
    plt.title("Mel Spectrogramme Généré")
    plt.xlabel("Temps (Frames)")
    plt.ylabel("Bins de Fréquence Mel")
    cbar = plt.colorbar(img, label='Intensité (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "mel_spectrogram.png"), dpi=150, bbox_inches='tight')
    print("Mel Spectrogramme sauvegardé.")

if __name__ == "__main__":
    main()