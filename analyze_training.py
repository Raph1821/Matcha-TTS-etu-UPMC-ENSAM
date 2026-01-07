import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import shutil  # Import nécessaire pour supprimer des dossiers récursivement
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.ticker as ticker

def analyze_training(log_dir="lightning_logs", output_dir="training_analysis"):
    # --- 1. Préparation ---
    if not os.path.exists(log_dir):
        print(f"Erreur : Le dossier {log_dir} est introuvable.")
        return

    # --- MODIFICATION : Suppression du dossier de sortie s'il existe ---
    if os.path.exists(output_dir):
        print(f"Suppression de l'ancien dossier '{output_dir}'...")
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"Erreur lors de la suppression : {e}")
            return
    # -----------------------------------------------------------------

    # Création du dossier de sortie
    os.makedirs(output_dir)
    print(f"Dossier '{output_dir}' créé.")

    # Récupération et tri des dossiers de version
    version_dirs = glob.glob(os.path.join(log_dir, "version_*"))
    if not version_dirs:
        print("Aucun historique d'entraînement trouvé.")
        return
    
    version_dirs.sort(key=os.path.getmtime)
    print(f"Analyse de {len(version_dirs)} sessions d'entraînement...")

    # --- 2. Configuration des Métriques ---
    # Structure pour le Graphique (Train + Val)
    plot_config = {
        'Duration Loss': {
            'train': ['sub_loss/train_dur_loss_epoch', 'sub_loss/train_dur_loss'],
            'val':   ['sub_loss/val_dur_loss_epoch', 'sub_loss/val_dur_loss']
        },
        'Prior Loss': {
            'train': ['sub_loss/train_prior_loss_epoch', 'sub_loss/train_prior_loss'],
            'val':   ['sub_loss/val_prior_loss_epoch', 'sub_loss/val_prior_loss']
        },
        'Diffusion Loss': {
            'train': ['sub_loss/train_diff_loss_epoch', 'sub_loss/train_diff_loss'],
            'val':   ['sub_loss/val_diff_loss_epoch', 'sub_loss/val_diff_loss']
        }
    }

    # Structure pour le CSV (Train uniquement, comme demandé précédemment)
    csv_map = {
        'Epoch': 'epoch',
        'Total Loss': 'loss/train_epoch',
        'Duration Loss': 'sub_loss/train_dur_loss_epoch',
        'Prior Loss': 'sub_loss/train_prior_loss_epoch',
        'Diffusion Loss': 'sub_loss/train_diff_loss_epoch'
    }

    # Stockage des données
    # plot_data structure: {'Duration Loss': {'train': {epoch: val}, 'val': {epoch: val}}, ...}
    plot_data = {k: {'train': {}, 'val': {}} for k in plot_config.keys()}
    
    # csv_data structure: liste de dicts pour DataFrame
    raw_csv_rows = []

    # --- 3. Extraction des Données ---
    for v_dir in version_dirs:
        event_files = glob.glob(os.path.join(v_dir, "events.out.tfevents.*"))
        for evt_file in event_files:
            try:
                ea = EventAccumulator(evt_file)
                ea.Reload()
                tags = ea.Tags()['scalars']

                # On a besoin de la métrique 'epoch' pour synchroniser
                if 'epoch' not in tags:
                    continue

                # Mapping Step -> Epoch pour ce fichier
                step_to_epoch = {}
                for e in ea.Scalars('epoch'):
                    step_to_epoch[e.step] = int(e.value)

                # --- A. Remplissage pour le Graphique (Train & Val) ---
                for metric_name, keys in plot_config.items():
                    # Train
                    valid_train = next((k for k in keys['train'] if k in tags), None)
                    if valid_train:
                        for e in ea.Scalars(valid_train):
                            ep = get_epoch_from_step(e.step, step_to_epoch)
                            if ep is not None:
                                plot_data[metric_name]['train'][ep] = e.value
                    
                    # Val
                    valid_val = next((k for k in keys['val'] if k in tags), None)
                    if valid_val:
                        for e in ea.Scalars(valid_val):
                            ep = get_epoch_from_step(e.step, step_to_epoch)
                            if ep is not None:
                                plot_data[metric_name]['val'][ep] = e.value

                # --- B. Remplissage pour le CSV (Train global) ---
                # On prépare un dict par époque trouvée dans ce fichier
                file_epochs = {} # epoch -> {col: val}
                
                # Initialisation des lignes avec l'epoch
                for e in ea.Scalars('epoch'):
                    ep = int(e.value)
                    if ep not in file_epochs:
                        file_epochs[ep] = {'Epoch': ep, 'Version': os.path.basename(v_dir)}

                # Récupération des valeurs
                for col_name, tag_key in csv_map.items():
                    if col_name == 'Epoch': continue
                    
                    if tag_key in tags:
                        for e in ea.Scalars(tag_key):
                            ep = get_epoch_from_step(e.step, step_to_epoch)
                            if ep is not None and ep in file_epochs:
                                file_epochs[ep][col_name] = e.value
                
                # Ajout à la liste globale
                for ep_data in file_epochs.values():
                    if len(ep_data) > 2: # Epoch + Version + au moins 1 loss
                        raw_csv_rows.append(ep_data)

            except Exception as e:
                print(f"Erreur lecture {os.path.basename(evt_file)}: {e}")

    # --- 4. Génération du Graphique ---
    print("Génération du graphique...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"Historique d'entraînement (Train vs Validation)", fontsize=16)
    
    colors = {'Duration Loss': '#1f77b4', 'Prior Loss': '#ff7f0e', 'Diffusion Loss': '#2ca02c'}

    for i, (metric_name, data) in enumerate(plot_data.items()):
        ax = axes[i]
        base_color = colors.get(metric_name, 'black')

        # Train
        train_epochs = sorted(data['train'].keys())
        train_values = [data['train'][e] for e in train_epochs]
        if train_epochs:
            ax.plot(train_epochs, train_values, label='Train', color=base_color, marker='o', markersize=3, alpha=0.7)
            ax.text(train_epochs[-1], train_values[-1], f"{train_values[-1]:.4f}", fontsize=8, color=base_color, fontweight='bold')

        # Val
        val_epochs = sorted(data['val'].keys())
        val_values = [data['val'][e] for e in val_epochs]
        if val_epochs:
            ax.plot(val_epochs, val_values, label='Validation', color='red', linestyle='--', marker='x', markersize=4, alpha=0.9)
            ax.text(val_epochs[-1], val_values[-1], f"{val_values[-1]:.4f}", fontsize=8, color='red', fontweight='bold', ha='right')

        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Loss Moyenne")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Epochs")
    axes[-1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Force epoch entier

    plot_path = os.path.join(output_dir, "losses_plot.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # --- 5. Génération du CSV ---
    print("Génération du tableau CSV...")
    if raw_csv_rows:
        df = pd.DataFrame(raw_csv_rows)
        # Nettoyage
        df = df.sort_values(by=['Epoch', 'Version'])
        df = df.drop_duplicates(subset=['Epoch'], keep='last')
        
        # Sélection des colonnes
        cols = ['Epoch', 'Total Loss', 'Duration Loss', 'Prior Loss', 'Diffusion Loss']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        csv_path = os.path.join(output_dir, "losses_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSuccès ! Fichiers générés dans '{output_dir}':")
        print(f" - {plot_path}")
        print(f" - {csv_path}")
    else:
        print("Attention : Pas assez de données pour générer le CSV.")

def get_epoch_from_step(step, step_to_epoch):
    """Helper pour trouver l'époque correspondante à un step"""
    if step in step_to_epoch:
        return step_to_epoch[step]
    if not step_to_epoch:
        return None
    # Recherche du step le plus proche
    closest_step = min(step_to_epoch.keys(), key=lambda x: abs(x - step))
    if abs(closest_step - step) < 100: # Tolérance
        return step_to_epoch[closest_step]
    return None

if __name__ == "__main__":
    analyze_training()
