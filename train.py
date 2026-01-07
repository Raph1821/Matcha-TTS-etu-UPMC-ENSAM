import os
import glob
import pytorch_lightning as pl
from matcha.data_management.ljspeech_datamodule import LJSpeechDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.text_to_ID.symbols import symbols
from pathlib import Path


def find_latest_checkpoint(logs_dir="lightning_logs", checkpoint_path=None):
    """
    Recherche le fichier checkpoint le plus récent.
    
    Args:
        logs_dir: Chemin du répertoire lightning_logs
        checkpoint_path: Chemin de checkpoint spécifié (optionnel)
    
    Returns:
        Chemin du checkpoint ou None si aucun n'est trouvé
    """
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"Utilisation du checkpoint spécifié: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"Le checkpoint spécifié n'existe pas: {checkpoint_path}, démarrage depuis le début")
            return None
    
    if not os.path.exists(logs_dir):
        print(f"Le répertoire {logs_dir} n'existe pas, démarrage depuis le début")
        return None
    
    pattern = os.path.join(logs_dir, "**", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"Aucun fichier checkpoint trouvé, démarrage depuis le début")
        return None
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"Checkpoint le plus récent trouvé: {latest_file}")
    print(f"   Taille du fichier: {os.path.getsize(latest_file) / (1024**2):.2f} MB")
    return latest_file


def main(checkpoint_path=None, resume_from_latest=True):
    """
    Fonction principale d'entraînement.
    
    Args:
        checkpoint_path: Chemin de checkpoint optionnel (None pour recherche automatique)
        resume_from_latest: Si True, reprend l'entraînement depuis le dernier checkpoint
    """
    PROJECT_ROOT = Path(__file__).resolve().parent
    data_dir = PROJECT_ROOT / "data" / "LJSpeech-1.1"

    ckpt_path = None
    if resume_from_latest:
        ckpt_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)
    
    data_module = LJSpeechDataModule(
        data_dir=data_dir, 
        batch_size=16, 
        num_workers=4,
        pin_memory=False,
        persistent_workers=False
    )

    if ckpt_path is not None:
        print("Chargement du modèle depuis le checkpoint...")
        model = MatchaTTS.load_from_checkpoint(ckpt_path)
        print(f"   État d'entraînement restauré (epoch, step, etc. seront restaurés automatiquement)")
    else:
        print("Initialisation d'un nouveau modèle...")
        model = MatchaTTS(
            n_vocab=len(symbols),
            out_channels=80,
            hidden_channels=192
        )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="loss/val",
                mode="min",
                save_top_k=3,
                filename="best-{epoch:02d}-{loss/val:.3f}",
            ),
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                filename="last-{epoch:02d}-{step}",
            ),
        ],
    )

    if ckpt_path is not None:
        print("Reprise de l'entraînement...")
    else:
        print("Démarrage de l'entraînement Matcha-TTS depuis le début...")
    
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    print("Entraînement terminé !")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement du modèle Matcha-TTS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin du checkpoint (optionnel, recherche automatique du plus récent si non spécifié)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ne pas reprendre depuis un checkpoint, forcer le démarrage depuis le début"
    )
    
    args = parser.parse_args()
    
    resume_from_latest = not args.no_resume
    
    main(
        checkpoint_path=args.checkpoint,
        resume_from_latest=resume_from_latest
    )