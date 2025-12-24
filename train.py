import pytorch_lightning as pl
from matcha.data_management.ljspeech_datamodule import LJSpeechDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.text_to_ID.symbols import symbols

def main():
    # 1. Configuration des chemins
    # Utilisez le chemin absolu vers votre dossier LJSpeech-1.1
    data_dir = r"C:\Users\mathi\OneDrive\Documents\1. COURS SORBONNE\Machine Learning\Matcha TTS\Nouveau Matcha-TTS\data\LJSpeech-1.1"

    # 2. Initialisation du DataModule
    data_module = LJSpeechDataModule(
        data_dir=data_dir, 
        batch_size=16, 
        num_workers=4
    )

    # 3. Initialisation du Modèle
    model = MatchaTTS(
        n_vocab=len(symbols),
        out_channels=80,
        hidden_channels=192
    )

    # 4. Configuration du Trainer (Optimisé pour votre GPU)
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",    # On repasse sur GPU maintenant que CUDA est OK
        devices=1,
        precision="32-true", # Plus stable pour commencer
        log_every_n_steps=10
    )

    # 5. Lancement de l'entraînement
    print("🚀 Démarrage de l'entraînement Matcha-TTS...")
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()