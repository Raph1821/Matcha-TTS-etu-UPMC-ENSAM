import pytorch_lightning as pl
from matcha.data_management.ljspeech_datamodule import LJSpeechDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.text_to_ID.symbols import symbols
from pathlib import Path

def main():
    # 1. Configuration des chemins
    PROJECT_ROOT = Path(__file__).resolve().parent
    data_dir = PROJECT_ROOT / "data" / "LJSpeech-1.1"

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

    # 4. Configuration du Trainer（优化配置）
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # 梯度裁剪
        accumulate_grad_batches=2,  # 梯度累积（相当于增大 batch size）
    )

    # 5. Lancement de l'entraînement
    print("🚀 Démarrage de l'entraînement Matcha-TTS...")
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()