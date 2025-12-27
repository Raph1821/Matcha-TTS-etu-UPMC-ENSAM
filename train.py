import os
import glob
import pytorch_lightning as pl
from matcha.data_management.ljspeech_datamodule import LJSpeechDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.text_to_ID.symbols import symbols
from pathlib import Path


def find_latest_checkpoint(logs_dir="lightning_logs", checkpoint_path=None):
    """
    查找最新的 checkpoint 文件
    
    Args:
        logs_dir: lightning_logs 目录路径
        checkpoint_path: 如果指定，直接使用这个路径
    
    Returns:
        checkpoint 路径，如果不存在则返回 None
    """
    # 如果明确指定了 checkpoint 路径
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"✅ 使用指定的 checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"⚠️  指定的 checkpoint 不存在: {checkpoint_path}，将从头开始训练")
            return None
    
    # 自动查找最新的 checkpoint
    if not os.path.exists(logs_dir):
        print(f"ℹ️  {logs_dir} 目录不存在，将从头开始训练")
        return None
    
    # 递归查找所有 .ckpt 文件
    pattern = os.path.join(logs_dir, "**", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"ℹ️  未找到 checkpoint 文件，将从头开始训练")
        return None
    
    # 按修改时间排序，选择最新的
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ 找到最新的 checkpoint: {latest_file}")
    print(f"   文件大小: {os.path.getsize(latest_file) / (1024**2):.2f} MB")
    return latest_file


def main(checkpoint_path=None, resume_from_latest=True):
    """
    主训练函数
    
    Args:
        checkpoint_path: 可选的 checkpoint 路径，如果为 None 则自动查找
        resume_from_latest: 是否从最新的 checkpoint 恢复训练（如果存在）
    """
    # 1. Configuration des chemins
    PROJECT_ROOT = Path(__file__).resolve().parent
    data_dir = PROJECT_ROOT / "data" / "LJSpeech-1.1"

    # 2. 查找 checkpoint（如果需要）
    ckpt_path = None
    if resume_from_latest:
        ckpt_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)
    
    # 3. Initialisation du DataModule
    data_module = LJSpeechDataModule(
        data_dir=data_dir, 
        batch_size=16, 
        num_workers=4
    )

    # 4. Initialisation du Modèle
    # 如果从 checkpoint 恢复，模型会自动从 checkpoint 加载
    if ckpt_path is not None:
        print("📦 从 checkpoint 加载模型...")
        model = MatchaTTS.load_from_checkpoint(ckpt_path)
        print(f"   已恢复训练状态（epoch, step 等会自动恢复）")
    else:
        print("🆕 初始化新模型...")
        model = MatchaTTS(
            n_vocab=len(symbols),
            out_channels=80,
            hidden_channels=192
        )

    # 5. Configuration du Trainer（优化配置）
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # 梯度裁剪
        accumulate_grad_batches=2,  # 梯度累积（相当于增大 batch size）
        # 启用自动保存 checkpoint
        enable_checkpointing=True,
        # 保存最佳模型和最新模型
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="loss/val",  # 监控验证损失
                mode="min",
                save_top_k=3,  # 保存最好的 3 个模型
                filename="best-{epoch:02d}-{loss/val:.3f}",
            ),
            pl.callbacks.ModelCheckpoint(
                save_last=True,  # 总是保存最新的模型
                filename="last-{epoch:02d}-{step}",
            ),
        ],
    )

    # 6. Lancement de l'entraînement
    if ckpt_path is not None:
        print("🔄 恢复训练...")
    else:
        print("🚀 从头开始训练 Matcha-TTS...")
    
    # 如果指定了 checkpoint，使用它恢复训练
    # 否则从头开始
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    print("✅ 训练完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练 Matcha-TTS 模型")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="指定 checkpoint 路径（可选，如果不指定则自动查找最新的）"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不从 checkpoint 恢复，强制从头开始训练"
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --no-resume，则不恢复训练
    resume_from_latest = not args.no_resume
    
    main(
        checkpoint_path=args.checkpoint,
        resume_from_latest=resume_from_latest
    )