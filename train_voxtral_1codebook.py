"""
Voxtral Single-Codebook Codec - Training on MGB2 Arabic (Colab)
================================================================
Paste each cell block into Colab. TensorBoard logs audio comparisons.
"""

# ============================================================
# CELL 1: Install (run this cell first!)
# ============================================================
# !pip install -q torch torchaudio transformers datasets soundfile librosa einops tensorboard

# ============================================================
# CELL 2: Setup & Imports
# ============================================================
import os, sys, time, random, math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import numpy as np

# Add parent dir so voxtral_1cb package is importable
# In Colab, upload the voxtral_1cb/ folder or clone from your repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")

from voxtral_1cb import (SingleCodebookCodec, MultiResolutionDiscriminator,
                          reconstruction_loss, stft_magnitude_loss,
                          feature_matching_loss, discriminator_loss,
                          generator_adversarial_loss)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 3: Streaming MGB2 Arabic Dataset
# ============================================================
class StreamingMGB2(torch.utils.data.IterableDataset):
    def __init__(self, segment_samples=96000, target_sr=24000):
        self.segment_samples = segment_samples
        self.target_sr = target_sr

    def __iter__(self):
        ds = load_dataset("MohamedRashad/mgb2-arabic", split="train",
                          streaming=True, trust_remote_code=True)
        ds = ds.shuffle(seed=random.randint(0, 2**31), buffer_size=1000)
        for sample in ds:
            try:
                audio = sample["audio"]
                wav = torch.tensor(audio["array"], dtype=torch.float32)
                sr = audio["sampling_rate"]
                if sr != self.target_sr:
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.target_sr).squeeze(0)
                if wav.shape[0] < self.target_sr:  # skip < 1s
                    continue
                T = wav.shape[0]
                if T >= self.segment_samples:
                    start = random.randint(0, T - self.segment_samples)
                    wav = wav[start:start + self.segment_samples]
                else:
                    wav = F.pad(wav, (0, self.segment_samples - T))
                peak = wav.abs().max()
                if peak > 1e-6:
                    wav = wav / peak
                yield wav.unsqueeze(0)  # (1, T)
            except Exception:
                continue

# ============================================================
# CELL 4: TensorBoard Audio Logger
# ============================================================
class AudioLogger:
    def __init__(self, log_dir="./runs/voxtral_1cb", sample_rate=24000):
        self.writer = SummaryWriter(log_dir)
        self.sr = sample_rate
        print(f"TensorBoard log dir: {log_dir}")
        print(f"Run: tensorboard --logdir {log_dir}")

    def log_scalars(self, log_dict, step):
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, step)

    def log_audio(self, tag, audio, step):
        """audio: (1, T) or (T,) tensor"""
        if audio.dim() == 2: audio = audio.squeeze(0)
        audio = audio.detach().cpu().float()
        peak = audio.abs().max()
        if peak > 0: audio = audio / peak
        self.writer.add_audio(tag, audio.unsqueeze(0), step, self.sr)

    def log_reconstruction(self, x_real, x_hat, step, n=2):
        """Log original vs reconstructed audio for comparison."""
        B = min(x_real.shape[0], n)
        for i in range(B):
            self.log_audio(f"audio/original_{i}", x_real[i], step)
            self.log_audio(f"audio/reconstructed_{i}", x_hat[i], step)

    def log_codebook_usage(self, indices, codebook_size, step):
        unique = indices.unique().numel()
        utilization = unique / codebook_size * 100
        self.writer.add_scalar("codebook/utilization_pct", utilization, step)
        self.writer.add_scalar("codebook/unique_codes", unique, step)
        # Histogram of code usage
        counts = torch.bincount(indices.reshape(-1), minlength=codebook_size).float()
        self.writer.add_histogram("codebook/usage_histogram", counts[counts > 0], step)

    def close(self):
        self.writer.close()

# ============================================================
# CELL 5: Training Function
# ============================================================
def train(
    max_steps=100_000,
    batch_size=2,
    segment_sec=4.0,
    lr_g=3e-4,
    lr_d=1e-4,
    hidden_dim=512,
    latent_dim=256,
    codebook_size=8192,
    log_every=50,
    save_every=5000,
    audio_log_every=500,
    save_dir="./checkpoints_1cb",
    log_dir="./runs/voxtral_1cb",
    disc_start_step=10_000,
    w_feat=1.0,
    w_vq=0.1,
    w_adv=0.1,
    rec_decay_steps=50_000.0,
    resume_path=None,
):
    os.makedirs(save_dir, exist_ok=True)
    target_sr = 24000
    segment_samples = int(segment_sec * target_sr)

    # --- Model ---
    model = SingleCodebookCodec(
        hidden_dim=hidden_dim, latent_dim=latent_dim,
        codebook_size=codebook_size, sample_rate=target_sr,
    ).to(DEVICE)
    print(model.info())

    disc = MultiResolutionDiscriminator().to(DEVICE)
    print(f"Disc: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M params")

    # --- Optimizers ---
    opt_g = optim.AdamW(model.parameters(), lr=lr_g, betas=(0.8, 0.99), weight_decay=1e-4)
    opt_d = optim.AdamW(disc.parameters(), lr=lr_d, betas=(0.8, 0.99), weight_decay=1e-4)

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    # --- Resume ---
    step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "disc" in ckpt: disc.load_state_dict(ckpt["disc"])
        if "opt_g" in ckpt: opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt: opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt.get("step", 0)
        print(f"Resumed from step {step}")

    # --- Data ---
    ds = StreamingMGB2(segment_samples=segment_samples, target_sr=target_sr)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=2,
        pin_memory=True, drop_last=True, prefetch_factor=2)

    # --- TensorBoard ---
    logger = AudioLogger(log_dir=log_dir, sample_rate=target_sr)

    # --- Train ---
    running = {}
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Training {max_steps} steps | batch={batch_size} | seg={segment_sec}s")
    print(f"Disc starts at step {disc_start_step}")
    print(f"{'='*60}\n")

    for batch in loader:
        if step >= max_steps:
            break

        x_real = batch.to(DEVICE)

        # === Generator ===
        model.train()
        opt_g.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x_real)
            x_hat, vq_loss, indices = out["x_hat"], out["vq_loss"], out["indices"]

            disc.eval()
            with torch.no_grad():
                dr = disc(x_real)
            df = disc(x_hat)
            fmaps_real = [fm for _, fm in dr]
            fmaps_fake = [fm for _, fm in df]
            logits_fake = [lg for lg, _ in df]

            l_rec, rec_w = reconstruction_loss(x_real, x_hat, step, decay_steps=rec_decay_steps)
            l_stft = rec_w * stft_magnitude_loss(x_real, x_hat)
            l_feat = w_feat * feature_matching_loss(fmaps_real, fmaps_fake)
            l_vq = w_vq * vq_loss
            l_adv = w_adv * generator_adversarial_loss(logits_fake) if step >= disc_start_step else x_real.new_zeros(())

            g_loss = l_rec + l_stft + l_feat + l_vq + l_adv
            
        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        # extract scalar values for logging
        g_loss_val = g_loss.item()
        l_rec_val = l_rec.item()
        l_stft_val = l_stft.item()
        l_feat_val = l_feat.item()
        l_vq_val = l_vq.item()
        l_adv_val = l_adv.item() if torch.is_tensor(l_adv) else l_adv

        # free memory
        del out, dr, df, fmaps_real, fmaps_fake, logits_fake, g_loss, l_rec, l_stft, l_feat, l_vq, l_adv

        # === Discriminator ===
        d_loss_val = 0.0
        if step >= disc_start_step:
            disc.train()
            opt_d.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                dr2 = disc(x_real)
                df2 = disc(x_hat.detach())
                d_loss = discriminator_loss([l for l,_ in dr2], [l for l,_ in df2])
                
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()
            d_loss_val = d_loss.item()
            del dr2, df2, d_loss
            
        # === Logging ===
        unique_codes = indices.unique().numel()
        log = {"loss/total": g_loss_val, "loss/l1": l_rec_val,
               "loss/stft": l_stft_val, "loss/feat": l_feat_val,
               "loss/vq": l_vq_val, "loss/adv_g": l_adv_val,
               "loss/disc": d_loss_val, "loss/rec_weight": rec_w,
               "codebook/utilization_pct": unique_codes / codebook_size * 100}

        for k, v in log.items():
            running[k] = running.get(k, 0) + v

        step += 1

        # Console + TensorBoard scalars
        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = {k: v / log_every for k, v in running.items()}
            logger.log_scalars(avg, step)
            parts = [f"{k.split('/')[-1]}={v:.4f}" for k, v in avg.items()]
            print(f"[{step:7d}] {' | '.join(parts)} | {log_every/max(elapsed,1e-6):.1f} it/s")
            running = {}
            t0 = time.time()

        # TensorBoard audio comparison
        if step % audio_log_every == 0:
            model.eval()
            with torch.no_grad():
                logger.log_reconstruction(x_real, x_hat, step, n=2)
                logger.log_codebook_usage(indices, codebook_size, step)
            model.train()

        # Save checkpoint
        if step % save_every == 0:
            path = os.path.join(save_dir, f"ckpt_{step:07d}.pt")
            torch.save({"step": step, "model": model.state_dict(),
                        "disc": disc.state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict()}, path)
            print(f"  Saved: {path}")

    # Final save
    path = os.path.join(save_dir, f"ckpt_final_{step:07d}.pt")
    torch.save({"step": step, "model": model.state_dict()}, path)
    logger.close()
    print(f"\nDone! Final: {path}")
    return model

# ============================================================
# CELL 6: Run!
# ============================================================
if __name__ == "__main__":
    model = train(
        max_steps=100_000,
        batch_size=4,
        segment_sec=4.0,
        hidden_dim=512,
        latent_dim=256,
        codebook_size=8192,
        log_every=50,
        save_every=5000,
        audio_log_every=500,
        disc_start_step=10_000,
    )
