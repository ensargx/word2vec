import os
import sys
import signal
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        checkpoint_dir,
        use_amp=True,
        grad_clip=None,
        log_interval=100,
        save_each_epoch=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_each_epoch = save_each_epoch
        self.stop_requested = False

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, _sig, _frame):
        if not self.stop_requested:
            print("\n[!] Stop requested. Current epoch will finish, then checkpoint will be saved.")
            self.stop_requested = True
        else:
            print("\n[!] Forced exit.")
            sys.exit(0)

    def _raw_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    @property
    def latest_path(self):
        return self.checkpoint_dir / "latest.pt"

    def save_checkpoint(self, epoch, metrics=None):
        state = {
            "epoch": epoch,
            "model": self._raw_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics or {},
        }

        torch.save(state, self.latest_path)

        if self.save_each_epoch:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

    def load_checkpoint(self):
        if not self.latest_path.exists():
            return 0

        ckpt = torch.load(self.latest_path, map_location=self.device, weights_only=False)

        self._raw_model().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        return ckpt["epoch"] + 1

    def train_epoch(self, loader, epoch, max_epochs):
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs}", leave=True)

        for step, (pos_u, pos_v) in enumerate(pbar):
            pos_u = pos_u.to(self.device, non_blocking=True)
            pos_v = pos_v.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.model(pos_u, pos_v)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._raw_model().parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            total_batches += 1

            if step % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total_loss / total_batches:.4f}")

        return {
            "loss": total_loss / max(total_batches, 1),
            "batches": total_batches,
        }

    def fit(self, loader, epochs, resume=True):
        start_epoch = self.load_checkpoint() if resume else 0

        for epoch in range(start_epoch, epochs):
            metrics = self.train_epoch(loader, epoch, epochs)
            self.save_checkpoint(epoch, metrics)

            print(f"epoch={epoch} loss={metrics['loss']:.4f} batches={metrics['batches']}")

            if self.stop_requested:
                break