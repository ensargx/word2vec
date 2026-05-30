from pathlib import Path
import signal
import sys
import time
from datetime import datetime

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.use_amp = cfg.training.use_amp and self.device.type == "cuda"
        self.grad_clip = cfg.training.grad_clip
        self.log_interval = cfg.training.log_interval

        self.scaler = GradScaler(enabled=self.use_amp)

        self.ckpt_dir = Path(cfg.paths.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{cfg.paths.tensorboard_dir}/{run_name}")

        self.stop_requested = False
        self.force_quit_requested = False

        self.state = {
            "status": "idle",
            "epoch": 0,
            "step": 0,
            "global_step": 0,
            "loss": None,
            "avg_loss": None,
            "steps_per_sec": None,
            "eta_seconds": None,
            "eta_minutes": None,
            "eta_hours": None,
        }

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, *_):
        if not self.stop_requested:
            self.stop_requested = True
            self.state["status"] = "stop_requested"
            return

        self.force_quit_requested = True
        self.state["status"] = "force_quit_requested"
        self.writer.flush()
        self.writer.close()
        sys.exit(130)

    def _raw_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    @property
    def checkpoint_path(self):
        return self.ckpt_dir / "latest.pt"

    def save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "global_step": self.state["global_step"],
            "state": self.state,
            "model": self._raw_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return 0

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self._raw_model().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.state.update(ckpt.get("state", {}))
        self.state["global_step"] = ckpt.get("global_step", self.state.get("global_step", 0))

        return ckpt["epoch"] + 1

    def _estimate_total_steps(self, loader, start_epoch):
        try:
            steps_per_epoch = len(loader)
        except TypeError:
            return None, None

        remaining_epochs = self.cfg.training.epochs - start_epoch
        total_remaining_steps = steps_per_epoch * remaining_epochs

        return steps_per_epoch, total_remaining_steps

    def _update_state(self, epoch, step, loss, avg_loss, steps_per_sec, eta_seconds):
        self.state.update({
            "status": "training",
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "avg_loss": avg_loss,
            "steps_per_sec": steps_per_sec,
            "eta_seconds": eta_seconds,
            "eta_minutes": eta_seconds / 60 if eta_seconds is not None else None,
            "eta_hours": eta_seconds / 3600 if eta_seconds is not None else None,
        })

    def _write_tensorboard(self):
        global_step = self.state["global_step"]

        self.writer.add_scalar("train/loss", self.state["loss"], global_step)
        self.writer.add_scalar("train/avg_loss", self.state["avg_loss"], global_step)
        self.writer.add_scalar("train/epoch", self.state["epoch"], global_step)
        self.writer.add_scalar("train/step", self.state["step"], global_step)
        self.writer.add_scalar("train/steps_per_sec", self.state["steps_per_sec"], global_step)

        if self.state["eta_seconds"] is not None:
            self.writer.add_scalar("train/eta_seconds", self.state["eta_seconds"], global_step)
            self.writer.add_scalar("train/eta_minutes", self.state["eta_minutes"], global_step)
            self.writer.add_scalar("train/eta_hours", self.state["eta_hours"], global_step)

        self.writer.add_text("train/status", "\n".join(f"{k}: {v}" for k, v in self.state.items()), global_step)
        self.writer.flush()

    def fit(self, loader):
        start_epoch = self.load_checkpoint()
        steps_per_epoch, total_remaining_steps = self._estimate_total_steps(loader, start_epoch)

        self.state["status"] = "running"

        for epoch in range(start_epoch, self.cfg.training.epochs):
            self.model.train()

            running_loss = 0.0
            n_batches = 0
            epoch_started_at = time.time()

            for step, (pos_u, pos_v) in enumerate(loader):
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

                loss_val = float(loss.detach().cpu())
                running_loss += loss_val
                n_batches += 1

                elapsed = time.time() - epoch_started_at
                steps_per_sec = n_batches / max(elapsed, 1e-9)

                eta_seconds = None
                if steps_per_epoch is not None:
                    current_remaining_epochs = self.cfg.training.epochs - epoch - 1
                    remaining_steps = max((steps_per_epoch - step - 1) + (current_remaining_epochs * steps_per_epoch), 0)
                    eta_seconds = remaining_steps / max(steps_per_sec, 1e-9)

                self._update_state(epoch, step, loss_val, running_loss / n_batches, steps_per_sec, eta_seconds)

                if step % self.log_interval == 0:
                    self._write_tensorboard()

                self.state["global_step"] += 1

                if self.force_quit_requested:
                    sys.exit(130)

            epoch_loss = running_loss / max(n_batches, 1)

            self.writer.add_scalar("epoch/loss", epoch_loss, epoch)
            self.writer.add_scalar("epoch/batches", n_batches, epoch)
            self.writer.add_scalar("epoch/duration_seconds", time.time() - epoch_started_at, epoch)

            self.state["status"] = "epoch_finished"
            self.save_checkpoint(epoch)
            self.writer.flush()

            if self.stop_requested:
                self.state["status"] = "stopped"
                self.save_checkpoint(epoch)
                break

        if not self.stop_requested:
            self.state["status"] = "finished"

        self.writer.add_text("train/final_status", "\n".join(f"{k}: {v}" for k, v in self.state.items()), self.state["global_step"])
        self.writer.flush()
        self.writer.close()
