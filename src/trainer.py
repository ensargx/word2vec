import os
import csv
import torch
import signal
import sys
import re
from src.config import cfg
from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, optimizer, dim, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.stop_requested = False
        self.scaler = GradScaler()
        self.dim = dim

        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        self._init_csv()
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, _sig, _frame):
        if not self.stop_requested:
            print("\n[!] Durdurma sinyali alındı. Mevcut epoch bitince duracak...")
            self.stop_requested = True
        else:
            print("\n[!] Zorla kapatılıyor...")
            sys.exit(0)

    def _init_csv(self):
        if not os.path.exists(self._get_csv_path()):
            with open(self._get_csv_path(), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch", "loss"])

    def log_to_csv(self, epoch, batch, loss):
        with open(self._get_csv_path(), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, f"{loss:.6f}"])

    def _get_csv_path(self):
        return os.path.join(self._get_dim_path(), "train_log.csv")

    def _get_dim_path(self):
        """boyuta özel klasör yolunu döndürür ve yoksa oluşturur."""
        path = os.path.join(cfg.paths.checkpoint_dir, f"dim_{self.dim}")
        os.makedirs(path, exist_ok=True)
        return path

    def load_latest_checkpoint(self):
        """Klasörü tarar, en son epoch'u bulur ve yükler."""
        dim_dir = self._get_dim_path()
        files = [f for f in os.listdir(dim_dir) if f.startswith("model_e") and f.endswith(".pt")]
        if not files:
            print("[*] Kayıtlı model bulunamadı. Sıfırdan başlanıyor.")
            return 0

        # model_e{n}.pt formatındaki n değerlerini çek
        epochs = [int(re.search(r'model_e(\d+)', f).group(1)) for f in files]
        latest_epoch = max(epochs)
        path = os.path.join(dim_dir, f"model_e{latest_epoch}.pt")
        print(f"[+] yükleniyor: {path}")

        checkpoint = torch.load(path, map_location=cfg.device, weights_only=False)
        state_dict = checkpoint['state_dict']

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        raw_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        raw_model.load_state_dict(state_dict, strict=False)

        return latest_epoch + 1

    def train(self, loader, start_epoch=0, max_epochs=None):
        print(f"[*] Eğitim {start_epoch}. epoch üzerinden başlatıldı.")
        epoch = start_epoch

        while (max_epochs is None or epoch < max_epochs) and not self.stop_requested:
            total_loss = 0
            batch_count = 0
            epoch_logs = []

            for i, (pos_u, pos_v) in enumerate(loader):
                pos_u = pos_u.to(cfg.device, non_blocking=True)
                pos_v = pos_v.to(cfg.device, non_blocking=True)

                self.optimizer.zero_grad()

                with autocast(device_type='cuda' if 'cuda' in str(cfg.device) else 'cpu'):
                    loss = self.model(pos_u, pos_v)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if i % self.log_interval == 0:
                    current_loss = loss.item()
                    epoch_logs.append((i, current_loss))
                    total_ep = max_epochs if max_epochs is not None else "∞"
                    print(f"E:{epoch}/{total_ep} | B:{i} | Loss:{current_loss:.4f}")

                batch_count += 1
                total_loss += loss.item()

            for batch_idx, loss_val in epoch_logs:
                self.log_to_csv(epoch, batch_idx, loss_val)

            avg_loss = total_loss / batch_count if batch_count > 0 else 0

            print(f"\n--- Epoch {epoch} bitti. Ortalama Loss: {avg_loss:.4f} ---")

            raw_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            state = {
                'epoch': epoch,
                'state_dict': raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            save_path = os.path.join(self._get_dim_path(), f"model_e{epoch}.pt")
            torch.save(state, save_path)

            epoch += 1
