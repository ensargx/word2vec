import os
import csv
import torch
import signal
import sys
import re
from src.config import DEVICE, CHECKPOINT_DIR

class Trainer:
    def __init__(self, model, optimizer, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.csv_path = os.path.join(CHECKPOINT_DIR, "train_log.csv")
        self.stop_requested = False

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch", "loss"])

    def log_to_csv(self, epoch, batch, loss):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, f"{loss:.6f}"])

    def load_latest_checkpoint(self):
        """Klasörü tarar, en son epoch'u bulur ve yükler."""
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_e") and f.endswith(".pt")]
        if not files:
            print("[*] Kayıtlı model bulunamadı. Sıfırdan başlanıyor.")
            return 0

        # model_e{n}.pt formatındaki n değerlerini çek
        epochs = [int(re.search(r'model_e(\d+)', f).group(1)) for f in files]
        latest_epoch = max(epochs)

        path = os.path.join(CHECKPOINT_DIR, f"model_e{latest_epoch}.pt")
        print(f"[+] En son bulunan model yükleniyor: {path}")

        checkpoint = torch.load(path, map_location=DEVICE)
        # Eğer checkpoint içinde sadece state_dict değil de tüm 'state' varsa onu kullan
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.model.load_state_dict(checkpoint)

        return latest_epoch + 1

    def train(self, loader, start_epoch=0):
        print(f"[*] Eğitim {start_epoch}. epoch üzerinden başlatıldı.")
        epoch = start_epoch

        while True:
            total_loss = 0
            batch_count = 0
            epoch_logs = []

            for i, (pos_u, pos_v, neg_v) in enumerate(loader):
                pos_u, pos_v, neg_v = pos_u.to(DEVICE), pos_v.to(DEVICE), neg_v.to(DEVICE)

                self.optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()

                if i % self.log_interval == 0:
                    current_loss = loss.item()
                    epoch_logs.append((i, current_loss))
                    print(f"E:{epoch} | B:{i} | Loss:{current_loss:.4f}")

                batch_count += 1
                total_loss += loss.item()

            for batch_idx, loss_val in epoch_logs:
                self.log_to_csv(epoch, batch_idx, loss_val)

            avg_loss = total_loss / batch_count
            print(f"\n--- Epoch {epoch} bitti. Ortalama Loss: {avg_loss:.4f} ---")

            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, os.path.join(CHECKPOINT_DIR, f"model_e{epoch}.pt"))

            if self.stop_requested:
                print("[*] Eğitim sonlandırıldı.")
                break

            epoch += 1
