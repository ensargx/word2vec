import os
import csv
import torch
from src.config import DEVICE, CHECKPOINT_DIR

class Trainer:
    def __init__(self, model, optimizer, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval

        self.current_epoch = 0
        self.csv_path = os.path.join(CHECKPOINT_DIR, "train_log.csv")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Dosya yoksa header (başlık) oluşturur."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch", "loss"])

    def log_to_csv(self, epoch, batch, loss):
        """Loss değerini CSV dosyasının sonuna ekler."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, f"{loss:.6f}"])

    def train(self, loader, epochs):
        print(f"[*] Kayıt dosyası: {self.csv_path}")

        for epoch in range(epochs):
            self.current_epoch = epoch

            for i, (pos_u, pos_v, neg_v) in enumerate(loader):
                # GPU/Device Transfer
                pos_u, pos_v, neg_v = pos_u.to(DEVICE), pos_v.to(DEVICE), neg_v.to(DEVICE)

                # Optimizer step
                self.optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()

                if i % self.log_interval == 0:
                    current_loss = loss.item()
                    self.log_to_csv(epoch, i, current_loss)
                    print(f"E:{epoch} | B:{i} | Loss:{current_loss:.4f}")

            torch.save(self.model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_e{epoch}.pt"))
