import os
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import torch

class SkipGramDataset(IterableDataset):
    def __init__(self, data_idxs, window_size=5):
        self.data = data_idxs 
        self.window_size = window_size

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Tek bir worker varsa verinin tamami
            iter_start = 0
            iter_end = len(self.data)
        else:
            # Birden fazla worker varsa veriyi paylastir
            per_worker = int(np.ceil(len(self.data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))

        # Kendi dilimimiz uzerinde donuyoruz
        for i in range(iter_start, iter_end):
            target_idx = self.data[i]

            # Dinamik window size (Word2Vec standarti)
            window = np.random.randint(1, self.window_size + 1)

            start = max(0, i - window)
            end = min(len(self.data), i + window + 1)

            for j in range(start, end):
                if i == j: continue

                # Tensor donusturme isini burada degil, DataLoader'in 
                # default_collate fonksiyonuna birakmak (sadece ham deger dondurerek) daha hizlidir.
                yield target_idx, self.data[j]

def save_processed_data(data_dict, filename="training_data.pt", data_dir="data"):
    """Sözlük ve eğitim verilerini torch formatında kaydeder."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path = os.path.join(data_dir, filename)
    torch.save(data_dict, path)

def load_processed_data(filename="training_data.pt", data_dir="data"):
    """Torch ile kaydedilmiş paketi geri yükler."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    from src.config import DEVICE
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    return data
