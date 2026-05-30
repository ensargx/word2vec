import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class SkipGramDataset(IterableDataset):
    def __init__(self, data_idxs, window_size=5):
        self.data = data_idxs
        self.window_size = window_size
        self.length = self._estimate_len()

    def _estimate_len(self):
        total = 0
        avg_window = (self.window_size + 1) / 2

        for sentence in self.data:
            n = len(sentence)
            if n < 2:
                continue

            total += int(n * 2 * avg_window)

        return total

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.data)
        else:
            per_worker = int(np.ceil(len(self.data) / worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))

        for sentence in self.data[iter_start:iter_end]:
            sentence = np.asarray(sentence, dtype=np.int64).reshape(-1)
            sent_len = len(sentence)

            if sent_len < 2:
                continue

            for i in range(sent_len):
                target_idx = int(sentence[i])
                window = np.random.randint(1, self.window_size + 1)

                start = max(0, i - window)
                end = min(sent_len, i + window + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    yield target_idx, int(sentence[j])