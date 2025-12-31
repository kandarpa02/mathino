import os
import gdown
import numpy as np

class dataset:
    def __init__(self, path, link):
        if not os.path.exists(path):
            gdown.download(link, path, fuzzy=True)

        data = np.load(path)
        self.keys = list(data.keys())
        self.arrays = [data[k] for k in self.keys]

        n = len(self.arrays[0])
        for a in self.arrays:
            assert len(a) == n

        self.length = n

    def __len__(self):
        return self.length

    def load_data(self):
        return self.arrays


