import torch
import os
import numpy as np
from PIL import Image
import PIL

from .dataset import center_crop

class CarsRealTraffic(object):
    """Data Handler that loads cars data."""
    def __init__(self, data_root, train=True, seq_len=1, image_size=64,
                  **kwargs):
        self.root_dir = data_root
        self.image_size = image_size

        if train:
            self.dir = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
                        if f[0] == 'f' if int(f[-3:]) < 550]
            self.dir = self.dir[:550]
        else:
            self.dir = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
                        if f[0] == 'f' if int(f[-3:]) >= 550]

        self.data = []
        for i in range(len(self.dir)):
            dir_name = self.dir[i]
            seq_ims = sorted([os.path.join(dir_name, f) for f in os.listdir(dir_name)
                              if f[-5:] == '.jpeg'])
            for j in range(len(seq_ims)-3*seq_len):
                self.data.append(seq_ims[j:j+2*seq_len:2])

        # print(">>> N: %d" % len(self.data))
        self.N = len(self.data)
        self.seq_len = seq_len

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        images = self.data[index % len(self.data)]

        image_seq = []
        for i in range(len(images)):
            img = center_crop(np.asarray(Image.open(images[i]).convert('RGB')))
            img = Image.fromarray(img, 'RGB').resize((self.image_size, self.image_size), PIL.Image.LANCZOS)
            img = (np.asarray(img).reshape(1, self.image_size, self.image_size, 3).astype('float32') - 127.5) / 127.5

            # img = (np.asarray(Image.open(images[i]).resize((self.image_size,
            #       self.image_size), PIL.Image.LANCZOS)).reshape(1,
            #       self.image_size, self.image_size, 3).astype('float32')
            #       - 127.5) / 127.5

            image_seq.append(torch.from_numpy(img[0, :, :, :]).permute(2, 0, 1))

        return torch.stack(image_seq, 0).squeeze(), 0