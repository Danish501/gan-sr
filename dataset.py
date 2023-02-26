# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:13:03 2023

@author: Danish
"""

import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob 


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        # self.data = glob.glob("challengedataset/train/"+"*.*")
        self.root_dir = root_dir
        self.data = glob.glob(root_dir+"*.*")
        

        # for index, name in enumerate(self.class_names):
        #     files = os.listdir(os.path.join(root_dir, name))
        #     self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # img_file, label = self.data[index]
        # root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(self.data[index]))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="challengedataset/train/")
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()