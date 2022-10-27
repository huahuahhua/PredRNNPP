import csv
import glob
import os

import numpy as np
from imageio import imread
from torch.utils.data import Dataset
import torch


class Radar(Dataset):
    def __init__(self, root, is_train):
        super(Radar, self).__init__()
        self.root = root
        self.images_path = "Train_Data"
        self.category = os.path.join(root, self.images_path)

        self.path_lists = self.load_csv(filename=self.images_path + ".csv", root=root)  #全部的数据
        self.lenth = len(self.path_lists)

        if is_train:
            self.path_list = self.path_lists[0:int(0.8*self.lenth)]
        else:
            self.path_list = self.path_lists[int(0.8*self.lenth):]

    def __getitem__(self, item):
        ima_pa = self.path_list[item][1:]
        images =[]
        for path in ima_pa:
            Tc_image = self.image_read(path)
            images.append(Tc_image)
        images = torch.stack(images,dim=0)
        return images

    def __len__(self):
        return len(self.path_list)

    def load_csv(self, filename, root):
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            dirs = os.listdir(self.category)
            dirs.sort(key=lambda x: int(x.replace("sample_", "").split('.')[0]))
            with open(os.path.join(root, filename), mode='w', newline='') as f:  #
                writer = csv.writer(f)
                for dir in dirs:
                    images = []
                    images += glob.glob(os.path.join(self.category, dir, '*.png'))
                    images.sort(key=lambda x: int((x.split("\\")[-1]).split(".")[0]))
                    writer.writerow([dir] + images)

        images = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                images.append(row)
        return images

    def image_read(self, sinle_image):
        # print(np.max(np.array(imread(sinle_image))))
        # print(np.min(np.array(imread(sinle_image))))
        image = np.array(imread(sinle_image) / 70.0, dtype=np.float64)
        t_image = torch.from_numpy(image).float()
        tc_image = torch.unsqueeze(t_image, dim=0)
        return tc_image


if __name__ == '__main__':
    da = Radar("../../data",True)
    da.__getitem__(1)