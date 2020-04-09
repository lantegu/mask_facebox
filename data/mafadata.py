import torch.utils.data as data
import os
import cv2
import torch
import numpy as np
class MafaDetection(data.Dataset):
    def __init__(self,root,preproc=None,target_transform=None):
        self.root =root
        self.preproc = preproc
        self.target_transform = target_transform
        self._annopath = os.path.join(self.root,'train_annotations')
        self._imgpath = os.path.join(self.root, 'train_images')
        self.ids = list()
        with open(os.path.join(self.root, 'exam_label_train.txt'), 'r') as f:
          self.ids = [tuple(line.split()) for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(os.path.join(self._imgpath,img_id[0]), cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        with open(os.path.join(self._annopath,img_id[1]),'rt') as f:
            target_list = []
            for line in f:
                target_list.append(list(map(float, line.split()))[:-1])
            target = np.array(target_list)


        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target
