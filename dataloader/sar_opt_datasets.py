import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

img_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

label_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])

class Dataset_self(data.Dataset):
    def __init__(self, data_dir):
        super(Dataset_self, self).__init__()

        self.root_path = data_dir
        self.li = ['t1','t2','gt']
        self.filenames = [x for x in sorted(os.listdir(os.path.join(data_dir,self.li[0])))]
        self.img_transform = img_transform
        self.label_transform = label_transform


    def __getitem__(self, index):
        gt_filepath = os.path.join(self.root_path, self.li[2], self.filenames[index])
        t1_filepath = gt_filepath.replace('gt', 't1')
        t2_filepath = gt_filepath.replace('gt', 't2')

        ## without resize
        img1 = self.img_transform(Image.open(t1_filepath).convert('RGB'))
        img2 = self.img_transform(Image.open(t2_filepath).convert('RGB'))

        label = np.array(self.label_transform(Image.open(gt_filepath).convert('L')))
        return img1, img2, label

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    data_A_dir = 'E:/0-code/DSTCD/dataset/3/gloucester'
    img_data = Dataset_self(data_A_dir)
    data_loader_train = torch.utils.data.DataLoader(dataset=img_data,
                                                    batch_size=8,
                                                    shuffle=True)
    for x in data_loader_train:
        image1,image2, gt = x
        print(image1.shape)
        print(image2.shape)
        print(gt.shape)