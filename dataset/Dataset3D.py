import torch
import os
import torchvision
import numpy as np

class Dataset3D(torch.utils.data.Dataset):
    def __init__(self, root_path, transform=None):
        self.iroot_path = root_path
        self.transform = transform
        self.img_list = [os.path.join(root_path, image) for image in os.listdir(root_path)]


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        image_folder = self.img_list[idx]
        image = np.load(os.path.join(image_folder, "preprocessed_image.npy"))
        label = np.load(os.path.join(image_folder, "segmentation.npy"))
        image = np.transpose(image)
        label = np.transpose(label)
        item_dict= {'image': image, 'label': label}

        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)
        item_dict['image'] = image.unsqueeze(0).expand(1, -1, -1, -1)
        item_dict['label'] = label.unsqueeze(0).expand(1, -1, -1, -1)
        return item_dict['image'], item_dict['label'], image_folder
