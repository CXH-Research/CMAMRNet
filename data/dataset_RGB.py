import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        mas_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.mas_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mas_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Transpose(p=0.3),
            A.Flip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(p=0.3),
            A.RandomResizedCrop(height=img_options['h'], width=img_options['w']),
            ],
            additional_targets={
                'mask': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        mas_path = self.mas_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        mas_img = Image.open(mas_path).convert('L')

        inp_img = np.array(inp_img)
        mas_img = np.array(mas_img)

        transformed = self.transform(image=inp_img, mask=mas_img)

        inp_img = F.to_tensor(transformed['image'])

        mas_img = F.to_tensor(transformed['mask'])

        inp_img = inp_img * mas_img

        filename = os.path.basename(inp_path)

        return inp_img, mas_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        mas_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.mas_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mas_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            additional_targets={
                'mask': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        mas_path = self.mas_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        mas_img = Image.open(mas_path).convert('L')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            mas_img = np.array(mas_img)

            transformed = self.transform(image=inp_img, mask=mas_img)

            inp_img = transformed['image']
            mas_img = transformed['mask']

        inp_img = F.to_tensor(inp_img)
        mas_img = F.to_tensor(mas_img)

        inp_img = inp_img * mas_img

        filename = os.path.basename(inp_path)

        return inp_img, mas_img, filename
