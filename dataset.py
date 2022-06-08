import os
import random

import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as Transforms

class DoCaSet(Dataset):
    def __init__(self, root_path, category, transforms=None):

        self.category = category

        if category == 'train' or category == 'val' or category == 'test':
            data_root = os.path.join(root_path, category)
        else:
            print("Data category wrong, only train val and test are available.")
        self.data_img_paths = [os.path.join(data_root, img) for img in os.listdir(data_root)]

        if category == 'test':
            # Test images must be sorted
            self.data_img_paths = sorted(self.data_img_paths, key=lambda x:int(x.split('.')[-2].split('/')[-1]))
            self.data_labels = [-1 for data_path in self.data_img_paths] 
        else:
            self.data_labels = [1 if 'dog' == data_path.split('/')[-1].split('.')[0].split('_')[0] else 0 for data_path in self.data_img_paths]

        if transforms is None:
            normalize = Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            self.transforms = {
                'train':
                Transforms.Compose([
                    Transforms.Resize((224,224)),
                    Transforms.RandomHorizontalFlip(p=0.2),
                    Transforms.ToTensor(),
                    normalize
                    ]),
                'val':
                Transforms.Compose([
                    Transforms.Resize((224,224)),
                    Transforms.ToTensor(),
                    normalize
                    ]),
                'test':
                Transforms.Compose([
                    Transforms.Resize((224,224)),
                    Transforms.ToTensor(),
                    ]),
            }   

    def __getitem__(self, index):
        img_path = self.data_img_paths[index]
        label = self.data_labels[index]

        data = Image.open(img_path)
        data = self.transforms[self.category](data)
        return data, label

    def __len__(self):
        return len(self.data_img_paths)



if __name__ == '__main__':
    train_set = DoCaSet('../cat_dog', 'train')
    print(len(train_set))
    val_set = DoCaSet('../cat_dog', 'val')
    print(len(val_set))
    test_set = DoCaSet('../cat_dog', 'test')
    print(len(test_set))

    train_data = test_set.__getitem__(random.randint(1,2000))
    print(train_data[1])
    print(train_data[0][0])
    plt.imshow(train_data[0].T)
    plt.show()
