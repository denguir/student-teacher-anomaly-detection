import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class AnomalyDataset(Dataset):
    '''Anomaly detection dataset.
    - root_dir: path to the dataset to train the model on, eg: <path>/data/carpet
    - transform: list of transformation to apply on input image, eg: Resize, Normalize, etc
    - gt_transform: list of transformation to apply on gt image, eg: Resize.
    - constraint: filter to apply on the reading of the CSV file, a filter is a kwarg.
                  eg: type='train' to filter train data
                      label=0 to filter on anomaly-free data
    '''

    def __init__(self, root_dir, transform=transforms.ToTensor(), gt_transform=transforms.ToTensor(), **constraint):
        super(AnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.gt_dir = os.path.join(self.root_dir, 'ground_truth')
        self.dataset = self.root_dir.split('/')[-1]
        self.csv_file =  os.path.join(self.root_dir, self.dataset + '.csv')
        self.frame_list = self._get_dataset(self.csv_file, constraint)
    
    def _get_dataset(self, csv_file, constraint):
        '''Apply filter based on the contraint dict on the dataset'''
        df = pd.read_csv(csv_file, keep_default_na=False)
        df = df.loc[(df[list(constraint)] == pd.Series(constraint)).all(axis=1)]
        return df
    
    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.frame_list.iloc[idx]
        img_path = os.path.join(self.img_dir, item['image_name'])
        label = self.frame_list.iloc[idx]['label']
        image = Image.open(img_path)
 
        if item['gt_name']:
            gt_path = os.path.join(self.gt_dir, item['gt_name'])
            gt = Image.open(gt_path)
        else:
            gt = Image.new('L', image.size, color=0)

        sample = {'label': label}

        if self.transform:
            sample['image'] = self.transform(image)

        if self.gt_transform:
            sample['gt'] = self.gt_transform(gt)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys 
    
    DATASET = sys.argv[1]
    dataset = AnomalyDataset(root_dir=f'../data/{DATASET}',
                             transform=transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.RandomCrop((256, 256)),
                                transforms.ToTensor()]),
                             type='train',
                             label=0)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    for i, batch in enumerate(dataloader):
        print(i, batch['image'].size(), batch['label'].size())
        # display 3rd batch
        if i == 3:
            n = np.random.randint(0, len(batch['label']))

            image = rearrange(batch['image'][n, :, :, :], 'c h w -> h w c')
            label = batch['label'][n]

            plt.title(f"Sample #{n} - {'Anomalous' if label else 'Normal'}")
            plt.imshow(image)
            plt.show()
            break