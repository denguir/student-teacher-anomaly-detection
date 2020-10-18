import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from einops import rearrange
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class AnomalyDataset(Dataset):
    '''Anomaly detection dataset'''

    def __init__(self, csv_file, root_dir, transform=None, **constraint):
        super(AnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.frame_list = self._get_dataset(csv_file, constraint)
    
    def _get_dataset(self, csv_file, constraint):
        '''Apply filter based on the contraint dict on the dataset'''
        df = pd.read_csv(csv_file)
        df = df.loc[(df[list(constraint)] == pd.Series(constraint)).all(axis=1)]
        return df
    
    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frame_list.iloc[idx]['image_name'])
        label = self.frame_list.iloc[idx]['label']
        image = Image.open(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(image)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys 
    
    DATASET = sys.argv[1]
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                   root_dir=f'../data/{DATASET}/img',
                                   transform=transforms.Compose([
                                       #transforms.Grayscale(num_output_channels=3),
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