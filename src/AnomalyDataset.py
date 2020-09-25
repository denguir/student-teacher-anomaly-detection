import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class AnomalyDataset(Dataset):
    '''Anomaly detection dataset'''

    def __init__(self, csv_file, root_dir, transform=None):
        super(AnomalyDataset, self).__init__()
        self.frame_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
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

    brain_dataset = AnomalyDataset(csv_file='../data/brain/brain_tumor.csv',
                                   root_dir='../data/brain/img',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((256, 256)),
                                       transforms.RandomCrop((256, 256)),
                                       transforms.ToTensor()]))
    
    dataloader = DataLoader(brain_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for i, batch in enumerate(dataloader):
        print(i, batch['image'].size(), batch['label'].size())
        # display 3rd batch
        if i == 3:
            n = np.random.randint(0, len(batch['label']))

            image = torch.squeeze(batch['image'][n, :, :, :])
            label = batch['label'][n]

            plt.title(f"Sample #{n} - {'Tumor' if label else 'No Tumor'}")
            plt.imshow(image)
            plt.show()
            break