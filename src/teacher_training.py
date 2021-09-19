import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from einops import reduce, rearrange
from tqdm import tqdm
from AnomalyNet import AnomalyNet
from AnomalyResnet18 import AnomalyResnet18
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model


pH = 65
pW = 65
imH = 256
imW = 256
EPOCHS = 1_000
DATASET = sys.argv[1]


def distillation_loss(output, target):
    # dim: (batch, vector)
    err = torch.norm(output - target, dim=1)**2
    loss = torch.mean(err)
    return loss


def compactness_loss(output):
    # dim: (batch, vector)
    _, n = output.size()
    avg = torch.mean(output, axis=1)
    std = torch.std(output, axis=1)
    zt = output.T - avg
    zt /= std
    corr = torch.matmul(zt.T, zt) / (n - 1)
    loss = torch.sum(torch.triu(corr, diagonal=1)**2)
    return loss


if __name__ == '__main__':

    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Pretrained network for knowledge distillation
    resnet18 = AnomalyResnet18()
    resnet_model = f'../model/{DATASET}/resnet18.pt'
    load_model(resnet18, resnet_model)

    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18.eval().to(device)

    # Teacher network
    teacher = AnomalyNet.create((pH, pW))
    teacher.to(device)

    # Loading saved model
    model_name = f'../model/{DATASET}/teacher_net.pt'
    load_model(teacher, model_name)

    # Define optimizer
    optimizer = optim.Adam(teacher.parameters(), lr=2e-4, weight_decay=1e-5)

    # Load training data
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                    root_dir=f'../data/{DATASET}/img',
                                    transform=transforms.Compose([
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((imH, imW)),
                                        transforms.RandomCrop((pH, pW)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(180),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='train')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # training
    min_running_loss = np.inf
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, batch in tqdm(enumerate(dataloader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch['image'].to(device)
            with torch.no_grad():
                targets = rearrange(resnet18(inputs), 'b vec h w -> b (vec h w)') # h=w=1
                #targets = torch.squeeze(resnet18(inputs))
            outputs = teacher(inputs)
            loss = distillation_loss(outputs, targets) + compactness_loss(outputs)

            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print stats
        print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
            
        if running_loss < min_running_loss:
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Saving model to {model_name}.")
            torch.save(teacher.state_dict(), model_name)

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0

            


