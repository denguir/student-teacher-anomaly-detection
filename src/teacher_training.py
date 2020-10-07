import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader

pH = 65
pW = 65
imH = 256
imW = 256
EPOCHS = 1_000
DATASET = 'carpet'


def distillation_loss(output, target):
    err = torch.norm(output - target, dim=1)**2
    loss = torch.mean(err)
    return loss


def compactness_loss(output):
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
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
    resnet18.eval().to(device)

    # Teacher network
    teacher = AnomalyNet()
    teacher.to(device)

    # Loading saved model
    model_name = f'../model/{DATASET}/teacher_net.pt'
    try:
        print(f'Loading model from {model_name}.')
        teacher.load_state_dict(torch.load(model_name))
    except FileNotFoundError as e:
        print(e)
        print('No model available.')
        print('Initilialisation of a new model with random weights for teacher.')

    # Define optimizer
    optimizer = optim.Adam(teacher.parameters(), lr=2e-4, weight_decay=1e-5)

    # Load training data
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                   root_dir=f'../data/{DATASET}/img',
                                   transform=transforms.Compose([
                                       #transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((imH, imW)),
                                       transforms.RandomCrop((pH, pW)),
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
                targets = torch.squeeze(resnet18(inputs))
            outputs = torch.squeeze(teacher(inputs))
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

            


