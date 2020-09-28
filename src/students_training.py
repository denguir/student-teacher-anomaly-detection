import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from AnomalyNet import AnomalyNet
from ExtendedAnomalyNet import ExtendedAnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader

pH = 65
pW = 65
imH = 256
imW = 256
sL1, sL2, sL3 = 2, 2, 2
PATCH_SIZE = 65
IMG_SIZE = 256
EPOCHS = 1000
N_STUDENTS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device used: {device}')


if __name__ == '__main__':
    
    # Teacher network
    teacher_hat = AnomalyNet()
    teacher = ExtendedAnomalyNet(base_net=teacher_hat, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)

    # try load network in extended anomaly net just to see if it works
    
    # Students networks
    students = [AnomalyNet() for i in range(N_STUDENTS)]

    # Loading saved models
    for i in range(N_STUDENTS):
        model_name = f'../model/student_net_{i}.pt'
        try:
            print(f'Loading model from {model_name}.')
            students[i].load_state_dict(torch.load(model_name))
        except FileNotFoundError as e:
            print(e)
            print('No model available.')
            print(f'Initilialisation of a new model with random weights for student {i}.')

    # Define optimizer
    optimizers = [optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5) for student in students]

    # Load anomaly-free training data
    brain_dataset = AnomalyDataset(csv_file='../data/brain/brain_tumor.csv',
                                   root_dir='../data/brain/img',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                       transforms.ToTensor()]),
                                    type='train',
                                    label=0)
    dataloader = DataLoader(brain_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Preprocessing
    # Apply extended teacher network on training data
    
    # training
    min_running_loss = np.inf
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, batch in tqdm(enumerate(dataloader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch['image'].to(device)
            targets = torch.squeeze(resnet18(inputs))
            outputs = torch.squeeze(teacher(inputs))
            loss = distillation_loss(outputs, targets) + compactness_loss(outputs)

            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print stats
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
                
                if running_loss < min_running_loss:
                    print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                    print(f"Saving model to {model_name}.")
                    torch.save(teacher.state_dict(), model_name)

                min_running_loss = min(min_running_loss, running_loss)
                running_loss = 0.0

            


