import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from tqdm import tqdm
from einops import reduce
from torchsummary import summary
from AnomalyNet import AnomalyNet
from FDFEAnomalyNet import FDFEAnomalyNet
from ExtendedAnomalyNet import ExtendedAnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model

pH = 65
pW = 65
imH = 256
imW = 256
sL1, sL2, sL3 = 2, 2, 2 # stride of max pool layers in AnomalyNet
EPOCHS = 15
N_STUDENTS = 3
DATASET = sys.argv[1]


def student_loss(output, target):
    # dim: (batch, h, w, vector)
    err = reduce((output - target)**2, 'b h w vec -> b h w', 'sum')
    loss = torch.mean(err)
    return loss


if __name__ == '__main__':

    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')
    
    # Teacher network
    teacher_hat = AnomalyNet()
    teacher = FDFEAnomalyNet(base_net=teacher_hat, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    teacher.eval().to(device)

    # Load teacher model
    load_model(teacher, f'../model/{DATASET}/teacher_net.pt')

    # Students networks
    students_hat = [AnomalyNet() for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat]
    students = [student.to(device) for student in students]

    # Loading students models
    for i in range(N_STUDENTS):
        model_name = f'../model/{DATASET}/student_net_{i}.pt'
        load_model(students[i], model_name)

    # Define optimizer
    optimizers = [optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5) for student in students]

    # Load anomaly-free training data
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                    root_dir=f'../data/{DATASET}/img',
                                    transform=transforms.Compose([
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((imH, imW)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='train',
                                    label=0)
    

    # Preprocessing
    # Apply teacher network on anomaly-free dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Preprocessing of training dataset ...')

    # Compute incremental mean and var over traininig set
    # because the whole training set takes too much memory space 
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = teacher(inputs)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)
        # print('Saving mean and variance of teacher net ...')
        # torch.save(t_mu, '../model/t_mu.pt')
        # torch.save(t_var, '../model/t_var.pt')

    
    # Training
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for j, student in enumerate(students):
        print(f'Training Student {j} on anomaly-free dataset ...')
        min_running_loss = np.inf
        model_name = f'../model/{DATASET}/student_net_{j}.pt'
        for epoch in range(EPOCHS):
            running_loss = 0.0

            for i, batch in tqdm(enumerate(dataloader)):
                # zero the parameters gradient
                optimizers[j].zero_grad()

                # forward pass
                inputs = batch['image'].to(device)
                with torch.no_grad():
                    targets = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
                outputs = student(inputs)
                loss = student_loss(targets, outputs)

                # backward pass
                loss.backward()
                optimizers[j].step()
                running_loss += loss.item()

                # print stats
                if i % 10 == 9:
                    print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
                    
                    if running_loss < min_running_loss:
                        print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                        print(f"Saving model to {model_name}.")
                        torch.save(student.state_dict(), model_name)

                    min_running_loss = min(min_running_loss, running_loss)
                    running_loss = 0.0

            


