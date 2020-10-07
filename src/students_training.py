import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from AnomalyNet import AnomalyNet
from FDFEAnomalyNet import FDFEAnomalyNet
from ExtendedAnomalyNet import ExtendedAnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader

pH = 65
pW = 65
imH = 256
imW = 256
sL1, sL2, sL3 = 2, 2, 2 # stride of max pool layers in AnomalyNet
EPOCHS = 70
N_STUDENTS = 3
DATASET = 'carpet'


def increment_mean_and_var(mu_N, var_N, N, batch):
    '''Increment value of mean and variance based on
       current mean, var and new batch
    '''
    B = batch.size()[0] # batch size
    mu_B = torch.mean(batch, 0) # mean over batch
    S_B = B * torch.var(batch, 0, unbiased=False) 
    S_N = N * var_N
    mu_NB = N/(N + B) * mu_N + B/(N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N+B)
    return mu_NB, var_NB, N + B


def student_loss(output, target):
    err = torch.norm(output - target, dim=3)**2
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
    teacher.load_state_dict(torch.load(f'../model/{DATASET}/teacher_net.pt'))

    # Students networks
    students_hat = [AnomalyNet() for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat]
    students = [student.to(device) for student in students]

    # Loading students models
    for i in range(N_STUDENTS):
        model_name = f'../model/{DATASET}/student_net_{i}.pt'
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
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                   root_dir=f'../data/{DATASET}/img',
                                   transform=transforms.Compose([
                                       #transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((imH, imW)),
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
        mu, var, N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = teacher(inputs)
            mu, var, N = increment_mean_and_var(mu, var, N, t_out)
        # print('Saving mean and variance of teacher net ...')
        # torch.save(mu, '../model/mu.pt')
        # torch.save(var, '../model/var.pt')

    # Training
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for j, student in enumerate(students[1:], start=1):
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
                    targets = (teacher(inputs) - mu) / torch.sqrt(var)
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

            


