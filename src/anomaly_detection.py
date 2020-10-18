import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange, reduce
from torchsummary import summary
from AnomalyNet import AnomalyNet
from FDFEAnomalyNet import FDFEAnomalyNet
from ExtendedAnomalyNet import ExtendedAnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model, mc_dropout


pH = 65
pW = 65
imH = 256
imW = 256
sL1, sL2, sL3 = 2, 2, 2
EPOCHS = 10
N_STUDENTS = 3
N_TEST = 30
DATASET = sys.argv[1]


def get_error_map(students_pred, teacher_pred):
    # student: (batch, student_id, h, w, vector)
    # teacher: (batch, h, w, vector)
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    err = reduce((mu_students - teacher_pred)**2, 'b h w vec -> b h w', 'sum')
    return err


def get_variance_map(students_pred):
    # student: (batch, student_id, h, w, vector)
    sse = reduce(students_pred**2, 'b id h w vec -> b id h w', 'sum')
    msse = reduce(sse, 'b id h w -> b h w', 'mean')
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    var = msse - reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
    return var


def predict_student(students, inputs, n=2):
    s_out = torch.stack([student(inputs) for i in range(n) for student in students], dim=1)
    return s_out


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
    # students_hat = [AnomalyNet().apply(mc_dropout) for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat]
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(N_STUDENTS):
        model_name = f'../model/{DATASET}/student_net_{i}.pt'
        load_model(students[i], model_name)

    # Callibration on anomaly-free dataset
    callibration_dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                    root_dir=f'../data/{DATASET}/img',
                                    transform=transforms.Compose([
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((imH, imW)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='train',
                                    label=0)

    dataloader = DataLoader(callibration_dataset, batch_size=1, shuffle=False, num_workers=4)
    with torch.no_grad():
        print('Callibrating teacher on Student dataset.')
        t_mu, t_var, t_N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = teacher(inputs)
            t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)
        
        print('Callibrating scoring parameters on Student dataset.')
        max_err, max_var = 0, 0
        mu_err, var_err, N_err = 0, 0, 0
        mu_var, var_var, N_var = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs) for student in students], dim=1)
            # s_out = predict_student(students, inputs) # MC dropout
            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
            mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

            max_err = max(max_err, torch.max(s_err))
            max_var = max(max_var, torch.max(s_var))


    # Load testing data
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                    root_dir=f'../data/{DATASET}/img',
                                    transform=transforms.Compose([
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((imH, imW)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    test_set = iter(dataloader)

    unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # get back to original image
    # Build anomaly map
    with torch.no_grad():
        for i in range(N_TEST):
            batch = next(test_set)
            inputs = batch['image'].to(device)
            label = batch['label'].cpu()
            anomaly = 'with' if label.item() == 1 else 'without'

            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs) for student in students], dim=1)
            # s_out = predict_student(students, inputs) # MC dropout

            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            score_map = (s_err - mu_err) / torch.sqrt(var_err) + (s_var - mu_var) / torch.sqrt(var_var)

            img_in = unorm(rearrange(inputs, 'b c h w -> c h (b w)').cpu())
            img_in = rearrange(img_in, 'c h w -> h w c')

            score_map = rearrange(score_map, 'b h w -> h (b w)').cpu()

            # display results
            plt.figure(figsize=(13, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(img_in)
            plt.title(f'Original image - {anomaly} anomaly')

            plt.subplot(1, 2, 2)
            plt.imshow(score_map, cmap='jet')
            plt.imshow(img_in, cmap='gray', interpolation='none')
            plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
            plt.colorbar(extend='both')
            plt.title(f'Anomaly map')

            max_score = (max_err - mu_err) / torch.sqrt(var_err) + (max_var - mu_var) / torch.sqrt(var_var)
            plt.clim(0, max_score.item())

            plt.show(block=True)