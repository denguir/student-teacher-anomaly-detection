import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from AnomalyNet import AnomalyNet
from FDFEAnomalyNet import FDFEAnomalyNet
from ExtendedAnomalyNet import ExtendedAnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader
from students_training import increment_mean_and_var

pH = 65
pW = 65
imH = 256
imW = 256
sL1, sL2, sL3 = 2, 2, 2
EPOCHS = 10
N_STUDENTS = 3
N_TEST = 5


def get_error_map(students_pred, teacher_pred):
    mu_students = torch.mean(students_pred, 1)
    err = torch.norm(mu_students - teacher_pred, dim=3)**2
    return err

def get_variance_map(students_pred):
    sse = torch.norm(students_pred, dim=4)**2
    msse = torch.mean(sse, 1)
    mu_students = torch.mean(students_pred, 1)
    var = msse - torch.norm(mu_students, dim=3)**2
    return var


if __name__ == '__main__':
    
    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Teacher network
    teacher_hat = AnomalyNet()
    teacher = FDFEAnomalyNet(base_net=teacher_hat, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    teacher.eval().to(device)

    # Load teacher model
    teacher.load_state_dict(torch.load('../model/teacher_net.pt'))

    # Students networks
    students_hat = [AnomalyNet() for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat]
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(N_STUDENTS):
        model_name = f'../model/student_net_{i}.pt'
        print(f'Loading model from {model_name}.')
        students[i].load_state_dict(torch.load(model_name))

    # Callibration on anomaly-free dataset
    callibration_dataset = AnomalyDataset(csv_file='../data/brain/brain_tumor.csv',
                                   root_dir='../data/brain/img',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((imH, imW)),
                                       transforms.ToTensor()]),
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
        mu_err, var_err, N_err = 0, 0, 0
        mu_var, var_var, N_var = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs) for student in students], dim=1)
            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
            mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)


    # Load testing data
    brain_dataset = AnomalyDataset(csv_file='../data/brain/brain_tumor.csv',
                                   root_dir='../data/brain/img',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize((imH, imW)),
                                       transforms.ToTensor()]),
                                    type='test')
    dataloader = DataLoader(brain_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_set = iter(dataloader)
    # Build anomaly map
    with torch.no_grad():
        for i in range(N_TEST):
            batch = next(test_set)
            inputs = batch['image'].to(device)
            label = batch['label'].cpu()
            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs) for student in students], dim=1)
            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            score_map = (s_err - mu_var) / torch.sqrt(var_err) + (var_err - mu_var) / torch.sqrt(var_var)
            
            img_in = torch.squeeze(inputs).permute(1, 2, 0).cpu()
            score_map = torch.squeeze(score_map).cpu()

            fig = plt.figure()
            axes = fig.add_subplot(2, 1, 1)
            axes.matshow(img_in, cmap='gray')
            axes.set_title('Original image')

            axes = fig.add_subplot(2, 1, 2)
            axes.matshow(score_map, cmap='viridis')
            axes.set_title('Anomaly map')

            plt.show()
