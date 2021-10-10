import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from einops import reduce
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to train on (in data folder)")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to train")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65], help="Height and width of patch CNN")
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()
    return args


def student_loss(output, target):
    # dim: (batch, h, w, vector)
    err = reduce((output - target)**2, 'b h w vec -> b h w', 'sum')
    loss = torch.mean(err)
    return loss


def train(args):
    # Choosing device 
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f'Device used: {device}')
    
    # Teacher network
    teacher = AnomalyNet.create((args.patch_size, args.patch_size))
    teacher.eval().to(device)

    # Load teacher model
    load_model(teacher, f'../model/{args.dataset}/teacher_{args.patch_size}_net.pt')

    # Students networks
    students = [AnomalyNet.create((args.patch_size, args.patch_size)) for _ in range(args.n_students)]
    students = [student.to(device) for student in students]

    # Loading students models
    for i in range(args.n_students):
        model_name = f'../model/{args.dataset}/student_{args.patch_size}_net_{i}.pt'
        load_model(students[i], model_name)

    # Define optimizer
    optimizers = [optim.Adam(student.parameters(), 
                            lr=args.learning_rate, 
                            weight_decay=args.weight_decay) for student in students]

    # Load anomaly-free training data
    dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                             transform=transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                             type='train',
                             label=0)
    

    # Preprocessing
    # Apply teacher network on anomaly-free dataset
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)
    print(f'Preprocessing of training dataset {args.dataset}...')

    # Compute incremental mean and var over traininig set
    # because the whole training set takes too much memory space 
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = teacher.fdfe(inputs)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)

    # Training
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)

    for j, student in enumerate(students):
        min_running_loss = np.inf
        model_name = f'../model/{args.dataset}/student_{args.patch_size}_net_{j}.pt'
        print(f'Training Student {j} on anomaly-free dataset ...')

        for epoch in range(args.max_epochs):
            running_loss = 0.0

            for i, batch in tqdm(enumerate(dataloader)):
                # zero the parameters gradient
                optimizers[j].zero_grad()

                # forward pass
                inputs = batch['image'].to(device)
                with torch.no_grad():
                    targets = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
                outputs = student.fdfe(inputs)
                loss = student_loss(targets, outputs)

                # backward pass
                loss.backward()
                optimizers[j].step()
                running_loss += loss.item()

                # print stats
                if i % 10 == 9:
                    print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
                    
                    if running_loss < min_running_loss and epoch > 0:
                        torch.save(student.state_dict(), model_name)
                        print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                        print(f"Model saved to {model_name}.")

                    min_running_loss = min(min_running_loss, running_loss)
                    running_loss = 0.0


if __name__ == '__main__':
    args = parse_arguments()
    train(args)