import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from einops import reduce, rearrange
from tqdm import tqdm
from argparse import ArgumentParser
from AnomalyNet import AnomalyNet
from AnomalyResnet18 import AnomalyResnet18
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to train on (in data folder)")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65], help="Height and width of patch CNN")
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()
    return args


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


def train(args):
    # Choosing device 
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f'Device used: {device}')

    # Pretrained network for knowledge distillation
    resnet18 = AnomalyResnet18()
    resnet_model = f'../model/{args.dataset}/resnet18.pt'
    load_model(resnet18, resnet_model)

    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18.eval().to(device)

    # Teacher network
    teacher = AnomalyNet.create((args.patch_size, args.patch_size))
    teacher.to(device)

    # Loading saved model
    model_name = f'../model/{args.dataset}/teacher_{args.patch_size}_net.pt'
    load_model(teacher, model_name)

    # Define optimizer
    optimizer = optim.Adam(teacher.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    # Load training data
    dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                             transform=transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomCrop((args.patch_size, args.patch_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                             type='train')
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=args.num_workers)

    # training
    min_running_loss = np.inf
    for epoch in range(args.max_epochs):
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
            
        if running_loss < min_running_loss and epoch > 0:
            torch.save(teacher.state_dict(), model_name)
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Model saved to {model_name}.")

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0


if __name__ == '__main__':
    args = parse_arguments()
    train(args)


