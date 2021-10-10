import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from tqdm import tqdm
from argparse import ArgumentParser
from AnomalyResnet18 import AnomalyResnet18
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to train on (in data folder)")
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    return args


def train(args):
    # Choosing device 
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f'Device used: {device}')

    # Resnet pretrained network for knowledge distillation
    resnet18 = AnomalyResnet18()
    resnet18.to(device)

    # Loading saved model
    model_name = f'../model/{args.dataset}/resnet18.pt'
    load_model(resnet18, model_name)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), 
                          lr=args.learning_rate, 
                          momentum=args.momentum)

    # Load training data
    dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}/img',
                             transform=transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
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
        running_corrects = 0
        max_running_corrects = 0

        for i, batch in tqdm(enumerate(dataloader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            # backward pass
            loss.backward()
            optimizer.step()

            # loss and accuracy
            running_loss += loss.item()
            max_running_corrects += len(targets)
            running_corrects += torch.sum(preds == targets.data)
            

        # print stats
        print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
        accuracy = running_corrects.double() / max_running_corrects
        if running_loss < min_running_loss and epoch > 0:
            torch.save(resnet18.state_dict(), model_name)
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Accuracy: {accuracy}")
            print(f"Model saved to {model_name}.")

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0

            
if __name__ == '__main__':
    args = parse_arguments()
    train(args)