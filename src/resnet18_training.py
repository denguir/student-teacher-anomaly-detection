import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from AnomalyResnet18 import AnomalyResnet18
from AnomalyDataset import AnomalyDataset
from torchvision import transforms, utils
from torch.utils.data.dataloader import DataLoader
from utils import load_model

imH = 256
imW = 256
EPOCHS = 100
DATASET = 'brain'


if __name__ == '__main__':

    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Resnet pretrained network for knowledge distillation
    resnet18 = AnomalyResnet18()
    resnet18.to(device)

    # Loading saved model
    model_name = f'../model/{DATASET}/resnet18.pt'
    load_model(resnet18, model_name)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    # Load training data
    dataset = AnomalyDataset(csv_file=f'../data/{DATASET}/{DATASET}.csv',
                                    root_dir=f'../data/{DATASET}/img',
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize((imH, imW)),
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
        if running_loss < min_running_loss:
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Accuracy: {accuracy}")
            print(f"Saving model to {model_name}.")
            torch.save(resnet18.state_dict(), model_name)

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0

            


