from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from einops import rearrange, reduce
from tqdm import tqdm

from AnomalyDataset import load_teacher_train_set
from AnomalyNet import load_teacher_model
from AnomalyResnet18 import load_backbone_model
from config import Config


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="carpet",
        help="Dataset to train on (in data folder)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=65,
        choices=[17, 33, 65],
        help="Height and width of patch CNN",
    )
    parser.add_argument("--image_size", type=int, default=256)

    # trainer arguments
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument(
        "--gpus", type=int, default=(1 if torch.cuda.is_available() else 0)
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    return args


def distillation_loss(output, target):
    # dim: (batch, vector)
    err = torch.norm(output - target, dim=1) ** 2
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
    loss = torch.sum(torch.triu(corr, diagonal=1) ** 2)
    return loss


def train(args):
    # Choosing device
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f"Device used: {device}")

    # Pretrained network for knowledge distillation
    resnet18 = load_backbone_model(args.dataset)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18.eval().to(device)

    # Teacher network
    teacher = load_teacher_model(args.dataset, args.patch_size)
    teacher.to(device)

    # Define optimizer
    optimizer = optim.Adam(
        teacher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Load training data
    dataloader = load_teacher_train_set(
        args.dataset,
        args.patch_size,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )

    # training
    min_running_loss = np.inf
    model_path = str(
        Config.MODEL_PATH / args.dataset / f"teacher_{args.patch_size}_net.pt"
    )
    for epoch in range(args.max_epochs):
        running_loss = 0.0

        for i, batch in tqdm(enumerate(dataloader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch["image"].to(device)
            with torch.no_grad():
                targets = rearrange(
                    resnet18(inputs), "b vec h w -> b (vec h w)"
                )  # h=w=1
                # targets = torch.squeeze(resnet18(inputs))
            outputs = teacher(inputs)
            loss = distillation_loss(outputs, targets) + compactness_loss(outputs)

            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print stats
        print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")

        if running_loss < min_running_loss and epoch > 0:
            torch.save(teacher.state_dict(), model_path)
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Model saved to {model_path}.")

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
