from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from einops import reduce
from tqdm import tqdm

from AnomalyDataset import load_student_train_set
from AnomalyNet import load_students_model, load_teacher_model
from config import Config
from utils import increment_mean_and_var


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
        "--n_students", type=int, default=3, help="Number of students network to train"
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
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument(
        "--gpus", type=int, default=(1 if torch.cuda.is_available() else 0)
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    return args


def student_loss(output, target):
    # dim: (batch, h, w, vector)
    err = reduce((output - target) ** 2, "b h w vec -> b h w", "sum")
    loss = torch.mean(err)
    return loss


def train(args):
    # Choosing device
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f"Device used: {device}")

    # Teacher network
    teacher = load_teacher_model(args.dataset, args.patch_size)
    teacher.eval().to(device)

    # Students networks
    students = load_students_model(args.dataset, args.patch_size, args.n_students)
    for student in students:
        student.to(device)

    # Define optimizer
    optimizers = [
        optim.Adam(
            student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for student in students
    ]

    # Load anomaly-free training data
    dataloader = load_student_train_set(
        args.dataset, args.image_size, args.batch_size, args.num_workers
    )

    # Preprocessing
    # Apply teacher network on anomaly-free dataset
    print(f"Preprocessing of training dataset {args.dataset}...")

    # Compute incremental mean and var over traininig set
    # because the whole training set takes too much memory space
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch["image"].to(device)
            t_out = teacher.fdfe(inputs)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)

    # Training
    for j, student in enumerate(students):
        min_running_loss = np.inf
        model_path = str(
            Config.MODEL_PATH / args.dataset / f"student_{args.patch_size}_net_{j}.pt"
        )
        print(f"Training Student {j} on anomaly-free dataset ...")

        for epoch in range(args.max_epochs):
            running_loss = 0.0

            for i, batch in tqdm(enumerate(dataloader)):
                # zero the parameters gradient
                optimizers[j].zero_grad()

                # forward pass
                inputs = batch["image"].to(device)
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
                        torch.save(student.state_dict(), model_path)
                        print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                        print(f"Model saved to {model_path}.")

                    min_running_loss = min(min_running_loss, running_loss)
                    running_loss = 0.0


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
