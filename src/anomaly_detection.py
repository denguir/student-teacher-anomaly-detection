import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange, reduce
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to infer on (in data folder)")
    parser.add_argument('--test_size', type=int, default=30, help="Number of images to visualize")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to use")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65])
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    return args


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


def detect_anomaly(args):
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
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(args.n_students):
        model_name = f'../model/{args.dataset}/student_{args.patch_size}_net_{i}.pt'
        load_model(students[i], model_name)

    # Callibration on anomaly-free dataset
    callibration_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                          transform=transforms.Compose([
                                              transforms.Resize((args.image_size, args.image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                          type='train',
                                          label=0)

    dataloader = DataLoader(callibration_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)

    with torch.no_grad():
        print('Callibrating teacher on Student dataset.')
        t_mu, t_var, t_N = 0, 0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)
            t_out = teacher.fdfe(inputs)
            t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)
        
        print('Callibrating scoring parameters on Student dataset.')
        max_err, max_var = 0, 0
        mu_err, var_err, N_err = 0, 0, 0
        mu_var, var_var, N_var = 0, 0, 0

        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch['image'].to(device)

            t_out = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
            mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

            max_err = max(max_err, torch.max(s_err))
            max_var = max(max_var, torch.max(s_var))


    # Load testing data
    dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                             transform=transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                              type='test')
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)


    # Build anomaly map
    test_set = iter(dataloader)
    unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # get back to original image
    with torch.no_grad():
        for i in range(args.test_size):
            batch = next(test_set)
            inputs = batch['image'].to(device)
            label = batch['label'].cpu()
            anomaly = 'with' if label.item() == 1 else 'without'

            t_out = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

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


if __name__ == '__main__':
    args = parse_arguments()
    detect_anomaly(args)