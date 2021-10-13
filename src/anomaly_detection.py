import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange, reduce
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model
from sklearn.metrics import roc_curve, auc


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to infer on (in data folder)")
    parser.add_argument('--test_size', type=int, default=20, help="Number of batch for the test set")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to use")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--visualize', type=bool, default=True, help="Display anomaly map batch per batch")

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


@torch.no_grad()
def calibrate(teacher, students, dataloader, device):
    print('calibrating teacher on Student dataset.')
    t_mu, t_var, t_N = 0, 0, 0
    for _, batch in tqdm(enumerate(dataloader)):
        inputs = batch['image'].to(device)
        t_out = teacher.fdfe(inputs)
        t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)
    
    print('calibrating scoring parameters on Student dataset.')
    max_err, max_var = 0, 0
    mu_err, var_err, N_err = 0, 0, 0
    mu_var, var_var, N_var = 0, 0, 0

    for _, batch in tqdm(enumerate(dataloader)):
        inputs = batch['image'].to(device)

        t_out = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
        s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

        s_err = get_error_map(s_out, t_out)
        s_var = get_variance_map(s_out)
        mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
        mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

        max_err = max(max_err, torch.max(s_err))
        max_var = max(max_var, torch.max(s_var))
    
    return {"teacher": {"mu": t_mu, "var": t_var},
            "students": {"err":
                            {"mu": mu_err, "var": var_err, "max": max_err},
                         "var":
                            {"mu": mu_var, "var": var_var, "max": max_var}
                        }
            }


@torch.no_grad()
def get_score_map(inputs, teacher, students, params):
    t_out = (teacher.fdfe(inputs) - params['teacher']['mu']) / torch.sqrt(params['teacher']['var'])
    s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

    s_err = get_error_map(s_out, t_out)
    s_var = get_variance_map(s_out)
    score_map = (s_err - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                    + (s_var - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'])
    
    return score_map


def visualize(img, gt, score_map, max_score):
    plt.figure(figsize=(13, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title(f'Ground thuth anomaly')

    plt.subplot(1, 3, 3)
    plt.imshow(score_map, cmap='jet')
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
    plt.colorbar(extend='both')
    plt.title('Anomaly map')

    plt.clim(0, max_score)
    plt.show(block=True)


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

    # calibration on anomaly-free dataset
    calib_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                    transform=transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='train',
                                    label=0)

    calib_dataloader = DataLoader(calib_dataset, 
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   num_workers=args.num_workers)
    
    params = calibrate(teacher, students, calib_dataloader, device)


    # Load testing data
    test_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                  transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                  gt_transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor()]),
                                  type='test')

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=True, 
                                 num_workers=args.num_workers)

    
    # Build anomaly map
    y_score = np.array([])
    y_true = np.array([])
    test_iter = iter(test_dataloader)

    for i in range(args.test_size):
        batch = next(test_iter)
        inputs = batch['image'].to(device)
        gt = batch['gt'].cpu()

        score_map = get_score_map(inputs, teacher, students, params).cpu()
        y_score = np.concatenate((y_score, rearrange(score_map, 'b h w -> (b h w)').numpy()))
        y_true = np.concatenate((y_true, rearrange(gt, 'b c h w -> (b c h w)').numpy()))

        if args.visualize:
            unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # get back to original image
            max_score = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                + (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var']).item()
            img_in = rearrange(unorm(inputs).cpu(), 'b c h w -> b h w c')
            gt_in = rearrange(gt, 'b c h w -> b h w c')

            for b in range(args.batch_size):
                visualize(img_in[b, :, :, :].squeeze(), 
                          gt_in[b, :, :, :].squeeze(), 
                          score_map[b, :, :].squeeze(), 
                          max_score)
    
    # AUC ROC
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score)
    plt.figure(figsize=(13, 3))
    plt.plot(fpr, tpr, 'r', label="ROC")
    plt.plot(fpr, fpr, 'b', label="random")
    plt.title(f'ROC AUC: {auc(fpr, tpr)}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    detect_anomaly(args)