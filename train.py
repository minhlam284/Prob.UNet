import torch
import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from model.probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation

from dataset import get_lidc_dataset, get_mmis_dataset, get_qubiq_pan_dataset, get_qubiq_pan_les_dataset

def get_dataloader(args):
    if args.dataset == "lidc":
        train_dataset, val_dataset = get_lidc_dataset(args)
    elif args.dataset == "mmis":
        train_dataset, val_dataset = get_mmis_dataset(args)
    elif args.dataset == "qubiq_pan":
        train_dataset, val_dataset = get_qubiq_pan_dataset(args)
    elif args.dataset == "qubiq_pan_les":
        train_dataset, val_dataset = get_qubiq_pan_les_dataset(args)
    
    print("Number of training/val:", (len(train_dataset), len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    return train_dataloader, val_dataloader

def check_nan_grad(net):
    for name, param in net.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

def train(args):
    args.log_dir = args.log_dir + f"/{args.dataset}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_filepath = f"{args.log_dir}/log.txt"
    args_filepath = f"{args.log_dir}/args.json"

    with open(args_filepath, 'w') as f:
        json.dump(vars(args), f)
    
    train_dataloader, val_dataloader = get_dataloader(args)
    
    # TODO resume checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ProbabilisticUnet(input_channels=args.input_channel, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    
    for epoch in range(args.epoch):
        train_loss = 0
        for step, (mask, image) in enumerate(tqdm(train_dataloader)):
            image = image.to(device)
            mask = mask.to(device)
            net.forward(image, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            train_loss += loss.item()
            
            if torch.isnan(loss) or check_nan_grad(net):
                print(f"Skipping step {step}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_dataloader)
        with open(log_filepath, "w", encoding="utf-8") as file:
            file.write(f"epoch {epoch+1}/{args.epoch} --- train_loss {train_loss}\n")

        if (epoch + 1) % args.eval_frequency == 0:
            # TODO eval for val_dataloader
            model_path = f"{args.log_dir}/prob_unet_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print(f"Model saved at {model_path}")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Training Probabilistic Unet")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset to use.")
    parser.add_argument("--data_dir", default="data", type=str, help="Data folder.")
    parser.add_argument("--input_channel", default=1, type=int, help="Number of input channels.")
    parser.add_argument("--image_size", default=256, type=int, help="Size of image to resize for training")
    parser.add_argument("--mask_type", choices=["ensemble", "random"], default="random", type=str, help="Use ensemble mask or random mask for each datapoint")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help ="Number of workers to read data")
    parser.add_argument("--epoch", default=100, type=int, help ="Number of epoch to train")
    parser.add_argument("--eval_frequency", default=10, type=int, help ="Frequency (in number of epochs) for evaluation")
    parser.add_argument("--log_dir", default="logs", type=str, help="Log folder")
    args = parser.parse_args()
    
    train(args)