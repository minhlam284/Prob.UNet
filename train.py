import torch
import os
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation

from dataset.mmis import get_mmis_dataset

def get_dataloader(args):
    train_dataset, val_dataset = get_mmis_dataset(args, mode="both")
    
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
    os.makedirs(args.log_dir, exist_ok=True)
    train_dataloader, val_dataloader = get_dataloader(args)
    
    # TODO resume checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}")
        for step, (mask, image) in enumerate(tqdm(train_dataloader)):

            net.forward(image, mask, training=True)
            elbo = net.elbo(image)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            
            if torch.isnan(loss) or check_nan_grad(net):
                print(f"Skipping step {step}")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.eval_frequency == 0:
            # TODO eval for val_dataloader
            model_path = f"{args.log_dir}/prob_unet_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, model_path)
            print(f"Model saved at {model_path}")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Training Probabilistic Unet")
    # parser.add_argument("--dataset", default=None, type=str, help="Dataset to use.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to the dataset pickle file")
    parser.add_argument("--image_size", default=128, type=int, help="Size of image to resize for training")
    parser.add_argument("--mask_type", choices=["ensemble", "random"], default="random", type=str, help="Use ensemble mask or random mask for each datapoint")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help ="Number of workers to read data")
    parser.add_argument("--epoch", default=100, type=int, help ="Number of epoch to train")
    parser.add_argument("--eval_frequency", default=10, type=int, help ="Frequency (in number of epochs) for evaluation")
    parser.add_argument("--log_dir", default="log", type=str, help="Log folder")
    args = parser.parse_args()
    
    train(args)