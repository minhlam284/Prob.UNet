from typing import List
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
from dataset import (get_isic2016_dataset, get_isic2018_dataset,
                    get_lidc_dataset, get_msmri_dataset)

def get_dataloader(args):
    if args.dataset == "lidc":
        train_dataset, val_dataset = get_lidc_dataset(args, mode="val")
    elif args.dataset == "msmri":
        train_dataset, val_dataset = get_msmri_dataset(args, mode="val")
    elif "isic" in args.dataset:
        if args.dataset == "isic2016":
            train_dataset, val_dataset = get_isic2016_dataset(args, mode="val")
        elif args.dataset == "isic2018":
            train_dataset, val_dataset = get_isic2018_dataset(args, mode="val")
    
    print("Number of training/val:", (len(train_dataset), len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    return train_dataloader, val_dataloader

def post_process(logits):
    preds = (torch.sigmoid(logits) > 0.5).to(torch.float32)
    return preds

def save_image(labels, gts, preds, image_folder, batch):
    # labels: Tensor(b, c, w, h)
    # gts: List[Tensor(b, w, h) x n]
    # preds: List[Tensor(b, w, h) x n]
    os.makedirs(image_folder, exist_ok=True)

    for i in range(labels.shape[0]):
        image = labels[i][0].numpy()
        plt.imsave(f"{image_folder}/image_{batch}_{i}.png", image)
        for id in range(len(gts)):
            gt_image = gts[id][i].cpu().numpy()
            plt.imsave(f"{image_folder}/gt_{batch}_{i}_{id}.png", gt_image, cmap="gray")
        for id in range(len(preds)):
            pred_image = preds[id][i].cpu().numpy()
            plt.imsave(f"{image_folder}/pred_{batch}_{i}_{id}.png", pred_image, cmap="gray")

def compute_metric(preds: List[torch.Tensor], gts: List[torch.Tensor], batch: int):
    """_summary_
    Args:
        preds (_type_): List[Tensor(b, w, h) x n]
        gts (_type_): List[Tensor(b, w, h) x m]
    """
    def compute_iou(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        
        Args:
            output: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
            target: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ & target_).sum(axis=(-2, -1))
        union = (output_ | target_).sum(axis=(-2, -1))
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.mean()

    def compute_dice(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        Args:
            output: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
            target: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ * target_).sum(axis=(-2, -1))
        total = output_.sum(axis=(-2, -1)) + target_.sum(axis=(-2, -1))
        dice = (2. * intersection + smooth) / (total + smooth)
        
        return dice.mean()

    def compute_ged(preds: torch.Tensor, gts: torch.Tensor):
        """_summary_
        Args:
            preds (_type_): Tensor(n, w, h)
            gts (_type_): Tensor(m, w, h)
        """
        n, m = preds.shape[0], gts.shape[0]
        d1, d2, d3 = 0, 0, 0
        
        for i in range(n):
            for j in range(m):
                d1 = d1 + (1 - compute_iou(preds[i], gts[j]))
        
        for i in range(n):
            for j in range(n):
                d2 = d2 + (1 - compute_iou(preds[i], preds[j]))
        
        for i in range(m):
            for j in range(m):
                d3 = d3 + (1 - compute_iou(gts[i], gts[j]))
        
        d1, d2, d3 = (2*d1)/(n*m), d2/(n*n), d3/(m*m)
        ged = d1 - d2 - d3
        return ged

    def compute_max_dice(preds: torch.Tensor, gts: torch.Tensor):
        """_summary_
        Args:
            preds (_type_): Tensor(n, w, h)
            gts (_type_): Tensor(m, w, h)
        """
        max_dice = 0
        for gt in gts:
            dices = [compute_dice(pred, gt) for pred in preds]
            max_dice += max(dices)
        return max_dice / len(gts)

    def compute_ncc(sample_arr: torch.Tensor, gt_arr: torch.Tensor):
        """_summary_
        Args:
            preds (_type_): Tensor(n, w, h)
            gts (_type_): Tensor(m, w, h)
        """
        """
        :param sample_arr: expected shape N x X x Y 
        :param gt_arr: M x X x Y
        :return: 
        """

        def ncc_pair(a, v, zero_norm=True, eps=1e-8):
            a = a.flatten()
            v = v.flatten()
            if zero_norm:
                a = (a - np.mean(a) + eps) / (np.std(a) * len(a) + eps)
                v = (v - np.mean(v) + eps) / (np.std(v) + eps)
            else:
                a = (a + eps) / (np.std(a) * len(a) + eps)
                v = (v + eps) / (np.std(v) + eps)
            return np.correlate(a, v).item()

        def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
            log_samples = np.log(m_samp + eps)
            return -1.0*np.sum(m_gt*log_samples, axis=0)

        sample_arr = sample_arr.unsqueeze(1).detach().cpu().numpy() # (n, 1, w, h)
        gt_arr = gt_arr.unsqueeze(1).detach().cpu().numpy() # (m, 1, w, h)
        mean_seg = np.mean(sample_arr, axis=0)
        N = sample_arr.shape[0]
        M = gt_arr.shape[0]
        sX = sample_arr.shape[2]
        sY = sample_arr.shape[3]
        E_ss_arr = np.zeros((N,sX,sY))
        for i in range(N):
            E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
            E_ss = np.mean(E_ss_arr, axis=0)
            E_sy_arr = np.zeros((M,N, sX, sY))
            for j in range(M):
                for i in range(N):
                    E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])
            E_sy = np.mean(E_sy_arr, axis=1)
            ncc_list = []
            for j in range(M):
                ncc_list.append(ncc_pair(E_ss, E_sy[j,...]))
        return (1/M)*sum(ncc_list)

    ged, ncc, max_dice, dice, iou = 0, 0, 0, 0, 0

    preds = torch.stack(preds, dim=1) # b, n, w, h
    gts = torch.stack(gts, dim=1) # b, m, w, h
    # for batch
    for _preds, _gts in zip(preds, gts):
        # _preds: n, w, h
        # _gts: m, w, h
        ged += compute_ged(_preds, _gts)
        max_dice += compute_max_dice(_preds, _gts)
        ncc += compute_ncc(_preds, _gts)

        pred = _preds.mean(dim=0) # w, h
        gt = _gts.mean(dim=0) # w, h
        dice += compute_dice(pred, gt)
        iou += compute_iou(pred, gt)
    
    batch = preds.shape[0]
    return ged/batch, ncc / batch, max_dice/batch, dice/batch, iou/batch

def eval(args):
    checkpoint_path = Path(args.checkpoint)
    with open(checkpoint_path.parent / f"{args.filename}.txt", "w", encoding="utf-8") as file:
        file.write(f"checkpoint: {args.checkpoint}\n")
        file.write(f"n_ensemble: {args.n_ensemble}\n")
        file.write(f"batch_size: {args.batch_size}\n")

    val_dataloader = get_dataloader(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    model.to(device)

    dice, ncc, max_dice, iou, ged = {}, {}, {}, {}, {}
    for i in range(args.n_ensemble):
        dice[i], ncc[i], max_dice[i], iou[i], ged[i] = 0, 0, 0, 0, 0

    for batch_id, (masks, patches) in enumerate(tqdm(val_dataloader)):
        images = patches.to(device)
        if not isinstance(masks, list):
            masks = [masks]
        
        preds = []
        model(images, None, training=False)
        for i in tqdm(range(args.n_ensemble)):
            sample_logits = model.sample()
            preds.append(post_process(sample_logits).cpu().squeeze(dim=1))

            metrics = compute_metric(preds, masks, batch=images.shape[0])
            
            ged_iter, ncc_iter, max_dice_iter, dice_iter, iou_iter = metrics
            ged[i] += ged_iter
            ncc[i] += ncc_iter
            max_dice[i] += max_dice_iter
            dice[i] += dice_iter
            iou[i] += iou_iter

            with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
                file.write(f"n_ensemble: {i} --- " 
                            f"ged_iter: {ged_iter} --- "
                            f"ncc_iter: {ncc_iter} --- "
                            f"max_dice_iter: {max_dice_iter} --- "
                            f"dice_iter: {dice_iter} --- "
                            f"iou_iter: {iou_iter}\n")

        save_image(patches, masks, preds, checkpoint_path.parent / args.filename, batch_id)

    for i in range(args.n_ensemble):
        dice[i] /= len(val_dataloader)
        ncc[i] /=  len(val_dataloader)
        max_dice[i] /= len(val_dataloader)
        iou[i] /= len(val_dataloader)
        ged[i] /= len(val_dataloader)
        with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
            file.write(f"n_ensemble: {i} --- "
                        f"GED: {ged[i]} --- "
                        f"NCC: {ncc[i]} --- "
                        f"Max_Dice: {max_dice[i]} --- "
                        f"Dice: {dice[i]} --- "
                        f"IoU: {iou[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Probabilistic Unet")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset to use.")
    parser.add_argument("--image_size", default=256, type=int, help="Size of image to resize for training")
    parser.add_argument("--mask_type", choices=["ensemble", "random"], default="ensemble", type=str, help="Use ensemble mask or random mask for each datapoint")
    parser.add_argument("--n_ensemble", default=1, type=int, help="Number of samples to ensemble")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help ="Number of workers to read data")
    parser.add_argument("--checkpoint", default=None, type=str, help="Specify the checkpoint")
    parser.add_argument("--filename", default="eval", type=str, help="Specify the eval filename")
    args = parser.parse_args()
    
    eval(args)