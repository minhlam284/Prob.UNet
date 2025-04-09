from typing import Literal
import glob
import numpy as np
import random
from torch.utils.data import Dataset
from .transform import TransformDataset

def get_qubiq_dataset(args, mode: str = ""):

    val_dataset = QUBIQDataset(data_dir=args.data_dir, train_val_test_dir="Val", mask_type=args.mask_type)
    val_dataset = TransformDataset(val_dataset, image_size=args.image_size)
    if mode == "val":   
        return val_dataset
    
    train_dataset = QUBIQDataset(data_dir=args.data_dir, train_val_test_dir="Train", mask_type=args.mask_type)
    train_dataset = TransformDataset(train_dataset, image_size=args.image_size)
    if mode == "train":
        return train_dataset
    
    return train_dataset, val_dataset

class QUBIQDataset(Dataset):

    dataset_url = "https://mmis2024.com/info?task=1"
    masks_per_image = 2

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: Literal["ensemble", "random", "multi"] = "ensemble",
    ) -> None:
        super().__init__()

        if train_val_test_dir:
            img_dirs = [f"{data_dir}/{train_val_test_dir}/*/image*.npy"]
        else:
            img_dirs = [
                f"{data_dir}/Train/*/image*.npy",
                f"{data_dir}/Val/*/image*.npy",
            ]

        self.img_paths = [
            img_path for img_dir in img_dirs
            for img_path in glob.glob(img_dir)
        ]
        self.mask_type = mask_type

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        image = np.load(img_path)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        mask_path = img_path.replace('image', 'label')
        mask = np.load(mask_path).astype(np.uint8)
        
        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = mask[:, :, mask_id]
        elif self.mask_type == "ensemble":
            mask = mask.mean(axis=-1)
            mask = (mask > 0.5).astype(np.uint8)
        elif self.mask_type == "multi":
            mask = [mask[:,:, i] for i in range(self.masks_per_image)]
        
        return mask, image

if __name__ == "__main__":
    dataset = QUBIQDataset(data_dir='/Users/kaiser_1/Documents/Data/data/qubiq', mask_type="multi")
    print(len(dataset))
    id = random.randint(0, len(dataset) - 1)
    print(id)

    masks, images = dataset[id]
    print(images.shape, images.dtype, type(images))
    
    masks = np.stack(masks, axis=-1)
    mask_e = masks.mean(axis=-1)
    mask_var = masks.var(axis=-1)
    print(masks.shape, mask_var.shape, masks.shape)

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(images[:,:], cmap='gray')
    axs[0, 0].set_title('T1')
    axs[0, 0].axis("off")
    
    sns.heatmap(mask_var, ax=axs[1, 0], cmap="viridis")
    axs[1, 0].set_title('Variance Heatmap')
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mask_e, cmap='gray')
    axs[1, 1].set_title('Mask_e')
    axs[1, 1].axis("off")

    axs[1, 2].imshow(masks[:,:,0], cmap='gray')
    axs[1, 2].set_title('Mask_1')
    axs[1, 2].axis("off")
    
    axs[2, 0].imshow(masks[:,:,1], cmap='gray')
    axs[2, 0].set_title('Mask_2')
    axs[2, 0].axis("off")
    
    plt.tight_layout()
    plt.show()
