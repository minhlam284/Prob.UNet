# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transform import TransformDataset
# import random

# def get_mmis_dataset(args, mode: str = ""):
#     train_dataset, val_dataset = None, None

#     if mode in ["train", "both"]:
#         train_dataset = MMISDataset(data_dir=args.data_dir, mode="train", mask_type=args.mask_type)
#         train_dataset = TransformDataset(train_dataset, image_size=args.image_size)

#     if mode in ["val", "both"]:
#         val_dataset = MMISDataset(data_dir=args.data_dir, mode="val", mask_type=args.mask_type)
#         val_dataset = TransformDataset(val_dataset, image_size=args.image_size)

#     return train_dataset, val_dataset


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class MMISDataset(Dataset):
#     def __init__(self, data_dir, mode="train", mask_type=""):
#         self.data_dir = data_dir
#         self.mode = mode
#         self.mask_type = mask_type
#         self.data = {"images": [], "masks": []}
#         self.masks_per_image = 4  # Số lượng mask cho mỗi ảnh (a1, a2, a3, a4)
        
#         print("🔥 MMISDataset loaded from:", __file__)
#         # Duyệt qua các file trong thư mục con (train hoặc val)
#         images_folder = os.path.join(data_dir, mode)
#         self.load_data(images_folder)

#     def load_data(self, images_folder):
#         print(f"Loading data from {images_folder}")

#         # Duyệt qua tất cả các file ảnh trong thư mục con (train/val)
#         for filename in os.listdir(images_folder):
#             if filename.endswith('_image.npy'):
#                 base_name = filename.replace('_image.npy', '')
#                 image_path = os.path.join(images_folder, filename)
                
#                 # Tải ảnh
#                 image = np.load(image_path)

#                 # Tải các mask tương ứng (label_a1, label_a2, label_a3, label_a4)
#                 masks = []
#                 for i in range(1, 5):
#                     mask_path = os.path.join(images_folder, f"{base_name}_label_a{i}.npy")
#                     if os.path.exists(mask_path):
#                         mask = np.load(mask_path)
#                         masks.append(mask)
#                     else:
#                         # Nếu mask không tồn tại, có thể bỏ qua hoặc tạo mask mặc định
#                         masks.append(np.zeros_like(image))  # Tạo mask trống (tương đương với mask có giá trị 0)

#                 # Thêm ảnh và mask vào danh sách
#                 self.data["images"].append(image)
#                 self.data["masks"].append(masks)

#         print(f"Loaded {len(self.data['images'])} images and masks")

#     def __len__(self):
#         return len(self.data["images"])

#     def __getitem__(self, index):
#         image = self.data["images"][index]
#         # Chuẩn hóa ảnh
#         if image.max() > image.min():
#             image = (image - image.min()) / (image.max() - image.min())

#         if image.ndim == 3 and image.shape[2] == 3:
#             image = np.mean(image, axis=2)
#         # image = image[np.newaxis, ...]

#         if self.mask_type == "random":
#             mask_id = random.randint(0, self.masks_per_image - 1)
#             mask = self.data["masks"][index][mask_id]
#         else:
#             mask = self.data["masks"][index]
#             if self.mask_type == "ensemble":
#                 mask = np.stack(mask, axis=-1).mean(axis=-1)
#                 mask = (mask > 0.5).astype(np.uint8)
#             elif self.mask_type == "multi":
#                 pass
#         print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
#         return mask, image

# if __name__ == "__main__":
#     class Args:
#         data_dir = '/Users/kaiser_1/Documents/Data/mmis'  # Đường dẫn đến thư mục dữ liệu
#         mask_type = "random"  # Loại mask cần sử dụng (ví dụ: "multi")
#         image_size = 128  # Kích thước ảnh sau khi biến đổi
    
#     args = Args()
#     train_dataset, val_dataset = get_mmis_dataset(args, mode="both")

#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")

#     # Kiểm tra một ví dụ
#     image, mask = train_dataset[0]
#     print(f"Image shape: {image.shape}")
#     print(f"Mask shape: {mask.shape}")
    
#     # Tạo DataLoader
#     dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

#     # Kiểm tra batch đầu tiên
#     for batch_images, batch_masks in dataloader:
#         print(f"Batch images shape: {batch_images.shape}")
#         print(f"Batch masks shape: {batch_masks.shape}")
#         break


from typing import Literal
import glob
import numpy as np
import random
from torch.utils.data import Dataset
from .transform import TransformDataset

def get_mmis_dataset(args, mode: str = ""):

    val_dataset = MMISDataset(data_dir=args.data_dir, train_val_test_dir="Val", mask_type=args.mask_type)
    val_dataset = TransformDataset(val_dataset, image_size=args.image_size)
    if mode == "val":   
        return val_dataset
    
    train_dataset = MMISDataset(data_dir=args.data_dir, train_val_test_dir="Train", mask_type=args.mask_type)
    train_dataset = TransformDataset(train_dataset, image_size=args.image_size)
    if mode == "train":
        return train_dataset
    
    return train_dataset, val_dataset

class MMISDataset(Dataset):

    dataset_url = "https://mmis2024.com/info?task=1"
    masks_per_image = 4

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
    dataset = MMISDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', mask_type="multi")
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
    
    axs[0, 0].imshow(images[:,:,0], cmap='gray')
    axs[0, 0].set_title('T1')
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(images[:,:,1], cmap='gray')
    axs[0, 1].set_title('T1C')
    axs[0, 1].axis("off")
    
    axs[0, 2].imshow(images[:,:,2], cmap='gray')
    axs[0, 2].set_title('T2')
    axs[0, 2].axis("off")
    
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
    
    axs[2, 1].imshow(masks[:,:,2], cmap='gray')
    axs[2, 1].set_title('Mask_3')
    axs[2, 1].axis("off")
    
    axs[2, 2].imshow(masks[:,:,3], cmap='gray')
    axs[2, 2].set_title('Mask_4')
    axs[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()
