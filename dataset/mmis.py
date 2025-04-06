import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset.transform import TransformDataset
import random

def get_mmis_dataset(args, mode: str = ""):
    train_dataset, val_dataset = None, None

    if mode in ["train", "both"]:
        train_dataset = MMISDataset(data_dir=args.data_dir, mode="train", mask_type=args.mask_type)
        train_dataset = TransformDataset(train_dataset, image_size=args.image_size)

    if mode in ["val", "both"]:
        val_dataset = MMISDataset(data_dir=args.data_dir, mode="val", mask_type=args.mask_type)
        val_dataset = TransformDataset(val_dataset, image_size=args.image_size)

    return train_dataset, val_dataset


import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MMISDataset(Dataset):
    def __init__(self, data_dir, mode="train", mask_type=""):
        self.data_dir = data_dir
        self.mode = mode
        self.mask_type = mask_type
        self.data = {"images": [], "masks": []}
        self.masks_per_image = 4  # Số lượng mask cho mỗi ảnh (a1, a2, a3, a4)
        
        print("🔥 MMISDataset loaded from:", __file__)
        # Duyệt qua các file trong thư mục con (train hoặc val)
        images_folder = os.path.join(data_dir, mode)
        self.load_data(images_folder)

    def load_data(self, images_folder):
        print(f"Loading data from {images_folder}")

        # Duyệt qua tất cả các file ảnh trong thư mục con (train/val)
        for filename in os.listdir(images_folder):
            if filename.endswith('_image.npy'):
                base_name = filename.replace('_image.npy', '')
                image_path = os.path.join(images_folder, filename)
                
                # Tải ảnh
                image = np.load(image_path)

                # Tải các mask tương ứng (label_a1, label_a2, label_a3, label_a4)
                masks = []
                for i in range(1, 5):
                    mask_path = os.path.join(images_folder, f"{base_name}_label_a{i}.npy")
                    if os.path.exists(mask_path):
                        mask = np.load(mask_path)
                        masks.append(mask)
                    else:
                        # Nếu mask không tồn tại, có thể bỏ qua hoặc tạo mask mặc định
                        masks.append(np.zeros_like(image))  # Tạo mask trống (tương đương với mask có giá trị 0)

                # Thêm ảnh và mask vào danh sách
                self.data["images"].append(image)
                self.data["masks"].append(masks)

        print(f"Loaded {len(self.data['images'])} images and masks")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):
        image = self.data["images"][index]
        # Chuẩn hóa ảnh
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        if image.ndim == 3 and image.shape[2] == 3:
            image = np.mean(image, axis=2)
        # image = image[np.newaxis, ...]

        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = self.data["masks"][index][mask_id]
        else:
            mask = self.data["masks"][index]
            if self.mask_type == "ensemble":
                mask = np.stack(mask, axis=-1).mean(axis=-1)
                mask = (mask > 0.5).astype(np.uint8)
            elif self.mask_type == "multi":
                pass
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        return mask, image

if __name__ == "__main__":
    class Args:
        data_dir = '/Users/kaiser_1/Documents/Data/mmis'  # Đường dẫn đến thư mục dữ liệu
        mask_type = "random"  # Loại mask cần sử dụng (ví dụ: "multi")
        image_size = 128  # Kích thước ảnh sau khi biến đổi
    
    args = Args()
    train_dataset, val_dataset = get_mmis_dataset_np(args, mode="both")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Kiểm tra một ví dụ
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Tạo DataLoader
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Kiểm tra batch đầu tiên
    for batch_images, batch_masks in dataloader:
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch masks shape: {batch_masks.shape}")
        break
