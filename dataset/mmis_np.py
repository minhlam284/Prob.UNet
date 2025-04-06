import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transform import TransformDataset
import random
# def get_mmis_dataset(args, mode: str = ""):
#     train_dataset, val_dataset = None, None

#     if mode in ["train", "both"]:
#         train_dataset = MMISDataset(data_dir=args.data_dir, mode="train", mask_type=args.mask_type)
#         train_dataset = TransformDataset(train_dataset, image_size=args.image_size)

#     if mode in ["val", "both"]:
#         val_dataset = MMISDataset(data_dir=args.data_dir, mode="val", mask_type=args.mask_type)
#         val_dataset = TransformDataset(val_dataset, image_size=args.image_size)

#     return train_dataset, val_dataset


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
                    mask_path = os.path.join(images_folder, f"{base_name}label_a{i}.npy")
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

    # def __getitem__(self, index):
    #     image = self.data["images"][index]

    #     # Chuẩn hóa ảnh
    #     if image.max() > image.min():
    #         image = (image - image.min()) / (image.max() - image.min())

    #     if self.mask_type == "random":
    #         mask_id = random.randint(0, self.masks_per_image - 1)
    #         mask = self.data["masks"][index][mask_id]
    #     else:
    #         mask = self.data["masks"][index]
    #         if self.mask_type == "ensemble":
    #             mask = np.stack(mask, axis=-1).mean(axis=-1)
    #             mask = (mask > 0.5).astype(np.uint8)
    #         elif self.mask_type == "multi":
    #             pass
        
    #     return mask, image

    def __getitem__(self, index):
        image = self.data["images"][index]

        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # Always convert image to float32
        image = image.astype(np.float32)

        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = self.data["masks"][index][mask_id]
            mask = mask.astype(np.uint8)

        elif self.mask_type == "ensemble":
            masks = self.data["masks"][index]
            mask = np.stack(masks, axis=-1).mean(axis=-1)
            mask = (mask > 0.5).astype(np.uint8)

        elif self.mask_type == "multi":
            # Return 4-channel mask
            masks = self.data["masks"][index]
            mask = np.stack(masks, axis=0).astype(np.uint8)  # shape: (4, H, W)

        else:
            # Default: use first mask only
            mask = self.data["masks"][index][0].astype(np.uint8)

        return mask, image

        # # Chuyển đổi ảnh thành tensor
        # image = torch.from_numpy(image).float()
        # if len(image.shape) == 2:
        #     image = image.unsqueeze(0)  # Thêm chiều kênh nếu ảnh là grayscale
        # elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # Nếu ảnh có nhiều kênh
        #     image = image.permute(2, 0, 1)  # Chuyển từ (H, W, C) thành (C, H, W)

        # # Xử lý các mask
        # mask_tensors = []
        # for mask in masks:
        #     mask_tensor = torch.from_numpy(mask).float()
        #     if len(mask_tensor.shape) == 2:
        #         mask_tensor = mask_tensor.unsqueeze(0)  # Thêm chiều kênh
        #     mask_tensors.append(mask_tensor)

        # # Kiểm tra nếu không có mask nào được thêm vào thì báo lỗi
        # if len(mask_tensors) == 0:
        #     raise ValueError(f"No masks found for image {index}")

        # # Kết hợp tất cả các mask thành tensor có shape (4, H, W)
        # mask = torch.stack(mask_tensors)  # (4, H, W)

        # return image, mask

if __name__ == "__main__":
    pickle_path = '/Users/kaiser_1/Documents/Data/mmis'
    
    dataset = MMISDataset(data_dir=pickle_path, mask_type="random")
    
    print(f"Dataset size: {len(dataset)}")
    

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")



# Ví dụ sử dụng
# if __name__ == "__main__":
#     class Args:
#         data_dir = '/Users/kaiser_1/Documents/Data/mmis'  # Đường dẫn đến thư mục dữ liệu
#         mask_type = "random"  # Loại mask cần sử dụng (ví dụ: "multi")
#         image_size = (128, 128)  # Kích thước ảnh sau khi biến đổi
    
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
