import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, image_size: int):
        self.dataset = dataset
        self.transform = A.Compose(transforms=[
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mask, image = self.dataset[idx]
        
        # Chuyển hình ảnh và mặt nạ thành numpy array nếu cần
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)  # Chuyển từ tensor (C, H, W) thành (H, W, C)
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()  # Chuyển mặt nạ thành numpy array nếu nó là tensor
        if isinstance(mask, list):
            mask = [m.numpy() if isinstance(m, torch.Tensor) else m for m in mask]  # Chuyển tất cả các mặt nạ trong danh sách thành numpy array nếu cần

        # Áp dụng biến đổi cho cả hình ảnh và mặt nạ cùng một lúc
        if isinstance(mask, list):
            transformed = self.transform(image=image, masks=mask)
            mask = [m.unsqueeze(0).to(torch.float32) for m in transformed["masks"]]
        else:
            transformed = self.transform(image=image, mask=mask)
            mask = transformed["mask"].unsqueeze(0).to(torch.float32)

        return mask, transformed["image"].to(torch.float32)


# import torch
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# import numpy as np

# class TransformDataset(Dataset):
#     def __init__(self, dataset: Dataset, image_size: int):
#         self.dataset = dataset
#         self.transform = A.Compose([
#             A.Resize(image_size, image_size),
#             A.HorizontalFlip(p=0.5),
#             ToTensorV2()
#         ])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         mask, image = self.dataset[idx]
        
#         # Chuyển đổi ảnh từ tensor sang numpy array (H, W, C)
#         image = image.numpy().transpose(1, 2, 0)
        
#         # Chuyển đổi mask sang numpy array và kiểm tra kích thước
#         if isinstance(mask, list):
#             mask = [m.numpy() for m in mask]
#         else:
#             mask = mask.numpy()
#             # Kiểm tra kích thước mask
#             if len(mask.shape) == 3:
#                 # Nếu mask có dạng (C, H, W), chuyển thành (H, W, C)
#                 if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
#                     mask = mask.transpose(1, 2, 0)
#                 # Nếu chỉ có 1 kênh, có thể squeeze nhưng cần kiểm tra
#                 if mask.shape[0] == 1:
#                     mask = mask.squeeze(0)
#                 elif mask.shape[2] == 1:
#                     mask = mask.squeeze(2)
        
#         # Áp dụng biến đổi
#         if isinstance(mask, list):
#             transformed = self.transform(image=image, masks=mask)
#             mask = [torch.tensor(m).unsqueeze(0).to(torch.float32) for m in transformed["masks"]]
#         else:
#             transformed = self.transform(image=image, mask=mask)
#             # Thêm lại chiều kênh nếu cần
#             mask = torch.tensor(transformed["mask"]).to(torch.float32)
#             if len(mask.shape) == 2:  # Nếu mask là 2D, thêm chiều kênh
#                 mask = mask.unsqueeze(0)
        
#         # Chuyển đổi ảnh đã biến đổi thành tensor
#         image = torch.tensor(transformed["image"]).to(torch.float32)
        
#         return mask, image