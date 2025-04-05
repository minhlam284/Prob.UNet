import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MMISDataset(Dataset):

    def __init__(self, pickle_path, transform=None, mask_type="ensemble"):

        self.pickle_path = pickle_path
        self.transform = transform
        self.mask_type = mask_type
        self.data = {"images": [], "masks": []}
        self.masks_per_image = 4 
        

        if os.path.isdir(pickle_path):
            pickle_files = [os.path.join(pickle_path, f) for f in os.listdir(pickle_path) 
                           if f.endswith('.pickle') or f.endswith('.pkl')]
            for file_path in pickle_files:
                self.load_data(file_path)
        else:
            self.load_data(pickle_path)
    
    def load_data(self, pickle_path):
        max_bytes = 2**31 - 1
        
        print(f"Loading data from {pickle_path}")
        bytes_in = bytearray(0)
        input_size = os.path.getsize(pickle_path)
        with open(pickle_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        
        new_data = pickle.loads(bytes_in)
        self.data["images"].extend(new_data["images"])
        self.data["masks"].extend(new_data["masks"])
        
        print(f"Loaded {len(new_data['images'])} images and masks from {os.path.basename(pickle_path)}")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):

        image = self.data["images"][index]

        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        

        if self.mask_type == "random":

            import random
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = self.data["masks"][index][mask_id]
        else:
            mask = self.data["masks"][index]
            if self.mask_type == "ensemble":
                mask = np.stack(mask, axis=-1).mean(axis=-1)
                mask = (mask > 0.5).astype(np.uint8)
            elif self.mask_type == "multi":
                pass
        
        image = torch.from_numpy(image).float()
        
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]: 
            image = image.permute(2, 0, 1) 
        
        # Xử lý mask
        if self.mask_type == "multi":
            mask_tensors = []
            for m in mask:
                m_tensor = torch.from_numpy(m).float()
                if len(m_tensor.shape) == 2:
                    m_tensor = m_tensor.unsqueeze(0)  
                mask_tensors.append(m_tensor)
            mask = torch.stack(mask_tensors)  # (4, 1, H, W)
        else:
            # Chuyển mask thành tensor
            mask = torch.from_numpy(mask).float()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0) 
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask


if __name__ == "__main__":
    pickle_path = '/Users/kaiser_1/Documents/Data uncertainty & privacy/Code data/data/mmis'
    
    dataset = MMISDataset(pickle_path=pickle_path, mask_type="ensemble")
    
    print(f"Dataset size: {len(dataset)}")
    

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Tạo DataLoader
    # # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    # # Kiểm tra batch đầu tiên
    # for batch_images, batch_masks in dataloader:
    #     print(f"Batch images shape: {batch_images.shape}")
    #     print(f"Batch masks shape: {batch_masks.shape}")
    #     break