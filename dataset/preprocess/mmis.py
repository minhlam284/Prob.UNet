import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Hàm xử lý ảnh và mask từ file .h5
def normalize_and_concat_modalities(h5f):
    datasets = [key for key in h5f.keys() if key in ["t1", "t2", "t1c"]]
    data_list = []

    for key in datasets:
        img = np.array(h5f[key])
        img = img - img.min()
        if img.max() > 0:
            img = (img / img.max()) * 255
        img = img.astype(np.uint8)
        img = np.expand_dims(img, axis=-1)
        data_list.append(img)

    if len(data_list) == 3:
        concatenated = np.concatenate(data_list, axis=-1)
        return concatenated
    else:
        return None

# Hàm chính để xử lý tất cả các file .h5 trong folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Duyệt qua tất cả các file trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.endswith('.h5'):
            input_h5_path = os.path.join(input_folder, filename)
            
            # Xử lý từng file .h5
            with h5py.File(input_h5_path, 'r') as f:
                # Kết hợp các modality và lưu file .npy cho ảnh
                image = normalize_and_concat_modalities(f)
                if image is not None:
                    base_name = os.path.splitext(filename)[0]
                    np.save(os.path.join(output_folder, f"{base_name}_image.npy"), image)
                    
                    # Lưu các mask
                    for label_name in ['label_a1', 'label_a2', 'label_a3', 'label_a4']:
                        label = np.array(f[label_name])
                        np.save(os.path.join(output_folder, f"{base_name}_{label_name}.npy"), label)

            print(f"Processed {filename}")

input_folder = '/Users/kaiser_1/Documents/Data/val'  # Thư mục chứa các file .h5
output_folder = '/Users/kaiser_1/Documents/Data/mmis/val'  # Thư mục để lưu các file .npy

process_folder(input_folder, output_folder)