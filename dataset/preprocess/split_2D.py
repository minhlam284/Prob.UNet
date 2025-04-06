import os
import h5py
import numpy as np
from tqdm import tqdm

def split_h5_by_slices(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    for file in tqdm(h5_files, desc="🔪 Splitting H5 by slice"):
        input_path = os.path.join(input_folder, file)
        file_base = os.path.splitext(file)[0]  # "filename.h5" -> "filename"

        with h5py.File(input_path, "r") as h5f:
            # Lấy dữ liệu 3D
            t1 = h5f["t1"][:]
            t2 = h5f["t2"][:]
            t1c = h5f["t1c"][:]
            label_a1 = h5f["label_a1"][:]
            label_a2 = h5f["label_a2"][:]
            label_a3 = h5f["label_a3"][:]
            label_a4 = h5f["label_a4"][:]

            num_slices = t1.shape[0]

            for i in range(num_slices):
                output_name = f"{file_base}_slice_{i}.h5"
                output_path = os.path.join(output_folder, output_name)

                with h5py.File(output_path, "w") as out_h5:
                    out_h5.create_dataset("t1", data=t1[i])
                    out_h5.create_dataset("t2", data=t2[i])
                    out_h5.create_dataset("t1c", data=t1c[i])
                    out_h5.create_dataset("label_a1", data=label_a1[i])
                    out_h5.create_dataset("label_a2", data=label_a2[i])
                    out_h5.create_dataset("label_a3", data=label_a3[i])
                    out_h5.create_dataset("label_a4", data=label_a4[i])

                print(f"Đã lưu slice {i} của {file} -> {output_name}")

    print("Hoàn tất quá trình tách slice!")

input_h5_folder = "/Users/kaiser_1/Documents/Data/MMIS2024TASK1/validation"
output_slices_folder = "/Users/kaiser_1/Documents/Data/val"
split_h5_by_slices(input_h5_folder, output_slices_folder)