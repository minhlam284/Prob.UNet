import nibabel as nib
import numpy as np
from pathlib import Path

def process_sample_v2(sample_dir, output_dir):
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dữ liệu
    image = nib.load(sample_dir / "image.nii.gz").get_fdata()
    mask1 = nib.load(sample_dir / "task01_seg01.nii.gz").get_fdata()
    mask2 = nib.load(sample_dir / "task01_seg02.nii.gz").get_fdata()

    assert image.shape == mask1.shape == mask2.shape, f"Mismatch at {sample_dir}"

    for i in range(image.shape[0]):
        img_slice = image[i,:, :]
        m1_slice = mask1[i,:, :]
        m2_slice = mask2[i,:, :]

        # Save ảnh 2D
        np.save(output_dir / f"image_{i}.npy", img_slice)

        # Save label 2 mask
        label = np.stack([m1_slice, m2_slice], axis=-1)
        np.save(output_dir / f"label_{i}.npy", label)


def process_all_samples(dataset_root, output_root):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    for sample_dir in dataset_root.iterdir():
        if sample_dir.is_dir():
            sample_output = output_root / sample_dir.name
            print(f"Đang xử lý: {sample_dir.name}")
            process_sample_v2(sample_dir, sample_output)

if __name__ == '__main__':
    data_dir = Path("../Data")

    process_all_samples(
        dataset_root=data_dir / "MMIS2024TASK1/training",
        output_root=data_dir / "Train"
    )
    process_all_samples(
        dataset_root=data_dir / "MMIS2024TASK1/validation",
        output_root=data_dir / "Val"
    )


