import os
from glob import glob
from PIL import Image
from tqdm import tqdm

def check_images(dataset_folder):
    images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
    
    for image_path in tqdm(images_paths):
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            print(f"Image path: {image_path}, error: {e}")

dataset_folder = '/mnt/sda3/2024_Projects/npr/datasets/msls/images/train'

check_images(dataset_folder)