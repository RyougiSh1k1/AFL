#!/usr/bin/env python
"""
Script to download and prepare TinyImageNet dataset
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def download_file(url, dest):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def prepare_val_folder(val_dir):
    """
    Reorganize validation folder structure to match training folder structure
    """
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # Read validation annotations
    val_dict = {}
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                class_name = parts[1]
                if class_name not in val_dict:
                    val_dict[class_name] = []
                val_dict[class_name].append(filename)
    
    # Create class folders and move images
    for class_name, filenames in val_dict.items():
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for filename in filenames:
            src = os.path.join(val_dir, 'images', filename)
            dst = os.path.join(class_dir, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    # Remove the now-empty images folder and annotations file
    images_dir = os.path.join(val_dir, 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(val_annotations_file):
        os.remove(val_annotations_file)

def main():
    # Create data directory
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # TinyImageNet URL
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')
    
    # Check if already exists
    if os.path.exists(extract_path):
        print(f"TinyImageNet already exists at {extract_path}")
        return
    
    # Download if not exists
    if not os.path.exists(zip_path):
        print(f"Downloading TinyImageNet from {url}...")
        download_file(url, zip_path)
    else:
        print(f"Zip file already exists at {zip_path}")
    
    # Extract
    print("Extracting TinyImageNet...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Prepare validation folder
    print("Reorganizing validation folder...")
    val_dir = os.path.join(extract_path, 'val')
    prepare_val_folder(val_dir)
    
    # Clean up zip file
    os.remove(zip_path)
    
    print(f"TinyImageNet successfully prepared at {extract_path}")
    print("Dataset structure:")
    print(f"  - Training: {extract_path}/train")
    print(f"  - Validation: {extract_path}/val")
    print(f"  - Test: {extract_path}/test")

if __name__ == '__main__':
    main()