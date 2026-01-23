import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

# Base URLs
ANNOTATION_BASE_URL = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell"
IMAGES_URL = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip"

# Annotation files
ANNOTATIONS = {
    "train": f"{ANNOTATION_BASE_URL}/livecell_coco_train.json",
    "val": f"{ANNOTATION_BASE_URL}/livecell_coco_val.json",
    "test": f"{ANNOTATION_BASE_URL}/livecell_coco_test.json"
}

def create_directory_structure(base_path: Path):
    """Create the directory structure for the dataset."""
    dirs = [
        base_path / "annotations",
        base_path / "train" / "images",
        base_path / "val" / "images",
        base_path / "test" / "images",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure at {base_path}")

def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded {destination.name}")

def download_annotations(base_path: Path):
    """Download COCO annotation files."""
    print("\nDownloading annotations...")
    annotations_dir = base_path / "annotations"
    
    for split, url in ANNOTATIONS.items():
        destination = annotations_dir / f"livecell_coco_{split}.json"
        if destination.exists():
            print(f"{destination.name} already exists, skipping...")
            continue
        download_file(url, destination, f"Downloading {split} annotations")

def download_and_extract_images(base_path: Path):
    """Download and extract image files."""
    print("\nDownloading images (this may take a while)...")
    
    zip_path = base_path / "images.zip"
    
    # Download zip file
    download_file(IMAGES_URL, zip_path, "Downloading images.zip")

    # Extract zip file
    print("\nExtracting images...")
    temp_extract_path = base_path / "temp_images"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)
    
    print("Extracted images!")
    
    # Organize images into train/val/test folders
    print("\nOrganizing images...")
    
    # The structure is: temp_images/images/livecell_train_val_images/ and temp_images/images/livecell_test_images/
    train_val_source = temp_extract_path / "images" / "livecell_train_val_images"
    test_source = temp_extract_path / "images" / "livecell_test_images"
    
    if not train_val_source.exists() or not test_source.exists():
        # Try alternative structure
        train_val_source = temp_extract_path / "livecell_train_val_images"
        test_source = temp_extract_path / "livecell_test_images"
    
    # Move test images
    if test_source.exists():
        test_dest = base_path / "test" / "images"
        print(f"Moving test images to {test_dest}...")
        for img in test_source.glob("*"):
            if img.is_file():
                shutil.move(str(img), str(test_dest / img.name))
        print(f"Moved test images")
    
    # Split train/val images based on annotations
    if train_val_source.exists():
        import json
        
        # Load train and val annotations to get image lists
        train_json = base_path / "annotations" / "livecell_coco_train.json"
        val_json = base_path / "annotations" / "livecell_coco_val.json"
        
        train_images = set()
        val_images = set()
        
        if train_json.exists():
            with open(train_json, 'r') as f:
                train_data = json.load(f)
                train_images = {img['file_name'] for img in train_data['images']}
        
        if val_json.exists():
            with open(val_json, 'r') as f:
                val_data = json.load(f)
                val_images = {img['file_name'] for img in val_data['images']}
        
        # Move images to appropriate folders
        train_dest = base_path / "train" / "images"
        val_dest = base_path / "val" / "images"
        
        print(f"Organizing {len(train_images)} train and {len(val_images)} val images...")
        
        for img in tqdm(list(train_val_source.glob("*")), desc="Moving images"):
            if img.is_file():
                if img.name in train_images:
                    shutil.move(str(img), str(train_dest / img.name))
                elif img.name in val_images:
                    shutil.move(str(img), str(val_dest / img.name))
        
        print("✓ Organized train/val images")
    
    # Clean up temporary directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_extract_path)

    zip_path.unlink()
    print("✓ Deleted images.zip")

def main():
    """Main download function."""

    print("LIVECell Dataset Downloader")

    # Assumes script is in scripts/ directory, parent.parent gets to project root
    BASE_DIR = Path(__file__).resolve().parent.parent
    base_path = BASE_DIR / "data"
    
    print(f"Project root: {BASE_DIR}")
    print(f"Data will be downloaded to: {base_path.absolute()}")
    
    # Create directory structure
    create_directory_structure(base_path)
    
    # Download annotations
    download_annotations(base_path)
    
    # Download and extract images
    download_and_extract_images(base_path)

    print("Dataset download complete!")
    print(f"Data saved to: {base_path.absolute()}")

if __name__ == "__main__":
    main()