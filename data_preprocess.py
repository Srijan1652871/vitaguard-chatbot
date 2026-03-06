import os
import cv2
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Paths
RAW_DATASET = "dataset"
PROCESSED_DATASET = "processed_dataset"

IMG_SIZE = 224
TEST_SPLIT = 0.2

def create_folders():
    for split in ["train", "test"]:
        for disease in os.listdir(RAW_DATASET):
            path = os.path.join(PROCESSED_DATASET, split, disease)
            os.makedirs(path, exist_ok=True)

def clean_and_resize():
    print("Cleaning and resizing images...")
    
    for disease in os.listdir(RAW_DATASET):
        disease_path = os.path.join(RAW_DATASET, disease)
        
        images = []
        
        for img_name in os.listdir(disease_path):
            img_path = os.path.join(disease_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append((img, img_name))
                
            except:
                continue
        
        # Split train/test
        train_imgs, test_imgs = train_test_split(images, test_size=TEST_SPLIT)
        
        # Save processed images
        for img, name in train_imgs:
            save_path = os.path.join(PROCESSED_DATASET, "train", disease, name)
            cv2.imwrite(save_path, img)
            
        for img, name in test_imgs:
            save_path = os.path.join(PROCESSED_DATASET, "test", disease, name)
            cv2.imwrite(save_path, img)

def main():
    create_folders()
    clean_and_resize()
    print("Dataset preprocessing completed!")

if __name__ == "__main__":
    main()