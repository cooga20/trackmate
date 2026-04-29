import os
import shutil
import glob
import random

# Paths
BASE = r"C:\Users\nidhi\OneDrive\Desktop\trackmate"
CCTV_TRAIN = os.path.join(BASE, "data", "raw", "CCTV Indoor Person Detection.v1i.yolov8", "train")
CCTV_VAL = os.path.join(BASE, "data", "raw", "CCTV Indoor Person Detection.v1i.yolov8", "valid")
PROCESSED = os.path.join(BASE, "data", "processed")

random.seed(42)

def copy_low_density(src_img_folder, src_lbl_folder, subset, max_images=150):
    """
    Copy low density CCTV images into processed folder
    """
    images = glob.glob(os.path.join(src_img_folder, "*.jpg"))
    
    # Also check for png
    images += glob.glob(os.path.join(src_img_folder, "*.png"))
    
    random.shuffle(images)
    images = images[:max_images]  # take only 150 max
    
    out_img = os.path.join(PROCESSED, subset, "images")
    out_lbl = os.path.join(PROCESSED, subset, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    copied = 0
    for img_path in images:
        img_name = os.path.basename(img_path)
        new_name = f"C1_low_cctv_{img_name}"
        
        # Copy image
        shutil.copy(img_path, os.path.join(out_img, new_name))
        
        # Copy label if exists
        lbl_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        lbl_path = os.path.join(src_lbl_folder, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(
                out_lbl, new_name.replace(".jpg", ".txt").replace(".png", ".txt")
            ))
        else:
            # Create empty label file (empty coach = no persons)
            open(os.path.join(out_lbl, new_name.replace(".jpg", ".txt").replace(".png", ".txt")), 'w').close()
        
        copied += 1
    
    print(f"Copied {copied} low density images to {subset} folder")
    return copied

print("Adding low density CCTV images to processed folder...")
print("=" * 50)

# Add to train (120 images)
copy_low_density(
    os.path.join(CCTV_TRAIN, "images"),
    os.path.join(CCTV_TRAIN, "labels"),
    "train",
    max_images=120
)

# Add to val (30 images)
copy_low_density(
    os.path.join(CCTV_VAL, "images"),
    os.path.join(CCTV_VAL, "labels"),
    "val",
    max_images=30
)

print("=" * 50)

# Count total images now
train_total = len(glob.glob(os.path.join(PROCESSED, "train", "images", "*.jpg")))
train_total += len(glob.glob(os.path.join(PROCESSED, "train", "images", "*.png")))
val_total = len(glob.glob(os.path.join(PROCESSED, "val", "images", "*.jpg")))
val_total += len(glob.glob(os.path.join(PROCESSED, "val", "images", "*.png")))

print(f"\nUpdated dataset totals:")
print(f"Train images: {train_total}")
print(f"Val images:   {val_total}")
print(f"\n✅ Low density images added successfully!")
print("Now retrain on Google Colab with the updated dataset!")