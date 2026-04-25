import os
import shutil
import random
import glob

# Paths
BASE = r"C:\Users\nidhi\OneDrive\Desktop\trackmate"
SHANGHAITECH = os.path.join(BASE, "data", "raw", "ShanghaiTech")
ANNOTATIONS = os.path.join(BASE, "data", "annotations")
PROCESSED = os.path.join(BASE, "data", "processed")

COACH_CAPACITY = 180
random.seed(42)  # so results are reproducible

def get_density_level(label_file):
    """Count persons in label file and return density level"""
    if not os.path.exists(label_file):
        return "low"
    with open(label_file) as f:
        count = len(f.readlines())
    density = min(100, round((count / COACH_CAPACITY) * 100))
    if density < 40:
        return "low"
    elif density < 70:
        return "medium"
    else:
        return "high"

def process_split(img_folder, ann_folder, split_name):
    """Process one split (e.g. part_A_train)"""
    images = glob.glob(os.path.join(img_folder, "*.jpg"))
    print(f"\n{split_name}: found {len(images)} images")

    low, medium, high = [], [], []

    for img_path in images:
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(ann_folder, label_name)
        level = get_density_level(label_path)
        if level == "low":
            low.append((img_path, label_path))
        elif level == "medium":
            medium.append((img_path, label_path))
        else:
            high.append((img_path, label_path))

    print(f"  Low: {len(low)} | Medium: {len(medium)} | High: {len(high)}")
    return low, medium, high

def copy_files(file_pairs, subset, density_level, coach_id):
    """Copy image and label to processed folder"""
    img_out = os.path.join(PROCESSED, subset, "images")
    lbl_out = os.path.join(PROCESSED, subset, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_path, label_path in file_pairs:
        img_name = os.path.basename(img_path)
        new_name = f"C{coach_id}_{density_level}_{img_name}"
        shutil.copy(img_path, os.path.join(img_out, new_name))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(lbl_out, new_name.replace(".jpg", ".txt")))

# Collect all images from all splits
all_low, all_medium, all_high = [], [], []

splits = [
    ("part_A/train_data/images", "part_A_train"),
    ("part_A/test_data/images",  "part_A_test"),
    ("part_B/train_data/images", "part_B_train"),
    ("part_B/test_data/images",  "part_B_test"),
]

for img_subfolder, ann_subfolder in splits:
    img_folder = os.path.join(SHANGHAITECH, img_subfolder)
    ann_folder = os.path.join(ANNOTATIONS, ann_subfolder)
    l, m, h = process_split(img_folder, ann_folder, ann_subfolder)
    all_low += l
    all_medium += m
    all_high += h

# Shuffle all
random.shuffle(all_low)
random.shuffle(all_medium)
random.shuffle(all_high)

# Split 80% train, 20% val
def split_80_20(lst):
    cut = int(0.8 * len(lst))
    return lst[:cut], lst[cut:]

low_train, low_val = split_80_20(all_low)
med_train, med_val = split_80_20(all_medium)
high_train, high_val = split_80_20(all_high)

# Assign to 6 coaches
# C1, C2 → low density coaches
# C3, C4 → medium density coaches
# C5, C6 → high density coaches
print("\nCopying files to processed folder...")

copy_files(low_train[:len(low_train)//2],  "train", "low",    1)
copy_files(low_train[len(low_train)//2:],  "train", "low",    2)
copy_files(med_train[:len(med_train)//2],  "train", "medium", 3)
copy_files(med_train[len(med_train)//2:],  "train", "medium", 4)
copy_files(high_train[:len(high_train)//2],"train", "high",   5)
copy_files(high_train[len(high_train)//2:],"train", "high",   6)

copy_files(low_val[:len(low_val)//2],      "val", "low",    1)
copy_files(low_val[len(low_val)//2:],      "val", "low",    2)
copy_files(med_val[:len(med_val)//2],      "val", "medium", 3)
copy_files(med_val[len(med_val)//2:],      "val", "medium", 4)
copy_files(high_val[:len(high_val)//2],    "val", "high",   5)
copy_files(high_val[len(high_val)//2:],    "val", "high",   6)

print("\n✅ Dataset split complete!")
print(f"Train images: {len(glob.glob(os.path.join(PROCESSED, 'train', 'images', '*.jpg')))}")
print(f"Val images:   {len(glob.glob(os.path.join(PROCESSED, 'val', 'images', '*.jpg')))}")