import scipy.io as sio
import os
import glob

# Paths
PART_A_TRAIN = r"C:\Users\nidhi\OneDrive\Desktop\trackmate\data\raw\ShanghaiTech\part_A\train_data"
PART_A_TEST  = r"C:\Users\nidhi\OneDrive\Desktop\trackmate\data\raw\ShanghaiTech\part_A\test_data"
PART_B_TRAIN = r"C:\Users\nidhi\OneDrive\Desktop\trackmate\data\raw\ShanghaiTech\part_B\train_data"
PART_B_TEST  = r"C:\Users\nidhi\OneDrive\Desktop\trackmate\data\raw\ShanghaiTech\part_B\test_data"

OUTPUT_DIR = r"C:\Users\nidhi\OneDrive\Desktop\trackmate\data\annotations"

COACH_CAPACITY = 180  # Namma Metro coach capacity

def convert_mat_to_yolo(data_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    gt_folder = os.path.join(data_path, "ground-truth")
    img_folder = os.path.join(data_path, "images")

    mat_files = glob.glob(os.path.join(gt_folder, "*.mat"))
    print(f"Found {len(mat_files)} annotation files in {data_path}")

    for mat_file in mat_files:
        # Load the .mat annotation file
        mat = sio.loadmat(mat_file)
        points = mat['image_info'][0][0][0][0][0]  # person coordinates

        # Get image name
        img_name = os.path.basename(mat_file).replace("GT_", "").replace(".mat", ".jpg")
        img_path = os.path.join(img_folder, img_name)

        # Get image size
        from PIL import Image
        try:
            img = Image.open(img_path)
            W, H = img.size
        except:
            W, H = 1024, 768  # fallback size

        # Write YOLO label file
        label_file = os.path.join(output_folder, img_name.replace(".jpg", ".txt"))
        with open(label_file, "w") as f:
            for point in points:
                x, y = point[0], point[1]
                # YOLO format: class cx cy width height (normalised)
                cx = x / W
                cy = y / H
                bw = 0.05  # estimated box width
                bh = 0.1   # estimated box height
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # Calculate density
        count = len(points)
        density = min(100, round((count / COACH_CAPACITY) * 100))
        level = "low" if density < 40 else "medium" if density < 70 else "high"
        print(f"{img_name}: {count} persons → {density}% ({level})")

# Convert all 4 splits
convert_mat_to_yolo(PART_A_TRAIN, os.path.join(OUTPUT_DIR, "part_A_train"))
convert_mat_to_yolo(PART_A_TEST,  os.path.join(OUTPUT_DIR, "part_A_test"))
convert_mat_to_yolo(PART_B_TRAIN, os.path.join(OUTPUT_DIR, "part_B_train"))
convert_mat_to_yolo(PART_B_TEST,  os.path.join(OUTPUT_DIR, "part_B_test"))

print("\n✅ All annotations converted to YOLO format!")