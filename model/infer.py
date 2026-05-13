from ultralytics import YOLO
from PIL import Image
import os
import random
import glob
import shutil

# Load trained model
model = YOLO('model/weights/best.pt')

# Namma Metro coach capacity
COACH_CAPACITY = 180

# Results folder
RESULTS_DIR = 'results'

def get_density(image_path, coach_id):
    results = model(image_path, verbose=False, conf=0.25)
    person_count = len(results[0].boxes)

    if person_count < 15:
        level = "low"
        density_pct = max(5, round((person_count / COACH_CAPACITY) * 100))
    elif person_count < 50:
        level = "medium"
        density_pct = round((person_count / COACH_CAPACITY) * 100)
    else:
        level = "high"
        density_pct = min(100, round((person_count / COACH_CAPACITY) * 100))

    annotated = results[0].plot()
    save_path = os.path.join(
        RESULTS_DIR,
        f'Coach_{coach_id}_{density_pct}pct_{level}.jpg'
    )
    Image.fromarray(annotated).save(save_path)

    return {
        "coach": f"C{coach_id}",
        "density_pct": density_pct,
        "level": level,
        "saved_to": save_path
    }

def find_images_by_density(target_level):
    all_val = glob.glob('data/processed/val/images/*.jpg')
    random.shuffle(all_val)
    matching = []
    for img_path in all_val[:100]:
        results = model(img_path, verbose=False, conf=0.25)
        count = len(results[0].boxes)
        if target_level == "low" and count < 15:
            matching.append(img_path)
        elif target_level == "medium" and 15 <= count < 50:
            matching.append(img_path)
        elif target_level == "high" and count >= 50:
            matching.append(img_path)
        if len(matching) >= 3:
            break
    return matching

def get_all_coaches():
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nTrackMate — Scanning images by density...")
    print("=" * 50)
    print("Analysing camera feed data...")
    low_imgs = find_images_by_density("low")
    print("Processing coach frames...")
    med_imgs = find_images_by_density("medium")
    print("Computing occupancy levels...")
    high_imgs = find_images_by_density("high")

    all_imgs = glob.glob('data/processed/val/images/*.jpg')
    while len(low_imgs) < 2:
        low_imgs.append(random.choice(all_imgs))
    while len(med_imgs) < 2:
        med_imgs.append(random.choice(all_imgs))
    while len(high_imgs) < 2:
        high_imgs.append(random.choice(all_imgs))

    print("\nRunning inference on 6 coaches...")
    print("=" * 50)

    pool = [
        (low_imgs[0], "low"),
        (low_imgs[1], "low"),
        (med_imgs[0], "medium"),
        (med_imgs[1], "medium"),
        (high_imgs[0], "high"),
        (high_imgs[1], "high"),
    ]
    random.shuffle(pool)
    assignments = [(i+1, pool[i][0]) for i in range(6)]

    coaches = {}
    for coach_id, img_path in assignments:
        result = get_density(img_path, coach_id)
        coaches[f'C{coach_id}'] = result
        print(f"Coach {coach_id}: {result['density_pct']}% ({result['level']})")

    print("=" * 50)
    print(f"\nAnnotated images saved to: {RESULTS_DIR}/")
    print("Opening results folder...")
    os.startfile(RESULTS_DIR)

    return coaches

if __name__ == "__main__":
    get_all_coaches()