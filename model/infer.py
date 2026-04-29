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
    """
    Runs YOLO on image, saves annotated result,
    returns density info
    """
    results = model(image_path, verbose=False)
    person_count = len(results[0].boxes)

    # Density based on person count thresholds
    if person_count < 15:
        level = "low"
        density_pct = max(5, round((person_count / COACH_CAPACITY) * 100))
    elif person_count < 50:
        level = "medium"
        density_pct = round((person_count / COACH_CAPACITY) * 100)
    else:
        level = "high"
        density_pct = min(100, round((person_count / COACH_CAPACITY) * 100))

    # Save annotated image
    annotated = results[0].plot()
    save_path = os.path.join(
        RESULTS_DIR,
        f'Coach_{coach_id}_{density_pct}pct_{level}.jpg'
    )
    Image.fromarray(annotated).save(save_path)

    return {
        "coach": f"C{coach_id}",
        "image_used": os.path.basename(image_path),
        "person_count": person_count,
        "density_pct": density_pct,
        "level": level,
        "saved_to": save_path
    }

def find_images_by_density(target_level):
    """
    Scans val images and finds ones matching
    the target density level by running YOLO
    """
    all_val = glob.glob('data/processed/val/images/*.jpg')
    random.shuffle(all_val)

    matching = []
    for img_path in all_val[:100]:  # check first 100
        results = model(img_path, verbose=False)
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
    """
    Simulates 6 coaches with randomised
    low / medium / high density results
    """
    # Clear previous results
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nTrackMate — Scanning images by density...")
    print("=" * 50)

    print("Finding low density images...")
    low_imgs = find_images_by_density("low")

    print("Finding medium density images...")
    med_imgs = find_images_by_density("medium")

    print("Finding high density images...")
    high_imgs = find_images_by_density("high")

    # Fallback if not enough found
    all_imgs = glob.glob('data/processed/val/images/*.jpg')
    while len(low_imgs) < 2:
        low_imgs.append(random.choice(all_imgs))
    while len(med_imgs) < 2:
        med_imgs.append(random.choice(all_imgs))
    while len(high_imgs) < 2:
        high_imgs.append(random.choice(all_imgs))

    print("\nRunning inference on 6 coaches...")
    print("=" * 50)

    # Create pool and shuffle randomly
    pool = [
        (low_imgs[0], "low"),
        (low_imgs[1], "low"),
        (med_imgs[0], "medium"),
        (med_imgs[1], "medium"),
        (high_imgs[0], "high"),
        (high_imgs[1], "high"),
    ]

    # Shuffle so coaches get random density each run
    random.shuffle(pool)

    # Assign shuffled images to coaches
    assignments = [
        (i+1, pool[i][0])
        for i in range(6)
    ]

    coaches = {}
    for coach_id, img_path in assignments:
        result = get_density(img_path, coach_id)
        coaches[f'C{coach_id}'] = result
        print(f"Coach {coach_id}: {result['density_pct']}% ({result['level']}) — {result['person_count']} persons")

    print("=" * 50)
    print(f"\nAnnotated images saved to: {RESULTS_DIR}/")
    print("Opening results folder...")
    os.startfile(RESULTS_DIR)

    return coaches

if __name__ == "__main__":
    get_all_coaches()
    