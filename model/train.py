from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train on our crowd dataset
results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='trackmate_v1',
    patience=10,
    save=True,
    device='cpu'
)

print("Training complete!")
print("Best weights saved at:", results.save_dir)