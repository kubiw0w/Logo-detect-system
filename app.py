import gradio as gr
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from collections import Counter

model = YOLO('C:/Users/kubas/Desktop/projekty/projekt_logo/runs/detect/train2/weights/best.pt')
class_names = ['Adidas','Amazon','Apple','Audi','BMW','Coca-Cola','DHL','Google','IKEA','InPost','KFC','McDonalds','Mercedes','NewYorker','Nike','Pepsi','UFC','Zara']

out_dir = Path('data/outputs')
out_dir.mkdir(exist_ok=True)

def detect_logo(image):
    img = image.copy()
    results = model(img)
    counts = Counter()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        counts[class_names[cls]] += 1

        # rysowanie bounding boxów
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{class_names[cls]} {conf:.2f}",
                    (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # tworzenie DataFrame dla csv
    row = {'image': 'input_image'}
    for name in class_names:
        row[name] = counts.get(name, 0)
    df = pd.DataFrame([row])
    df.to_csv(out_dir / 'stats.csv', index=False)

    return img, str(dict(counts))

title = "System wykrywania logo i marek"
description = "Wgraj zdjęcie, aby wykryć loga marek i zobaczyć statystyki."

iface = gr.Interface(
    fn=detect_logo,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    live=False,
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()
