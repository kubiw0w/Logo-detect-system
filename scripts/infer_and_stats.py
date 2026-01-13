from ultralytics import YOLO
import cv2
from pathlib import Path
from collections import Counter
import pandas as pd

model = YOLO('C:/Users/kubas/Desktop/projekty/projekt_logo/runs/detect/train2/weights/best.pt')
names = ['Adidas','Amazon','Apple','Audi','BMW','Coca-Cola','DHL','Google','IKEA','InPost','KFC','McDonalds','Mercedes','NewYorker','Nike','Pepsi','UFC','Zara']

out_dir = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/outputs')
out_dir.mkdir(exist_ok=True)

total = Counter()
rows = []

for img_path in Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images/val').glob('*.jpg'):
    res = model(img_path)
    boxes = res[0].boxes
    counts = Counter()
    img = cv2.imread(str(img_path))
    for b in boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        counts[names[cls]] += 1
        total[names[cls]] += 1
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imwrite(str(out_dir / img_path.name), img)
    rows.append({'image': img_path.name, **dict(counts)})

df = pd.DataFrame(rows).fillna(0)
class_names = ['Adidas','Amazon','Apple','Audi','BMW','Coca-Cola','DHL','Google','IKEA','InPost','KFC','McDonalds','Mercedes','NewYorker','Nike','Pepsi','UFC','Zara']
df[class_names] = df[class_names].astype(int)
df.to_csv(out_dir / 'per_image_counts.csv', index=False)
pd.DataFrame(total.items(), columns=['brand','count']).to_csv(out_dir / 'total_counts.csv', index=False)
print("Zapisano wyniki i statystyki do folderu:", out_dir)
