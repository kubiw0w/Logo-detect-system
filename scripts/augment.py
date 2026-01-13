import cv2
from pathlib import Path
import albumentations as A

src_images = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images/train')
src_labels = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels/train')
out_images = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images_aug')
out_labels = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels_aug')

out_images.mkdir(exist_ok=True, parents=True)
out_labels.mkdir(exist_ok=True, parents=True)

print(f"Folder z obrazami wejściowymi: {src_images.resolve()}")
images = list(src_images.glob('*.jpg'))
print(f"Znaleziono obrazów: {len(images)}")
if len(images) == 0:
    print("Nie znaleziono obrazów w folderze data/images/train")
    exit()

transform = A.Compose([
    A.Rotate(limit=25, p=0.7),
    A.RandomBrightnessContrast(p=0.6),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([A.RandomBrightnessContrast(p=0.7), A.RandomGamma(p=0.7)], p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for img_path in images:
    name = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Nie można wczytać obrazu {img_path}")
        continue
    label_path = src_labels / f'{name}.txt'
    if not label_path.exists():
        print(f"Brak etykiety: {label_path}")
        continue
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        print(f"Pusta etykieta: {label_path}")
        continue
    bboxes, class_labels = [], []
    for l in lines:
        try:
            c, x, y, bw, bh = l.split()
            bboxes.append([float(x), float(y), float(bw), float(bh)])
            class_labels.append(int(c))
        except Exception as e:
            print("Błąd parsowania etykiety:", l, e)
    print(f"Przetwarzanie {name} z {len(bboxes)} boxami")
    for i in range(3):  # ile kopii
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_classes = augmented['class_labels']
        out_name = f'{name}_aug{i}'
        cv2.imwrite(str(out_images / f'{out_name}.jpg'), aug_img)
        with open(out_labels / f'{out_name}.txt', 'w') as ol:
            for cls, bb in zip(aug_classes, aug_bboxes):
                x, y, bw, bh = bb
                ol.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Zapisano {out_name}.jpg i {out_name}.txt")
print("Gotowe.")
