import os, random, shutil
from pathlib import Path

random.seed(42)
src_images = Path('data/raw_images')
train_dir = Path('data/images/train')
val_dir = Path('data/images/val')

images = [f for f in src_images.glob('*.jpg')]
random.shuffle(images)
split = int(len(images)*0.8)
train = images[:split]
val = images[split:]

for p in train:
    shutil.copy(p, train_dir / p.name)
for p in val:
    shutil.copy(p, val_dir / p.name)
