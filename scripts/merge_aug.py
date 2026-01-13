from pathlib import Path
import shutil

for src in Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images_aug').glob('*.jpg'):
    shutil.copy(src, Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images/train')/src.name)
for src in Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels_aug').glob('*.txt'):
    shutil.copy(src, Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels/train')/src.name)
print("Scalono obrazy aug do folderu train")
