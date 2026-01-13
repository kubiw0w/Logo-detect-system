from pathlib import Path

imgs = list(Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/images/train').glob('*.jpg'))
errs = []
for img in imgs:
    txt = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels/train') / (img.stem + '.txt')
    if not txt.exists():
        errs.append(f"Brak etykiety dla {img.name}")
    else:
        with open(txt) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            errs.append(f"Pusty plik etykiety: {txt}")

print('Znaleziono obrazów:', len(imgs))
print('Błędów:', len(errs))
for e in errs[:50]:
    print(e)
