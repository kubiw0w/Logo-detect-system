from pathlib import Path
names = []
p = Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data/labels/train/classes.txt')
if p.exists():
    names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
else:
    names = ['Adidas','Amazon','Apple','Audi','BMW','Coca-Cola','DHL','Google','IKEA','InPost','KFC','McDonalds','Mercedes','NewYorker','Nike','Pepsi','UFC','Zara']
content = f"train: data/images/train\nval: data/images/val\n\nnc: {len(names)}\nnames: {names}\n"
Path('C:/Users/kubas/Desktop/projekty/projekt_logo/data.yaml').write_text(content)
print("Plik data.yaml został zapisany")
