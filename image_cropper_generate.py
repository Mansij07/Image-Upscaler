import os
from multiprocessing import Pool
from PIL import Image
import numpy as np

SRC = "data/DIV2K_train_HR"
DEST = "data/train_sub"
CROP = 480
STRIDE = 240
MIN_STD = 0


def positions(length):
    pos = list(range(0, length - CROP + 1, STRIDE))
    if pos[-1] != length - CROP:
        pos.append(length - CROP)
    return pos

def process(fname):
    img = Image.open(os.path.join(SRC, fname)).convert("RGB")
    w, h = img.size
    if w < CROP or h < CROP:
        return 0
    stem = os.path.splitext(fname)[0]
    count = 0
    for ri, y in enumerate(positions(h)):
        for ci, x in enumerate(positions(w)):
            crop = img.crop((x, y, x + CROP, y + CROP))
            if MIN_STD > 0 and np.asarray(crop).std() < MIN_STD:
                continue
            crop.save(os.path.join(DEST, f"{stem}_r{ri}_c{ci}.png"), compress_level=1)
            count += 1
    return count 

if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(".png"))
    with Pool() as pool:
        counts = pool.map(process, files)
    print(f"{len(files)} images -> {sum(counts)} crops")