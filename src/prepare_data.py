import os
import shutil

DATASET_DIR = "./archive/Garbage_classification"
OUTPUT_DIR = "./data"       
TRAIN_TXT = "./archive/one-indexed-files-notrash_train.txt"
VAL_TXT   = "./archive/one-indexed-files-notrash_val.txt"
TEST_TXT  = "./archive/one-indexed-files-notrash_test.txt"

id_to_class = {
    "1": "glass",
    "2": "paper",
    "3": "cardboard",
    "4": "plastic",
    "5": "metal",
    "6": "trash"
}

def load_pairs(txt_file):
    pairs = []
    with open(txt_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            fname, cid = line.strip().split()
            pairs.append((fname, cid))
    return pairs

def move_images(pairs, split):
    for fname, cid in pairs:
        cname = id_to_class[cid]
        src = os.path.join(DATASET_DIR, cname, fname)
        dst_dir = os.path.join(OUTPUT_DIR, split, cname)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dst_dir, fname))
        else:
            print(f"No image found: {src}")

train_pairs = load_pairs(TRAIN_TXT)
val_pairs   = load_pairs(VAL_TXT)
test_pairs  = load_pairs(TEST_TXT)

train_full_pairs = train_pairs + val_pairs

print(f"Train (train+val): {len(train_full_pairs)} images")
print(f"Test: {len(test_pairs)} images")

move_images(train_full_pairs, "train")
move_images(test_pairs, "test")

print("\nData has been divided into:")
print(f"  - Train: {len(train_full_pairs)} images")
print(f"  - Test:  {len(test_pairs)} images")
