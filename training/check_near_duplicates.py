"""
Near-duplicate check between the processed vision splits (evidence for DATA.md section 1.2).
Flip-invariant 256-bit difference hash; writes datasets/processed/flood_yolo/near_duplicates.json.
Usage (from the repo root):  python training/check_near_duplicates.py
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "flood_yolo"


def dhash_arr(a, size=16):
    g = np.asarray(Image.fromarray(a).resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def load_gray(p):
    im = Image.open(p)
    im.draft("L", (120, 90))
    return np.asarray(im.convert("L"))


def hashes(split, flips):
    names, hs = [], []
    for p in sorted((root / split / "images").iterdir()):
        a = load_gray(p)
        variants = [a] if not flips else [a, a[:, ::-1], a[::-1, :], a[::-1, ::-1]]
        names.append(p.name)
        hs.append(np.stack([dhash_arr(np.ascontiguousarray(v)) for v in variants]))
    return names, np.stack(hs)


def main():
    print("hashing train", flush=True)
    _, T = hashes("train", True)
    print("train hashed", T.shape, flush=True)
    result = {}
    for split in ("test", "val"):
        names, Q = hashes(split, False)
        d = np.array([(T != q[0]).sum(-1).min() for q in Q])
        result[split] = {
            "n": len(d),
            "min_hamming_quantiles_0_5_25_50": np.percentile(d, [0, 5, 25, 50]).tolist(),
            "le_10": int((d <= 10).sum()),
            "le_20": int((d <= 20).sum()),
            "le_40": int((d <= 40).sum()),
        }
        print(split, result[split], flush=True)
    (root / "near_duplicates.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
