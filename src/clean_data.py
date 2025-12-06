from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from imagededup.methods import PHash
from models.base import CLASS_LABELS

data_dir = Path("./data")
fer2013 = pd.read_csv(data_dir / "fer2013.csv")
fer_plus = pd.read_csv(data_dir / "fer2013new.csv")
fer2013 = fer2013.drop(columns=["emotion"])
fer_plus = fer_plus.drop(columns=["Image name"])

CLASS_LABELS = CLASS_LABELS + ["NF"]

data = pd.concat([fer2013, fer_plus[CLASS_LABELS]], axis=1)

for c in data[CLASS_LABELS]:
    data[c] = pd.to_numeric(data[c], errors="coerce")
print(f"NaN values: \n{data[CLASS_LABELS].isna().sum()}")


def decode_image(pixel_str):
    pixels = np.array([int(p) for p in pixel_str.split()], dtype=np.uint8)
    return pixels.reshape(48, 48)


np_images = {idx: decode_image(row["pixels"]) for idx, row in data.iterrows()}

# Remove images labeled as 'NF' (Not a Face)
data = data[data["NF"] < 10]

# Deduplicate images using perceptual hashing
phasher = PHash()
hashes = {
    str(key): phasher.encode_image(image_array=img) for key, img in np_images.items()
}

duplicates = phasher.find_duplicates(encoding_map=hashes, max_distance_threshold=2)

keys_order = [str(k) for k in np_images.keys()]
rank = {k: i for i, k in enumerate(keys_order)}

graph = defaultdict(set)
for a, neighs in duplicates.items():
    for b in neighs:
        graph[a].add(b)
        graph[b].add(a)

seen = set()
dup_groups = []
for node in keys_order:
    if node in seen or node not in graph:
        continue
    comp = []
    q = deque([node])
    seen.add(node)
    while q:
        u = q.popleft()
        comp.append(u)
        for v in graph[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    comp.sort(key=lambda k: rank.get(k, float("inf")))
    dup_groups.append(comp)


print(f"Original dataset size: {len(data)}")
print(f"Duplicate groups: {len(dup_groups)}")

indices_to_remove = {int(k) for comp in dup_groups for k in comp[1:]}
to_remove = set(indices_to_remove) & set(data.index)
data = data.drop(index=list(to_remove), columns=["NF"]).reset_index(drop=True)

print(f"Removed {len(indices_to_remove)} duplicate images")
print(f"After deduplication: {len(data)}")

data.to_csv(data_dir / "fer2013_clean.csv")
