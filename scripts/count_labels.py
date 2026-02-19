#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from collections import Counter
import sys

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/processed_batched')
if not base.exists():
    print(f"Data dir not found: {base}")
    sys.exit(1)

splits = ['train','val','test']
all_counts = Counter()
split_counts = {}

for s in splits:
    p = base / s
    cnt = Counter()
    if not p.exists():
        split_counts[s] = cnt
        continue
    files = list(p.glob('*.npz'))
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            lbl = int(data['label'])
            cnt[lbl] += 1
            all_counts[lbl] += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
    split_counts[s] = cnt

# Print per-split
print(f"Data directory: {base.resolve()}")
print()

total = sum(all_counts.values())
print(f"Total samples: {total}")
for cls in sorted(all_counts.keys()):
    c = all_counts[cls]
    pct = 100.0 * c / total if total>0 else 0.0
    print(f"Class {cls}: {c} ({pct:.2f}%)")

print('\nPer-split breakdown:')
for s in splits:
    sc = split_counts[s]
    stotal = sum(sc.values())
    print(f"\n{s.upper()} - {stotal} samples")
    for cls in sorted(set(list(sc.keys()) + list(all_counts.keys()))):
        c = sc.get(cls,0)
        pct = 100.0 * c / stotal if stotal>0 else 0.0
        print(f"  Class {cls}: {c} ({pct:.2f}%)")

# Provide class rates (per sample probability)
print('\nClass rates (probability):')
for cls in sorted(all_counts.keys()):
    rate = all_counts[cls]/total if total>0 else 0.0
    print(f"  Class {cls}: {rate:.4f}")
