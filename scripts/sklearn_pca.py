#!/usr/bin/env python3
import json, sys
import numpy as np
from sklearn.decomposition import PCA

def main():
    data = json.load(sys.stdin)
    landmarks = np.array(data['landmarks'], dtype=np.float32)
    sample_count = len(landmarks) // (478 * 3)
    landmarks = landmarks.reshape(sample_count, 478 * 3)
    pca = PCA(n_components=32)
    features = pca.fit_transform(landmarks)
    total_var = float(np.var(landmarks, axis=0, ddof=1).sum())
    out = {
        'features': features.flatten().tolist(),
        'total_var': total_var,
    }
    json.dump(out, sys.stdout)

if __name__ == '__main__':
    main()
