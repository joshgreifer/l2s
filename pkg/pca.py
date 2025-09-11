import joblib
import numpy as np
import torch

from pkg.config import Config
from pkg.dataset import SimpleDataset
from sklearn.decomposition import PCA

from pkg.util import log

class LandMarkPCA:
    def __init__(self, config: Config):
        self.config = config
        self.pca = None
        if config.pca_path is None:
            raise ValueError("PCA path is not set in the configuration.")
        # if pca already saved, just load it (to recreate, delete the file)
        try:
            self.pca = joblib.load(config.pca_path)
        except FileNotFoundError:
            log().warn(f"No PCA model found at {config.pca_path}, creating a new one.")
        if self.pca is None:
            dataset = SimpleDataset(capacity=config.train.dataset_capacity)
            dataset.load(config.dataset_path)

            self.fit(dataset)

    def fit(self, dataset: SimpleDataset):
        landmarks_matrix = self.get_landmarks_matrix(dataset)  # [num_samples, 468, 3]
        landmarks_flat = landmarks_matrix.reshape(landmarks_matrix.shape[0], -1)  # [num_samples, 1404]

        n_components = self.config.model.pca_num
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(landmarks_flat)  # [num_samples, n_components]
        joblib.dump(self.pca, self.config.pca_path)
        log().info(f"PCA model saved to {self.config.pca_path}")
        return X_train_pca

    def get_landmarks_matrix(self, dataset: SimpleDataset):
        # Grab only the valid items (depending on self.full/self.idx)
        num_samples = len(dataset)
        # Assume each landmark entry is [468, 3] (or whatever your landmark shape is)
        example_landmark = dataset._db[0][0]
        num_landmarks, num_coords = example_landmark.shape

        landmarks_array = np.zeros((num_samples, num_landmarks, num_coords), dtype=np.float32)

        for i in range(num_samples):
            landmarks, _ = dataset._db[i]
            if isinstance(landmarks, torch.Tensor):
                landmarks = landmarks.cpu().numpy()
            landmarks_array[i] = landmarks  # shape: [468, 3]

        return landmarks_array  # shape: [num_samples, 468, 3]

    def plot_variations(self, n_components=1000, std_multiples=[-16, -4, 0, 4, 16]):
        """
        Plot variations along the first few principal components.
        """
        n_components = min(n_components, self.pca.n_components_)
        for i in range(n_components):
            self.plot_variation(i, std_multiples=std_multiples)

    def plot_variation(self, pc_idx, *, std_multiples=[-16, -4, 0, 4, 16], landmark_shape=(478, 3)):
        """
        Plot how the landmarks change along a single principal component.

        pca: fitted sklearn PCA object
        pc_idx: index of the principal component to visualize (0=PC1)
        std_multiples: list of multiples of the component's std dev to visualize
        landmark_shape: shape to unflatten to (468, 3)
        """

        mean_face = self.pca.mean_
        pc_vector = self.pca.components_[pc_idx]
        explained_std = np.sqrt(self.pca.explained_variance_[pc_idx])

        plt.figure(figsize=(15, 3))
        for i, std_mult in enumerate(std_multiples):
            # Compute new face
            landmarks = mean_face + std_mult * explained_std * pc_vector
            landmarks = landmarks.reshape(landmark_shape)

            plt.subplot(1, len(std_multiples), i+1)
            plt.scatter(landmarks[:, 0], -landmarks[:, 1], s=3)  # y-flip for typical image coords
            plt.title(f"{std_mult:+}σ")
            plt.axis('equal')
            plt.axis('off')

        plt.suptitle(f"Variation along Principal Component {pc_idx+1}")
        plt.savefig(f"cache/pc{pc_idx+1}_variation.png", bbox_inches='tight')


import matplotlib.pyplot as plt


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "cache/config.json"

    config = Config(config_file)
    dataset = SimpleDataset(capacity=config.train.dataset_capacity)
    dataset.load(config.dataset_path)

    landmark_pca = LandMarkPCA(config)

    landmark_pca.plot_variations()


