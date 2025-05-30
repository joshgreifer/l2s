import joblib
import numpy as np

from pkg.config import Config
from pkg.simple_dataset import SimpleDataset
from sklearn.decomposition import PCA


def get_landmarks_matrix(dataset: SimpleDataset):
    # Grab only the valid items (depending on self.full/self.idx)
    num_samples = len(dataset)
    # Assume each landmark entry is [468, 3] (or whatever your landmark shape is)
    example_landmark = dataset._db[0][0]
    num_landmarks, num_coords = example_landmark.shape

    landmarks_array = np.zeros((num_samples, num_landmarks, num_coords), dtype=np.float32)

    for i in range(num_samples):
        landmarks, _ = dataset._db[i]
        landmarks_array[i] = landmarks  # shape: [468, 3]

    return landmarks_array  # shape: [num_samples, 468, 3]


import matplotlib.pyplot as plt

def plot_pca_variation(pca, pc_idx, std_multiples=[-32, -4, 0, 4, 32], landmark_shape=(478, 3)):
    """
    Plot how the landmarks change along a single principal component.

    pca: fitted sklearn PCA object
    pc_idx: index of the principal component to visualize (0=PC1)
    std_multiples: list of multiples of the component's std dev to visualize
    landmark_shape: shape to unflatten to (468, 3)
    """
    mean_face = pca.mean_
    pc_vector = pca.components_[pc_idx]
    explained_std = np.sqrt(pca.explained_variance_[pc_idx])

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
    plt.show()


def do_pca():
    config = Config()
    if config.pca_path is None:
        raise ValueError("PCA path is not set in the configuration.")

    # Load dataset
    dataset = SimpleDataset(capacity=config.dataset_capacity)
    dataset.load(config.dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples from {config.dataset_path}")

    # Get landmarks matrix
    landmarks_matrix = get_landmarks_matrix(dataset)  # [num_samples, 478, 3]
    landmarks_flat = landmarks_matrix.reshape(landmarks_matrix.shape[0], -1)  # [num_samples, 1404]

    n_components = config.pca_num
    print(f"Performing PCA with {n_components} components on landmarks...")
    pca = PCA(n_components=n_components)
    _ = pca.fit_transform(landmarks_flat)  # [num_samples, n_components]

    joblib.dump(pca, config.pca_path)
    print(f"Done. PCA model saved to {config.pca_path}")

    return pca

if __name__ == "__main__":

    config = Config()
    if config.pca_path is None:
        raise ValueError("PCA path is not set in the configuration.")
    # if pca already saved, just load it (to recreate, delete the file)
    try:
        pca = joblib.load(config.pca_path)
        print(f"PCA model loaded from {config.pca_path}")
        for pc in range(pca.n_components_):
            plot_pca_variation(pca, pc_idx=pc)
        exit(0)
    except FileNotFoundError:
        print(f"No PCA model found at {config.pca_path}, creating a new one.")

    dataset = SimpleDataset(capacity=config.dataset_capacity)
    dataset.load(config.dataset_path)
    landmarks_matrix = get_landmarks_matrix(dataset)  # [num_samples, 478, 3]
    landmarks_flat = landmarks_matrix.reshape(landmarks_matrix.shape[0], -1)  # [num_samples, 1404]

    n_components = config.pca_num
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(landmarks_flat)  # [num_samples, n_components]
    # print(f"Original shape: {landmarks_flat.shape}, PCA reduced shape: {X_train_pca.shape}")
    joblib.dump(pca, config.pca_path)
    print(f"PCA model saved to {config.pca_path}")

    for pc in range(5):
        plot_pca_variation(pca, pc_idx=pc)