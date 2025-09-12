# pkg/tools/bake_pca.py
import copy
import torch
import numpy as np
from pkg.model_pca_mlp import GazePCAMLP
from pkg.config import Config

def bake_pca_into_first_linear(model: GazePCAMLP) -> GazePCAMLP:
    """
    Mutates a loaded GazePCAMLP so that PCA is baked into the first Linear layer.
    After this, model expects raw flattened landmarks (B, 478*3) and does NOT
    call PCA at runtime (we replace it with identity).
    """
    assert isinstance(model.mlp[0], torch.nn.Linear), "Unexpected MLP[0] layer"
    first: torch.nn.Linear = model.mlp[0]

    # --- Pull PCA params (NumPy) ---
    pca = model.pca
    C = torch.from_numpy(pca.components_.astype(np.float32))         # (k, n_features)
    mu = torch.from_numpy(pca.mean_.astype(np.float32))              # (n_features,)
    device = first.weight.device
    C = C.to(device)
    mu = mu.to(device)

    k, n_features = C.shape  # k = n_components
    out = first.out_features

    # --- Compose affine transforms:  z = (x - mu) @ C.T;  y = z @ W1.T + b1
    # So: y = x @ (C.T @ W1.T) + (b1 - mu @ (C.T @ W1.T))
    # In PyTorch (row-major): W_new = W1 @ C,  b_new = b1 - (mu @ W_new.T)
    with torch.no_grad():
        W1 = first.weight                        # (out, k)
        b1 = first.bias                          # (out,)
        W_new = W1 @ C                           # (out, n_features)
        b_new = b1 - (mu @ W_new.T)              # (out,)

        # Replace first layer to take raw features directly
        new_first = torch.nn.Linear(n_features, out, bias=True).to(device)
        new_first.weight.copy_(W_new)
        new_first.bias.copy_(b_new)

        # Swap it in
        model.mlp[0] = new_first

        # Make runtime PCA a no-op (keeps the rest of forward untouched)
        class _IdentityPCA:
            def transform(self, X_np: np.ndarray) -> np.ndarray:
                # forward() expects numpy and will wrap back to torch
                return X_np  # shape: (B, n_features)

        model.pca = _IdentityPCA()

    return model

if __name__ == "__main__":
    # Smoke test: outputs should match closely before vs after baking.
    cfg = Config("cache/config.json")  # uses your cache/config.json
    base = GazePCAMLP(cfg).eval()       # loads weights & real PCA
    baked = copy.deepcopy(base)
    baked = bake_pca_into_first_linear(baked).eval()

    x = torch.randn(2, 478, 3)  # batch of landmarks
    with torch.no_grad():
        y0 = base(x)
        y1 = baked(x)

    print("max |Δ| =", (y0 - y1).abs().max().item())
