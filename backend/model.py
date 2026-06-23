"""
model.py
--------
The PyTorch Autoencoder that compresses the concatenated feature vector
(standardized metadata one-hot/numeric features + semantic title
embedding) into a low-dimensional latent space. HDBSCAN later clusters in
that latent space (cluster_service.py), and a 2D projection of it becomes
the x/y map coordinates the frontend renders.

Design notes
------------
- Encoder and decoder are simple symmetric MLPs (Linear -> ReLU stacks).
  That's intentionally simple: the inputs are already standardized
  numeric/one-hot/embedding floats, not images or sequences, so there's no
  benefit to convolutional or attention layers here.
- All tensor ops run on `get_device()` (CUDA if available, else CPU) so the
  same code scales from a laptop to a training box without edits.
- Reconstruction loss is MSE between input and decoder output — standard
  for a continuous-valued autoencoder.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    AUTOENCODER_HIDDEN_DIMS,
    AUTOENCODER_WEIGHTS_PATH,
    BATCH_SIZE,
    EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
)

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Single place that decides CPU vs GPU, so every tensor op in this
    module (and embeddings.py) stays consistent with whatever hardware is
    actually available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


class Autoencoder(nn.Module):
    """Symmetric encoder/decoder MLP.

    input_dim -> hidden_dims[0] -> hidden_dims[1] -> ... -> latent_dim
                                                          -> ... mirrored -> input_dim
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = LATENT_DIM):
        super().__init__()
        hidden_dims = hidden_dims or AUTOENCODER_HIDDEN_DIMS

        encoder_layers = []
        dims = [input_dim] + hidden_dims
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            encoder_layers += [nn.Linear(in_d, out_d), nn.ReLU(inplace=True)]
        encoder_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        rev_dims = [latent_dim] + hidden_dims[::-1]
        for in_d, out_d in zip(rev_dims[:-1], rev_dims[1:]):
            decoder_layers += [nn.Linear(in_d, out_d), nn.ReLU(inplace=True)]
        decoder_layers.append(nn.Linear(rev_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)


def train_autoencoder(
    feature_matrix: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    test_split: float = 1 - TRAIN_TEST_SPLIT,
) -> tuple:
    """
    Trains the autoencoder on `feature_matrix` (N, input_dim) and returns
    (model, device, history) where history tracks per-epoch train/test
    reconstruction loss. 70/30 train/test split per the spec; all tensors
    are moved to `get_device()` before any forward/backward pass.
    """
    torch.manual_seed(RANDOM_SEED)
    device = get_device()
    logger.info("Training autoencoder on device: %s", device)

    n = feature_matrix.shape[0]
    input_dim = feature_matrix.shape[1]

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    split_point = int(n * (1 - test_split))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    X = torch.tensor(feature_matrix, dtype=torch.float32)
    train_ds = TensorDataset(X[train_idx])
    test_ds = TensorDataset(X[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)               # <-- GPU op
            optimizer.zero_grad()
            recon, _ = model(batch)                 # <-- GPU op
            loss = criterion(recon, batch)          # <-- GPU op
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(train_ds)

        test_loss = evaluate_reconstruction_loss(model, test_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info("epoch %3d/%3d  train_loss=%.5f  test_loss=%.5f", epoch, epochs, train_loss, test_loss)

    torch.save(model.state_dict(), AUTOENCODER_WEIGHTS_PATH)
    logger.info("Saved autoencoder weights to %s", AUTOENCODER_WEIGHTS_PATH)
    return model, device, history


@torch.no_grad()
def evaluate_reconstruction_loss(model: Autoencoder, loader: DataLoader, device: torch.device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
        n_samples += batch.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def encode_all(model: Autoencoder, feature_matrix: np.ndarray, device: torch.device) -> np.ndarray:
    """Runs the full dataset through the trained encoder (in batches, on
    `device`) and returns the latent vectors as a numpy array."""
    model.eval()
    X = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    latents = []
    chunk = 2048
    for i in range(0, X.shape[0], chunk):
        z = model.encode(X[i : i + chunk])
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test with random data so this module can be sanity-checked
    # without the full pipeline.
    dummy = np.random.randn(500, 40).astype(np.float32)
    m, dev, hist = train_autoencoder(dummy, epochs=5)
    z = encode_all(m, dummy, dev)
    print("Latent shape:", z.shape)
