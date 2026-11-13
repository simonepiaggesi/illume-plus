"""A small end-to-end check of the installed public API."""

from importlib.metadata import version

import numpy as np
import torch

import illume_plus
from illume_plus import ILLUMEplus
from illume_plus import core


def test_fit_transform_on_cpu(monkeypatch):
    monkeypatch.setattr(core, "device", torch.device("cpu"))

    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(8, 3)).astype(np.float32)
    x_val = rng.normal(size=(4, 3)).astype(np.float32)
    model = ILLUMEplus(latent_dim=2, rank_dim=2, max_epochs=1, batch_size=4)
    losses_train, losses_val = model.fit(
        X=(x_train, x_val),
        idx_num_cat=[[0], [1], [2]],
        params_dict={"num_k": None, "l_rec": 0.0, "l_kld": 1.0,
                     "l_so": 0.0, "l_co": 0.0, "l_st": 0.0},
        seed=42,
    )

    weights, codes, decoder_weights = model.transform(
        x_val, return_decoder_weights=True
    )
    reconstructed = model.inverse_transform(codes, decoder_weights)

    assert illume_plus.__version__ == version("mimosa-illume-plus")
    assert len(losses_train) == len(losses_val) == 1
    assert np.isfinite(losses_train).all()
    assert np.isfinite(losses_val).all()
    assert weights.shape == decoder_weights.shape == (4, 2, 3)
    assert codes.shape == (4, 2)
    assert reconstructed.shape == x_val.shape
    assert np.isfinite(codes).all()
    assert np.isfinite(reconstructed).all()
