# Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions

Code and data for running [ILLUME+](https://arxiv.org/abs/2607.00931), a novel post-hoc explanation method for cancer drug response prediction from transcriptomics tabular data.

The [ILLUME+ example notebook](examples/illume_plus_example.ipynb) uses the
included `data/gdsc/<drug_id>/processed.pkl.gz` files. These contain the
training, validation, and test data, feature names, and fitted scaler. Run it from the repository
root or the `examples` directory.

The optional [transcriptomics preparation notebook](examples/load_transcriptomics_example.ipynb)
shows how to regenerate one drug's `processed.pkl.gz` from
`data/transcriptomics.feather` and the included split files. It selects the
drug ID, applies the training scaler to all three splits, and saves the
resulting dictionaries. The Feather file is not included; obtain it from
[Zenodo](https://zenodo.org/records/18605674) and place it in `data/` before
running this preparation notebook.

Paper: Martino Ciaperoni, Margherita Lalli, Simone Piaggesi, Martina Varisco, Francesco Carli, Riccardo Guidotti, Dino Pedreschi, Francesco Raimondi, Fosca Giannotti. "Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions." In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2026. [[link]](https://dl.acm.org/doi/abs/10.1145/3770855.3819010/)

## Installation

```bash
pip install mimosa-illume-plus
```

From a local clone, use `pip install -e ".[dev]"` to run the smoke test.
Use `pip install -e ".[notebook]"` for both notebooks. To enable the
commented-out black-box training example, install `pip install -e ".[notebook,blackbox]"`.

## Basic usage

```python
import numpy as np
from illume_plus import ILLUMEplus

X_train = np.random.default_rng(1).normal(size=(8, 3)).astype("float32")
X_val = np.random.default_rng(2).normal(size=(4, 3)).astype("float32")
model = ILLUMEplus(latent_dim=2, rank_dim=2, max_epochs=1, batch_size=4)
model.fit(
    X=(X_train, X_val),
    idx_num_cat=[[0], [1], [2]],
    params_dict={"num_k": None, "l_rec": 0.0, "l_kld": 1.0,
                 "l_so": 0.0, "l_co": 0.0, "l_st": 0.0},
)
W, Z = model.transform(X_val)
assert W.shape == (4, 2, 3) and Z.shape == (4, 2)
```

## Cite us

If you use the code in this repository, please consider citing us:

```bibtex
@inproceedings{ciaperoni2026explainable,
  title={Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions},
  author={Ciaperoni, Martino and Lalli, Margherita and Piaggesi, Simone and Varisco, Martina and Carli, Francesco and Guidotti, Riccardo and Pedreschi, Dino and Raimondi, Francesco and Giannotti, Fosca},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={10749--10760},
  year={2026}
}
```
