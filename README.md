# BindGraph

BindGraph is a sparse pair-embedding graph neural network for protein binder
screening. It is designed as the model-training and candidate-ranking module in
an AI-guided binder engineering workflow.

BindGraph can use structure-derived single and pair representations from
different backbone or complex-structure models. In the current examples:

- CHD: Boltz2 single embeddings as node features and Boltz2 pair embeddings as
  edge features.
- NTD: AF3 monomer raw single embeddings as node features and AF3 monomer raw
  pair embeddings as edge features.

The graph construction, GNN architecture, cross-validation training loop, and
ensemble prediction logic are shared across targets.

## Algorithm

1. Load one `.npz` embedding file per sequence or candidate.
2. Use the single embedding as node features.
3. Use the pair embedding as edge features.
4. Build a sparse directed graph:
   - sequence-local edges: `i-2`, `i-1`, `i+1`, `i+2`
   - non-local edges: top-8 pair-embedding L2-norm neighbors per node
5. Train an edge-conditioned attention GNN with attention pooling.
6. Select each fold checkpoint by validation Spearman rho.
7. Predict candidates with the mean of 5 fold models.

See [docs/algorithm.md](docs/algorithm.md) for a fuller description.

## Repository layout

```text
BindGraph/
  configs/                  Example CHD and NTD configs
  docs/                     Algorithm notes
  scripts/                  CLI entry points
  src/bindgraph/            Core package
```

Large data files are intentionally excluded from the repository:

- `.npz` embedding files
- graph `.pkl` files
- trained `.pth` checkpoints
- full prediction CSVs

Put those under `data/`, `outputs/`, or `checkpoints/`; they are ignored by
`.gitignore`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU training, install a PyTorch build that matches your CUDA runtime before
or after the editable install.

## Prepare graph data

Edit a config file first:

```bash
cp configs/chd_boltz2.yaml configs/my_chd.yaml
```

Then set:

- `input_csv`
- `embedding_dir`
- `output_pkl`
- CSV column names
- embedding file template

Run:

```bash
python scripts/prepare_graphs.py --config configs/my_chd.yaml
```

For NTD:

```bash
cp configs/ntd_af3_monomer.yaml configs/my_ntd.yaml
python scripts/prepare_graphs.py --config configs/my_ntd.yaml
```

## Train 5-fold models

```bash
python scripts/train_5fold.py \
  --graph-pkl data/graphs/chd_boltz2_topk8.pkl \
  --output-dir outputs/chd_boltz2 \
  --seed 42 \
  --device cuda:0
```

The output directory will contain:

- `fold1_best.pth` to `fold5_best.pth`
- `fold_results.csv`
- `training.log`

## Predict candidates

```bash
python scripts/predict_ensemble.py \
  --model-dir outputs/chd_boltz2 \
  --candidate-csv data/candidates.csv \
  --embedding-dir data/boltz2_embeddings \
  --output-csv outputs/chd_boltz2_all_predictions.csv \
  --embedding-format boltz2 \
  --filename-template 'embeddings_{name}.npz' \
  --name-column mutation_id \
  --device cuda:0
```

For AF3 monomer raw embeddings:

```bash
python scripts/predict_ensemble.py \
  --model-dir outputs/ntd_af3_monomer \
  --candidate-csv data/ntd_candidates.csv \
  --embedding-dir data/af3_monomer_embeddings \
  --output-csv outputs/ntd_af3_monomer_all_predictions.csv \
  --embedding-format af3_monomer \
  --filename-template '{name}_monomer_seed-1_embeddings.npz' \
  --name-column mutation_id \
  --lowercase-name \
  --device cuda:0
```

## Notes before making the GitHub repo public

- Do not commit private training tables if they should not be public.
- Do not commit embedding files or checkpoints.
- Add a license only after deciding the intended reuse policy.
- Replace placeholder paths in `configs/` with paths that make sense for the
  public repository or keep them as examples.

For the initial GitHub push, see [docs/github_setup.md](docs/github_setup.md).
