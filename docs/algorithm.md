# Algorithm description

## Summary

BindGraph is a sparse pair-embedding graph neural network for ranking protein
binder variants:

```text
single/pair embedding -> top-k sparse graph -> edge-conditioned GNN -> activity score
```

The same model can be used with different embedding sources. CHD and NTD are
included as reference use cases, where the graph construction, architecture,
training loop, and ensemble prediction logic are shared.

## Inputs

Each sample needs one single embedding and one pair embedding.

Expected shapes:

```text
single: (L, 384)
pair:   (L, L, 128)
```

Supported built-in `.npz` formats:

- `boltz2`: single is `s[0]`, pair is `z[0]`
- `af3_monomer`: single is `single_embeddings`, pair is `pair_embeddings`

Custom `.npz` keys can be supported by passing `single_key` and `pair_key` in a
config file.

## Graph construction

Residues or structure tokens are graph nodes. Edges are directed.

For each node `i`, the graph contains:

1. Sequence-neighbor edges to valid nodes in `i-2`, `i-1`, `i+1`, `i+2`.
2. Top-k non-local edges selected by the L2 norm of the pair embedding
   `pair[i, j, :]`, excluding self and already added sequence neighbors.

The default setting is `k = 8`.

The edge feature for edge `i -> j` is the full pair vector:

```text
edge_feature(i, j) = pair[i, j, :]
```

## Model

The model has:

- node projection: 384 to 256
- edge projection: 128 to 256
- two edge-conditioned graph-convolution layers
- per-target attention over incoming messages
- attention pooling over nodes
- MLP regression head

The message-passing layer computes:

```text
h_i' = W_self h_i + sum_j alpha_ij W_msg [h_j || e_ij]
```

where `alpha_ij` is an attention weight normalized across incoming neighbors of
target node `i`.

## Training

Default training setup:

- 5-fold cross-validation
- batch size 1
- AdamW
- learning rate `1e-4`
- weight decay `1e-3`
- MSE loss
- max epochs 200
- early stopping patience 50
- checkpoint selected by validation Spearman rho

## Prediction

Candidate prediction loads the five fold checkpoints, builds a graph for each
candidate embedding file, and averages the five fold predictions:

```text
score = mean(fold1, fold2, fold3, fold4, fold5)
prediction_std = std(fold1, fold2, fold3, fold4, fold5)
```

Candidates are ranked by descending `score`.

## CHD vs NTD

| dataset | node feature | edge feature | graph/model/training |
|---|---|---|---|
| CHD | Boltz2 single | Boltz2 pair | same |
| NTD | AF3 monomer raw single | AF3 monomer raw pair | same |
