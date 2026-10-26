# NID entry point

`nid.py` consumes task-specific encoder embeddings and matching labels and reports NID; training, threshold selection, and domain estimation stay with the caller. It only needs NumPy. Training-split embeddings can be exported by the IMU training entry point using `--export-pretrain-embeddings`.

```bash
python3 nid.py --embeddings outputs/m0_embeddings.npy \
  --labels outputs/m0_embeddings_labels.npy --variant swap \
  --traversals 10 --seed 42 --output outputs/nid.json
```

Paths, seed, and traversal count in this example are configuration examples.

## Explicit historical variants

The variant is mandatory because the retained source contains different calculations:

| Variant | Partition and aggregation | Source family |
|---|---|---|
| `swap` | Shuffle all samples into `2k` clips; compare the first and second `k` clips; exchange one clip pair between rounds; average per-class distances equally and then average `k` rounds | January 2026 `nid_swap*.py`; matches the current manuscript's formula when all classes occur in both halves |
| `legacy_random` | Independently random-split each class for every round; aggregate rounds; weight classes by their original sample counts | November 2023 `prism_open/nid.py` |
| `legacy_swap` | Random-split each class, truncate both halves to equal length, exchange 10% of corresponding positions per round; aggregate rounds and prevalence-weight classes | November 2023 `prism_open/nid.py` |

The variants compute different quantities, so keep them separate when reporting numbers.

`swap` supports `--metric l2`, `linf`, and `cosine` with scalar standard-deviation normalization. The retained L2 normalization variants are `zscore`, `none`, `minmax`, and `robust` (unscaled median absolute deviation). Other combinations are rejected because they were not present in the retained scripts. Legacy variants support L2 with standard-deviation normalization and `--aggregation mean` or `var`.

Normalization uses one scalar over all embedding elements of each class, as in the retained source. `zscore` is a historical option name: it scales the class mean difference; it does not independently standardize every feature.

## Input and robustness contract

- Embeddings must be finite numeric `[N,D]` arrays, without object/pickled data.
- Labels may be `[N]`, `[N,L]`, or `[N,T,L]`. Multi-column labels require an explicit `--label-index`. For temporal labels, the selected task label must be constant within every window; the first timestep is then selected.
- Keep task labels separate from domain metadata. Neither labels nor the input arrays are mutated.
- For `swap`, use at least `2k` samples. The historical code skips a class absent from one half and averages the remaining classes. JSON output reports class coverage and warnings. A round with no comparable class retains the historical numerical value zero and reports a warning.
- Legacy empty class splits and zero normalization scale now raise clear errors. Choose another seed explicitly or provide more samples if necessary.
- `swap` computes `k` states and performs `k-1` pair exchanges, exactly as in the retained code. 

Use training embeddings from the selected pretrained model; the NID script works from the arrays it is given.
