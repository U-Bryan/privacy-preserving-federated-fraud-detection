# Datasets

This repository does **not** redistribute raw credit-card data. Both source
files are publicly available from Kaggle under each platform's terms. You
must download them manually and place them under `./workspace/datasets/`
before any script will run.

## Required files

| File                  | Source                                                                                                   | Approx. size |
| --------------------- | -------------------------------------------------------------------------------------------------------- | ------------ |
| `creditcard.csv`      | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)                       | ≈ 150 MB     |
| `creditcard_2023.csv` | [nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) | ≈ 250 MB     |

## Dataset descriptions

### ULB 2013 — `creditcard.csv`

- 284 807 transactions over two days, September 2013, European cardholders
- 492 fraud cases (0.172 % positive rate)
- Columns: `Time`, `V1`…`V28` (PCA-transformed features), `Amount`, `Class`
- The `Time` column is dropped during preprocessing; `Amount` is log-transformed

### Kaggle 2023 — `creditcard_2023.csv`

- 568 630 transactions, 2023
- Class-balanced (≈ 50 % positive rate)
- Columns: `id`, `V1`…`V28`, `Amount`, `Class`
- The `id` column is dropped during preprocessing; `Amount` is log-transformed

## Expected layout

```
workspace/
└── datasets/
    ├── creditcard.csv
    └── creditcard_2023.csv
```

## Checksums

After downloading, verify file integrity:

```bash
cd workspace/datasets
sha256sum creditcard.csv creditcard_2023.csv
```

Record your checksums here the first time you set up the repo, then commit
this section — future reviewers can confirm they are working with the same
source data:

```text
creditcard.csv       <paste sha256 here>
creditcard_2023.csv  <paste sha256 here>
```

## Licensing

Each dataset is governed by the terms on its Kaggle page. The ULB 2013
dataset is released under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/);
the Kaggle 2023 dataset has its own CC-derived terms — consult its page.
This repository's MIT licence covers only the code, not the data.

## Preprocessing summary

All preprocessing logic lives in [`src/fraud_fl/data.py`](../src/fraud_fl/data.py).
The two source CSVs are merged after:

1. Dropping `Time` (ULB) and `id` (Kaggle) columns.
2. Applying `log(1 + Amount)` to the `Amount` column of each, clipped at 0.
3. Re-selecting the common feature set: `V1, V2, …, V28, Amount, Class`.
4. Concatenating and shuffling — the two are treated as one corpus of
   853 437 transactions.

Stratified-temporal partitioning across clients preserves the (small) global
fraud ratio on each client. Each client is further split 70 / 10 / 20 into
train / validation / test.
