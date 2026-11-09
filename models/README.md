---
license: mit
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - mlp
  - pytorch
  - architecture
  - built-environment
  - facades
  - new-york-city
datasets:
  - nyc-open-data/dob-now-safety-facades-compliance-filings
metrics:
  - f2
  - recall
  - precision
  - average_precision
---

<!-- TODO: replace <model-repo> with the final repository name, e.g. day4-team1-unsafe-facade-mlp, and update all numbers after your final run -->

# NYC facade MLP: which facades to inspect first

A small neural network (MLP) that estimates, **before an inspection**, the probability that a New York City facade inspection (FISP, Local Law 11)
comes out **UNSAFE**, from the previous inspection result, the borough, the cycle and the facade's materials and wall types.
It is the neural-network version of our [Day 2 random forest](https://huggingface.co/iaac-maai/day2-team1-unsafe-facade), trained and tested
on the same data, and it **performs the same, not better**.

## Model at a glance

![Test comparison](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/07_test_comparison.png)

✏️ TODO: one or two sentences reading the figure.

![Inspection ranking](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/08_inspection_ranking.png)

✏️ TODO

## Intended use

- **Users:** the NYC Department of Buildings, facade consultants.
- **Use:** at the start of a new FISP cycle, **rank buildings by risk** to decide which facades to inspect first.

**Out of scope:** replacing an inspection or judging whether a single facade is safe; other cities or buildings of 6 stories or less;
any decision about a building's owner (fines, insurance, sale price).

## Model details

| | |
|---|---|
| **Architecture** | MLP (`nn.Sequential`): 31 inputs → Linear 128 + ReLU + Dropout 0.2 → Linear 64 + ReLU + Dropout 0.2 → Linear 1 (logit) |
| **Parameters** | 12,417 |
| **Base model** | none (trained from scratch) |
| **Input** | one inspection filing: 19 columns (3 categorical, 16 yes/no), turned into 31 numbers by `scaler.joblib` |
| **Output** | one logit; `sigmoid(logit)` = P(UNSAFE). Decision threshold **0.30** (UNSAFE if P ≥ 0.30) |
| **Files** | `model.pt` (weights, `state_dict`), `scaler.joblib` (scikit-learn preprocessing: imputation, one-hot, `StandardScaler`), `config.json` (architecture, feature order, threshold) |
| **Libraries** | torch 2.15.0, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4 <!-- TODO: check with pip freeze --> |

## Classes and input features

**Classes:** `0` = not UNSAFE (SAFE or SWARMP, "safe with repair and maintenance program"), `1` = UNSAFE (a hazard that needs immediate protection and repair).

| Feature | Type | Meaning |
|---|---|---|
| `PRIOR_STATUS` | category | result of the building's previous cycle: SAFE, SWARMP, UNSAFE, NO REPORT FILED, or NO PRIOR CYCLE |
| `BOROUGH` | category | MANHATTAN, BRONX, BROOKLYN, QUEENS, STATEN ISLAND |
| `CYCLE` | category | FISP cycle, "6" to "10" (text) |
| `MATERIAL_*` (8) | 0/1 | facade material present: masonry, terra cotta, natural stone, concrete, stucco/EIFS, glass, metal, other |
| `WALL_TYPE_*` (8) | 0/1 | wall type present: brick masonry, masonry cavity wall, masonry load-bearing wall, terra cotta, steel frame, concrete, concrete frame, other |

## Training data

| | |
|---|---|
| **Dataset** | DOB NOW: Safety – Facades Compliance Filings, [NYC Open Data, xubg-57si](https://data.cityofnewyork.us/d/xubg-57si), cleaned in our [Day 2 notebook](https://github.com/iaac-maai/group1_day2_assignment) |
| **Licence** | NYC Open Data, free public use under the [NYC Terms of Use](https://www.nyc.gov/home/terms-of-use.page) |
| **Rows** | 54,049 initial filings of 15,767 buildings, 23.2% UNSAFE |
| **Splits** | grouped by building and stratified: train 34,593 · validation 8,646 · test 10,810 (the Day 2 test set) |
| **Coverage** | NYC buildings taller than 6 stories, FISP cycles 6–10 (about 2004–2026) |

## Training procedure

- `BCEWithLogitsLoss` with `pos_weight` = 3.31 (#not UNSAFE / #UNSAFE), Adam (learning rate 0.001), batches of 256, up to 60 epochs.
- Early stopping on validation loss (patience 10); weights of epoch **25** kept.
- Model selection: four experiments, one change each (base 32-16 · wider 128-64 · no dropout · smaller 8), chosen by validation average precision.
  The differences were small (validation AP 0.581–0.601).
- Decision threshold chosen on validation to maximise F2.
- Hardware: laptop CPU, about 1 minute of training. No data augmentation.

![Training curves](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/01_training_curves.png)

✏️ TODO

## Evaluation

Test set (10,810 filings of 3,181 buildings, used once):

| Model | F2 | Recall | Precision | Balanced accuracy | Average precision |
|---|---|---|---|---|---|
| Always UNSAFE ("inspect everything") | 0.602 | 1.000 | 0.232 | 0.500 | 0.232 |
| Day 2 random forest (threshold 0.30) | 0.631 | 0.915 | 0.281 | 0.604 | 0.605 |
| **This MLP (threshold 0.30)** | 0.629 | 0.910 | 0.281 | 0.604 | 0.601 |

Inspecting the riskiest 10% / 20% / 30% of filings first finds 34% / 48% / 60% of the UNSAFE facades (random order: 10% / 20% / 30%).

![Confusion matrices](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/06_test_confusion_matrices.png)

✏️ TODO

**Failures (validation set):** recall is lowest for previously **SAFE** facades (0.79) and for buildings with no report filed (0.67), and precision is
low everywhere except for previously UNSAFE facades. ✏️ TODO: your reading.

## Limitations and biases

✏️ TODO (team). Starting points: the model mostly repeats the previous inspection result; no building age, height, size or exposure; the label is a
human judgement and the rules changed between cycles; borough and cycle stand in for things we do not have; NYC only.

## Ethics and privacy

The data is public and describes buildings, not people. The model card and files contain no addresses or owner names. A risk score should not
be used to stigmatise owners or neighbourhoods; it only orders inspections, which stay mandatory for every building.

## How to use

```python
import json
import joblib
import pandas as pd
import torch
from torch import nn
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/<model-repo>"
cfg = json.load(open(hf_hub_download(REPO, "config.json")))
preprocess = joblib.load(hf_hub_download(REPO, "scaler.joblib"))   # pickle: only load files you trust


def make_mlp(n_inputs, hidden, dropout):
    layers, size = [], n_inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
        size = width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


model = make_mlp(cfg["n_inputs"], cfg["hidden"], cfg["dropout"])
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()

# One building: previously UNSAFE terra cotta facade in Manhattan, cycle 10
row = {f: 0.0 for f in cfg["binary_features"]}
row.update({"PRIOR_STATUS": "UNSAFE", "BOROUGH": "MANHATTAN", "CYCLE": "10",
            "MATERIAL_TERRA_COTTA": 1.0, "MATERIAL_MASONRY": 1.0,
            "WALL_TYPE_TERRA_COTTA": 1.0, "WALL_TYPE_STEEL_FRAME": 1.0})
X = pd.DataFrame([row])[cfg["features"]]

with torch.no_grad():
    p = torch.sigmoid(model(torch.tensor(preprocess.transform(X), dtype=torch.float32))).item()
print(f"P(UNSAFE) = {p:.3f} ->", "inspect first" if p >= cfg["threshold"] else "later")
```

## Reproducibility

- Code: https://github.com/iaac-maai/<github-repo> <!-- TODO -->
- Notebook: `notebook/neural_network.ipynb` · Requirements: `requirements.txt`
- Seed 42 everywhere; small differences between computers are possible.

## Security note

`model.pt` and `scaler.joblib` are pickle-based files: loading them can run code. Only load them from this repository or another source you trust.

## Authors and licences

Amelia, Jerome, Kazuko, Sahil – AI Methods I, Day 4, IAAC Master in AI for Architecture (2026/27).
Model and code: MIT. Data: NYC Open Data, [NYC Terms of Use](https://www.nyc.gov/home/terms-of-use.page).
