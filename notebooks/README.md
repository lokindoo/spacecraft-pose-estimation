# Notebooks

Three, deliberately. The original study ran across twenty-five Colab notebooks
that differed mostly by a seed literal; the logic now lives in
`src/speedplus_pose/`, and these exist to make it *visible* — not to hold it.

| Notebook | What it is for | Needs |
|---|---|---|
| `01_train_end_to_end.ipynb` | One arm from label file to selected checkpoint, with each step exposed: canonicalised labels, the augmentation stack rendered, the freeze policy, the split loss, the selection rule. | prepared subsets, a GPU for a real run |
| `02_measure_the_gap.ipynb` | The moment the study is about — a synthetic-trained model meeting real imagery. Range check first, then errors against the chance level, then whether the model's confidence survives. | a trained checkpoint |
| `03_reproduce_figures.ipynb` | The paper's two figures — the augmentation/domain triptych and the conformal reliability panel — plus the rotation-error chart that earlier drafts carried. | Figure 1 needs the subsets; Figure 2 needs prediction records; the extra needs nothing |

Each has a command-line equivalent under `scripts/`, which is what you want for
an actual run. Use the notebooks to read the pipeline or to show someone what it
does.

## Running on Colab

Every notebook opens with the same bootstrap cell. Edit only that:

```python
from google.colab import drive; drive.mount("/content/drive")
!pip install -q -e "/content/drive/MyDrive/<path to repo>"

os.environ["SPEEDPLUS_DATA_ROOT"] = "/content/drive/MyDrive/<...>/data"
os.environ["SPEEDPLUS_ARTIFACTS"] = "/content/drive/MyDrive/<...>/artifacts"
LOCAL_IMAGES = "/content/images"
```

Two things matter there:

- **Artifacts on the mounted Drive.** Checkpoints are written on even epochs, and
  a session that disconnects should cost you two epochs, not the run.
- **`LOCAL_IMAGES` under `/content`.** Reading tens of thousands of small JPEGs
  across the Drive mount will starve the GPU. The image archive copies over once
  and unpacks to local disk.

## A note on outputs

These are committed without executed outputs. That keeps diffs readable and
avoids shipping stale numbers that disagree with `results/` — the authoritative
numbers live there, and `scripts/reproduce_table.py` is what checks them.
