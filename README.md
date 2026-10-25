# English → Darija Neural Machine Translation with LSTM variants

Lab project, *Advanced NLP – Recurrent Neural Networks* (2026/2027).
Dataset: [`imomayiz/darija-english`](https://huggingface.co/datasets/imomayiz/darija-english).

We build an encoder-decoder translator whose recurrent cells are **implemented from scratch with `nn.Linear`** (no `nn.LSTM`), and compare:

| Model | Cell | Idea |
|---|---|---|
| Baseline | Vanilla LSTM (Eq. 1–6) | reference (verified equal to `nn.LSTMCell`, see `tests/`) |
| Model A | **CIFG** (Eq. 13–18) | input gate coupled to forget gate: `i = 1 − f` |
| Model B | **LSTMP** (Eq. 19–25) | hidden state projected from `H` to `P < H` |
| Bonus | WMC (Eq. 7–12), Bahdanau attention | optional extras |

## Project layout
```
cells.py             Vanilla / CIFG / WMC / LSTMP cells + StackedRNN
data.py              loading, 80/10/10 split (fixed seed), SentencePiece BPE, DataLoaders
train.py             Seq2Seq model, training loop, greedy decoding, BLEU/chrF, JSON logging
eda.ipynb            exploratory data analysis
aggregate.py         runs/*.json -> results/results.{csv,md} + figures
run_experiments.sh   all experiments (stage1 / stage2 / stage3 / bonus)
tests/test_cells.py  unit tests (vanilla == nn.LSTMCell, CIFG identity, shapes)
colab_run.ipynb      one-click runner for Google Colab / Kaggle
docs/                report template + slide outline
```

## Setup
```bash
pip install -r requirements.txt
python data.py            # prints columns, auto-detected source/target columns
python tests/test_cells.py
```
If the target column is wrong (e.g. you want Arabic script `darija_ar`), pass `--tgt_col darija_ar` to `train.py`.

## Reproducing the experiments
```bash
# 0) quick check (a few minutes): small subsample
EPOCHS=2 MAX_TRAIN=5000 bash run_experiments.sh stage1

# 1) architectures under IDENTICAL settings + parameter counts
bash run_experiments.sh stage1
# 2) hyperparameter sweep on the baseline: lr, optimizer, init, batch size
bash run_experiments.sh stage2
# 3) re-run the 3 architectures with the best hyperparameters found in stage 2
BEST_LR=1e-3 BEST_OPT=adam BEST_INIT=xavier BEST_BS=64 bash run_experiments.sh stage3
# 4) optional extras: WMC and attention
bash run_experiments.sh bonus
```
Each run writes `runs/<name>.json` (args, #params, loss curves, test BLEU/chrF, sample translations).
`aggregate.py` (called automatically) builds the tables and figures in `results/`.
Use `MAX_TRAIN=20000` if GPU time is limited; the subsample is seeded and identical for all runs.

## Evaluation protocol
* One fixed split (80/10/10, seed 42, deduplicated pairs), shared by all models.
* SentencePiece fitted on the training split only.
* Model selection on validation loss (early stopping, best checkpoint), final numbers on the test split.
* Greedy decoding; metrics: sacreBLEU and chrF (chrF is more reliable for noisy Darija spelling).
* Trainable parameters reported for every model.

## Results
_Fill in after running (copy from `results/results.md`)._

## Team
_Names of the 5–6 members and their contributions._
