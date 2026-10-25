# English → Darija Translation with LSTM Variants
**Team:** … | **Group lab:** … | **Date:** …   (target: 4–5 pages)

## 1. Goal of the task
Machine translation English → Darija (`imomayiz/darija-english`). Why it is hard: low-resource dialect, non-standard spelling, code-switching. Metrics: BLEU, chrF. Repo: <GitHub link>.

## 2. Exploratory data analysis
(Numbers/plots from `eda.ipynb`)
- Size, cleaning (duplicates, empties), split 80/10/10.
- Script of the Darija side (Arabic vs Latin/Arabizi) and share of Arabizi digits.
- Length statistics (words, subword tokens) → choice of `max_len = …` (covers …% of pairs).
- Vocabulary size, hapax share, Zipf curve, spelling variability (vocab reduction after collapsing repeated letters).
- OOV rate val/test, source leakage across splits, suspicious length ratios.

## 3. Modeling approach
- Encoder–decoder; embedding …, hidden …, layers …, dropout …; SentencePiece BPE (vocab …).
- Teacher forcing, cross-entropy ignoring PAD, gradient clipping 1.0, early stopping.
- Cells (give the equations): vanilla LSTM (baseline), CIFG (`i = 1 − f`), LSTMP (`h = W_p m + b_p`, P = …).
- Implementation check: vanilla cell equals `nn.LSTMCell` (unit test).
- Table: model | cell | trainable params.

## 4. Design choices of hyperparameters
Sweep on the baseline, one factor at a time (same split, same seed): learning rate, optimizer, initialization, batch size.
Table + figure per factor; comment on stability (divergence at lr=1e-2? SGD slow?), selected values and why.

## 5. Final performance and evaluation
- Table: BLEU, chrF, val loss, params, train time for baseline / CIFG / LSTMP (same settings, then best hyperparameters).
- Learning curves (`results/curves_*.png`).
- Discussion: does removing a gate (CIFG) hurt? Does projection (LSTMP) trade accuracy for parameters? Is the difference larger than seed noise (report ≥2 seeds if possible)?
- Qualitative examples: 3 good, 3 bad translations; error types (spelling variants, rare words, long sentences, repetition).

## 6. Potential improvements
Attention (our bonus result: …), beam search, bidirectional encoder, tied embeddings, joint vocabulary, back-translation / more data, Darija spelling normalization, pretrained models, Transformer baseline.

## References
Greff et al. 2017 (LSTM: A Search Space Odyssey, arXiv:1503.04069); Sak et al. 2014 (LSTMP); WMC paper (Neural Networks 2021); Bahdanau et al. 2015.
