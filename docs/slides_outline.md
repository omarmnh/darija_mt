# 10-minute presentation (English) – 10 slides, ~1 min each
1. **Title** – team, task, dataset.
2. **Goal** – English→Darija MT; challenges (dialect, spelling, low resource); metrics BLEU/chrF.
3. **EDA (1)** – dataset size, script of Darija, example pairs.
4. **EDA (2)** – length histogram, vocab/Zipf, max_len choice, OOV.
5. **Modeling** – encoder–decoder diagram; tokenization; training setup.
6. **Cells** – baseline LSTM vs CIFG (i = 1−f) vs LSTMP (projection); parameter table.
7. **Hyperparameters** – sweep results (lr, optimizer, init, batch size) + chosen values.
8. **Results** – final table (BLEU, chrF, params, time) + learning curves.
9. **Error analysis** – 2–3 translation examples (good/bad) + what fails.
10. **Improvements & conclusion** – attention, beam search, Transformer, more data; who did what.
Tips: one message per slide, figures from `results/`, rehearse to 9 minutes (1 min for questions).
