"""Collect runs/*.json -> results/results.csv, results/results.md and figures.

Naming convention for runs:
  main_<cell>            -> the 3 (or 4) architectures compared under identical settings
  hp_<factor>_<value>    -> hyperparameter sweep on the baseline (factor in lr, opt, init, bs)
  final_<cell>           -> each architecture re-run with the best hyperparameters
  bonus_*                -> extras (attention, WMC ...)
"""
import glob, json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
rows, hists = [], {}
for f in sorted(glob.glob("runs/*.json")):
    r = json.load(open(f)); a = r["args"]
    rows.append({"run": a["name"], "cell": a["cell"] + ("+attn" if a.get("attn") else ""),
                 "params": r["params"], "hid": a["hid"], "proj": a.get("proj"), "layers": a["layers"],
                 "lr": a["lr"], "opt": a["opt"], "init": a["init"], "bs": a["bs"], "dropout": a["dropout"],
                 "epochs_run": len(r["history"]), "best_val_loss": round(r["best_val_loss"], 4),
                 "BLEU": round(r["test_bleu"], 2), "chrF": round(r["test_chrf"], 2),
                 "train_min": round(r["train_sec"] / 60, 1)})
    hists[a["name"]] = r["history"]
if not rows:
    raise SystemExit("No runs/*.json found. Run train.py first.")

df = pd.DataFrame(rows)
df.to_csv("results/results.csv", index=False)
with open("results/results.md", "w") as f:
    for title, mask in (("Architecture comparison", df.run.str.startswith("main_")),
                        ("Final models (best hyperparameters)", df.run.str.startswith("final_")),
                        ("Hyperparameter sweep (baseline LSTM)", df.run.str.startswith("hp_")),
                        ("Bonus", df.run.str.startswith("bonus_"))):
        if mask.any():
            f.write(f"### {title}\n\n{df[mask].to_markdown(index=False)}\n\n")
print(open("results/results.md").read())

def curves(prefix, fname, title):
    names = [n for n in hists if n.startswith(prefix)]
    if not names: return
    plt.figure(figsize=(6, 4))
    for n in names:
        h = hists[n]; ep = [x["epoch"] for x in h]
        l, = plt.plot(ep, [x["val"] for x in h], label=n)
        plt.plot(ep, [x["train"] for x in h], "--", color=l.get_color(), alpha=.5)
    plt.xlabel("epoch"); plt.ylabel("loss (solid=val, dashed=train)"); plt.title(title)
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

curves("main_", "results/curves_main.png", "Architectures")
curves("final_", "results/curves_final.png", "Final models")

# bar charts per hyperparameter factor
for factor in ("lr", "opt", "init", "bs"):
    sub = df[df.run.str.startswith(f"hp_{factor}_")]
    if sub.empty: continue
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(sub[factor].astype(str), sub["chrF"], color="#4c72b0")
    for x, (b, c) in enumerate(zip(sub["BLEU"], sub["chrF"])):
        ax.text(x, c, f"BLEU {b}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("test chrF"); ax.set_title(f"Effect of {factor}"); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(f"results/hp_{factor}.png", dpi=150); plt.close()

main = df[df.run.str.startswith("main_")]
if not main.empty:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].bar(main.cell, main.chrF, color="#dd8452"); ax[0].set_title("Test chrF")
    ax[1].bar(main.cell, main.params / 1e6, color="#55a868"); ax[1].set_title("Trainable params (M)")
    for a_ in ax: a_.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig("results/architectures.png", dpi=150); plt.close()
print("Saved results/ (csv, md, png)")
