"""Data pipeline: HF dataset -> fixed 80/10/10 split -> SentencePiece -> DataLoaders.

Column names are auto-detected; override with --src_col / --tgt_col (train.py) if needed.
Set DARIJA_SYNTHETIC=1 to use a tiny toy dataset (offline smoke test only).
"""
import os, random
import torch
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

PAD, UNK, BOS, EOS = 0, 1, 2, 3
HF_ID = "imomayiz/darija-english"
SRC_CANDIDATES = ["eng", "english", "en", "source", "src"]
TGT_CANDIDATES = ["darija", "darija_latin", "ary", "dialect", "target", "tgt", "darija_ar"]


def _synthetic_df():
    import pandas as pd
    en = "i you we they like want see eat drink house water bread book big small good new today now".split()
    ar = "ana nta ahna homa kanbghi bghit nchouf nakol nchrob dar lma khobz ktab kbir sghir mzyan jdid lyoum daba".split()
    rng = random.Random(0)
    rows = []
    for _ in range(3000):
        k = rng.randint(3, 8)
        idx = [rng.randrange(len(en)) for _ in range(k)]
        rows.append((" ".join(en[i] for i in idx), " ".join(ar[i] for i in idx)))
    return pd.DataFrame(rows, columns=["eng", "darija"])


def load_raw_df(config=None):
    """Return one DataFrame holding all rows of all splits of the dataset."""
    if os.environ.get("DARIJA_SYNTHETIC") == "1":
        return _synthetic_df()
    import pandas as pd
    from datasets import load_dataset, get_dataset_config_names
    try:
        ds = load_dataset(HF_ID, config) if config else load_dataset(HF_ID)
    except ValueError:                                  # several configs -> pick one
        names = get_dataset_config_names(HF_ID)
        config = "sentences" if "sentences" in names else names[0]
        print(f"[data] dataset has configs {names}; using '{config}' (override with --config)")
        ds = load_dataset(HF_ID, config)
    return pd.concat([ds[k].to_pandas() for k in ds.keys()], ignore_index=True)


def resolve_columns(df, src_col=None, tgt_col=None):
    cols = {c.lower(): c for c in df.columns}
    pick = lambda given, cands: given if given else next((cols[c] for c in cands if c in cols), None)
    s, t = pick(src_col, SRC_CANDIDATES), pick(tgt_col, TGT_CANDIDATES)
    if s not in df.columns or t not in df.columns:
        raise ValueError(f"Could not detect columns in {list(df.columns)}. Use --src_col/--tgt_col.")
    return s, t


def load_pairs(seed=42, config=None, src_col=None, tgt_col=None):
    df = load_raw_df(config)
    s, t = resolve_columns(df, src_col, tgt_col)
    df = df[[s, t]].dropna()
    pairs = {(str(a).strip(), str(b).strip()) for a, b in zip(df[s], df[t]) if str(a).strip() and str(b).strip()}
    pairs = sorted(pairs)
    random.Random(seed).shuffle(pairs)                  # same split for EVERY model
    n = len(pairs)
    a, b = int(0.8 * n), int(0.9 * n)
    return pairs[:a], pairs[a:b], pairs[b:]


def train_spm(texts, prefix, vocab_size):
    if not os.path.exists(prefix + ".model"):
        with open(prefix + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        spm.SentencePieceTrainer.train(
            input=prefix + ".txt", model_prefix=prefix, vocab_size=vocab_size,
            model_type="bpe", character_coverage=1.0,
            pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, hard_vocab_limit=False)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


class MTDataset(Dataset):
    def __init__(self, pairs, sp_src, sp_tgt, max_len):
        self.items = []
        for s, t in pairs:
            si = sp_src.encode(s)[:max_len - 1] + [EOS]
            ti = [BOS] + sp_tgt.encode(t)[:max_len - 2] + [EOS]
            self.items.append((torch.tensor(si), torch.tensor(ti), t))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    src = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD)
    tgt = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD)
    return src, tgt, [b[2] for b in batch]


def prepare(batch_size=64, vocab_size=8000, max_len=64, workdir="spm", seed=42,
            config=None, src_col=None, tgt_col=None, max_train=None):
    os.makedirs(workdir, exist_ok=True)
    train, val, test = load_pairs(seed, config, src_col, tgt_col)
    if max_train and max_train < len(train):            # optional, seeded subsample (same for all runs)
        train = random.Random(seed).sample(train, max_train)
    print(f"[data] train {len(train):,} | val {len(val):,} | test {len(test):,}")
    sp_src = train_spm([s for s, _ in train], f"{workdir}/src", vocab_size)   # TRAIN ONLY
    sp_tgt = train_spm([t for _, t in train], f"{workdir}/tgt", vocab_size)
    mk = lambda p, sh: DataLoader(MTDataset(p, sp_src, sp_tgt, max_len), batch_size=batch_size,
                                  shuffle=sh, collate_fn=collate)
    return mk(train, True), mk(val, False), mk(test, False), sp_src, sp_tgt


if __name__ == "__main__":
    df = load_raw_df()
    print(df.shape, "| columns:", list(df.columns))
    print("Detected:", resolve_columns(df))
    print(df.head())
