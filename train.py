"""Seq2seq (encoder-decoder) English->Darija with custom LSTM cells (+ optional attention).

Examples:
  python train.py --cell lstm  --name main_lstm
  python train.py --cell cifg  --name main_cifg
  python train.py --cell lstmp --proj 256 --name main_lstmp
  python train.py --cell lstm  --attn --name bonus_lstm_attn
"""
import argparse, json, os, random, time
import numpy as np
import torch, torch.nn as nn
import sacrebleu
from cells import StackedRNN
from data import prepare, PAD, BOS, EOS


class Seq2Seq(nn.Module):
    def __init__(self, src_v, tgt_v, cell, emb, hid, layers, dropout, proj=None, attn=False):
        super().__init__()
        self.attn = attn
        self.src_emb = nn.Embedding(src_v, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_v, emb, padding_idx=PAD)
        self.enc = StackedRNN(cell, emb, hid, layers, dropout, proj)
        E = self.enc.out_size
        self.dec = StackedRNN(cell, emb + (E if attn else 0), hid, layers, dropout, proj)
        if attn:                                         # Bahdanau (additive) attention
            self.Wq = nn.Linear(self.dec.out_size, hid, bias=False)
            self.Wk = nn.Linear(E, hid)
            self.v = nn.Linear(hid, 1, bias=False)
        self.out = nn.Linear(self.dec.out_size + (E if attn else 0), tgt_v)
        self.drop = nn.Dropout(dropout)

    def encode(self, src):
        B, T = src.shape
        states = self.enc.init_state(B, src.device)
        e = self.drop(self.src_emb(src))
        outs = []
        for t in range(T):
            o, new = self.enc(e[:, t], states)
            m = (src[:, t] != PAD).unsqueeze(1)          # keep old state on padding
            states = [(torch.where(m, nh, h), torch.where(m, nc, c))
                      for (h, c), (nh, nc) in zip(states, new)]
            outs.append(o)
        mem = None
        if self.attn:
            outs = torch.stack(outs, 1)
            mem = (outs, self.Wk(outs), src != PAD)
        return states, mem

    def dec_step(self, tok_emb, states, mem):
        if self.attn:
            outs, kp, mask = mem
            s = self.v(torch.tanh(self.Wq(states[-1][0]).unsqueeze(1) + kp)).squeeze(-1)
            a = torch.softmax(s.masked_fill(~mask, -1e9), dim=-1)
            ctx = torch.bmm(a.unsqueeze(1), outs).squeeze(1)
            o, states = self.dec(torch.cat([tok_emb, ctx], -1), states)
            return self.out(torch.cat([o, ctx], -1)), states
        o, states = self.dec(tok_emb, states)
        return self.out(o), states

    def forward(self, src, tgt_in):                      # teacher forcing
        states, mem = self.encode(src)
        e = self.drop(self.tgt_emb(tgt_in))
        outs = []
        for t in range(tgt_in.size(1)):
            lg, states = self.dec_step(e[:, t], states, mem)
            outs.append(lg)
        return torch.stack(outs, 1)                      # B, T, V

    @torch.no_grad()
    def greedy(self, src, max_len=64):
        B = src.size(0)
        states, mem = self.encode(src)
        tok = torch.full((B,), BOS, device=src.device)
        res, done = [], torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            lg, states = self.dec_step(self.tgt_emb(tok), states, mem)
            tok = lg.argmax(-1)
            res.append(tok)
            done |= tok == EOS
            if done.all():
                break
        return torch.stack(res, 1)


def init_weights(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if hasattr(m, "reset_forget_bias"):
            m.reset_forget_bias()                        # forget-gate bias = 1


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_opt(name, params, lr):
    return {"adam": lambda: torch.optim.Adam(params, lr=lr),
            "adamw": lambda: torch.optim.AdamW(params, lr=lr),
            "rmsprop": lambda: torch.optim.RMSprop(params, lr=lr),
            "sgd": lambda: torch.optim.SGD(params, lr=lr, momentum=0.9)}[name]()


def run_epoch(model, loader, crit, dev, opt=None, clip=1.0):
    model.train(opt is not None)
    tot, n = 0.0, 0
    for src, tgt, _ in loader:
        src, tgt = src.to(dev), tgt.to(dev)
        with torch.set_grad_enabled(opt is not None):
            logits = model(src, tgt[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if opt:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        tot += loss.item(); n += 1
    return tot / n


def evaluate_bleu(model, loader, sp_tgt, dev, max_len=64, keep=10):
    model.eval()
    hyps, refs, srcs = [], [], []
    for src, _, raw in loader:
        out = model.greedy(src.to(dev), max_len).tolist()
        for ids, r in zip(out, raw):
            ids = ids[:ids.index(EOS)] if EOS in ids else ids
            hyps.append(sp_tgt.decode(ids)); refs.append(r)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return bleu, chrf, [{"hyp": h, "ref": r} for h, r in zip(hyps[:keep], refs[:keep])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--cell", default="lstm", choices=["lstm", "cifg", "wmc", "lstmp"])
    p.add_argument("--attn", action="store_true", help="add Bahdanau attention (bonus)")
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--hid", type=int, default=512)
    p.add_argument("--proj", type=int, default=None, help="LSTMP projection size P (default hid//2)")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", default="adam", choices=["adam", "adamw", "rmsprop", "sgd"])
    p.add_argument("--init", default="xavier", choices=["default", "xavier", "orthogonal", "kaiming"])
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--vocab", type=int, default=8000)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--max_train", type=int, default=None, help="subsample train set (seeded)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None)
    p.add_argument("--src_col", default=None)
    p.add_argument("--tgt_col", default=None)
    p.add_argument("--outdir", default="runs")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, sp_s, sp_t = prepare(a.bs, a.vocab, a.max_len, "spm", 42,
                                     a.config, a.src_col, a.tgt_col, a.max_train)

    model = Seq2Seq(len(sp_s), len(sp_t), a.cell, a.emb, a.hid, a.layers, a.dropout, a.proj, a.attn).to(dev)
    init_weights(model, a.init)
    print(f"[{a.name}] device={dev} trainable params: {n_params(model):,}")

    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    opt = make_opt(a.opt, model.parameters(), a.lr)
    os.makedirs(a.outdir, exist_ok=True)
    ckpt = f"{a.outdir}/{a.name}.pt"
    best, bad, hist, t_start = 1e9, 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, tr, crit, dev, opt)
        val = run_epoch(model, va, crit, dev)
        hist.append({"epoch": ep, "train": trl, "val": val, "sec": time.time() - t0})
        print(f"ep {ep:02d} train {trl:.3f} val {val:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if val < best:
            best, bad = val, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    bleu, chrf, samples = evaluate_bleu(model, te, sp_t, dev, a.max_len)
    print(f"TEST  BLEU {bleu:.2f}  chrF {chrf:.2f}")
    for s in samples[:5]:
        print("HYP:", s["hyp"], "\nREF:", s["ref"], "\n")
    json.dump({"args": vars(a), "params": n_params(model), "best_val_loss": best,
               "test_bleu": bleu, "test_chrf": chrf, "train_sec": time.time() - t_start,
               "history": hist, "samples": samples},
              open(f"{a.outdir}/{a.name}.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
