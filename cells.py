"""LSTM cell variants (Vanilla, CIFG, WMC, LSTMP) built only on nn.Linear.

Every cell has the same interface:
    cell(x, (h, c)) -> (h_new, c_new)
    cell.init_state(B, device) -> (h0, c0)
    cell.out_size  -> size of h (H, or P for LSTMP)
"""
import torch
import torch.nn as nn

sig, tanh = torch.sigmoid, torch.tanh


class VanillaLSTMCell(nn.Module):
    """Eq. (1)-(6). Gate order in the fused weights: i, f, o, g."""
    n_gates = 4
    forget_idx = 1

    def __init__(self, input_size, hidden_size, proj_size=None):
        super().__init__()
        self.H = hidden_size
        self.out_size = hidden_size
        self.x2g = nn.Linear(input_size, self.n_gates * hidden_size)            # W_x* x + b_*
        self.h2g = nn.Linear(hidden_size, self.n_gates * hidden_size, bias=False)  # W_h* h
        self.reset_forget_bias()

    def reset_forget_bias(self, value=1.0):
        with torch.no_grad():
            self.x2g.bias.zero_()
            k = self.forget_idx
            self.x2g.bias[k * self.H:(k + 1) * self.H].fill_(value)

    def init_state(self, B, device):
        return (torch.zeros(B, self.out_size, device=device),
                torch.zeros(B, self.H, device=device))

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        i, f, o, g = sig(i), sig(f), sig(o), tanh(g)
        c = f * c + i * g
        h = o * tanh(c)
        return h, c


class CIFGCell(VanillaLSTMCell):
    """Coupled input-forget gate, Eq. (13)-(18): i = 1 - f. Gates: f, o, g."""
    n_gates = 3
    forget_idx = 0

    def forward(self, x, state):
        h, c = state
        f, o, g = (self.x2g(x) + self.h2g(h)).chunk(3, dim=-1)
        f, o, g = sig(f), sig(o), tanh(g)
        i = 1.0 - f
        c = f * c + i * g
        h = o * tanh(c)
        return h, c


class WMCCell(VanillaLSTMCell):
    """Working Memory Connections, Eq. (7)-(12).
    i, f see tanh(W_c c_{t-1}); o sees tanh(W_co c_t) (the NEW cell state)."""

    def __init__(self, input_size, hidden_size, proj_size=None):
        super().__init__(input_size, hidden_size)
        self.Wci = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wcf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wco = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        i = sig(i + tanh(self.Wci(c)))
        f = sig(f + tanh(self.Wcf(c)))
        g = tanh(g)
        c = f * c + i * g
        o = sig(o + tanh(self.Wco(c)))
        h = o * tanh(c)
        return h, c


class LSTMPCell(VanillaLSTMCell):
    """LSTM with projection, Eq. (19)-(25). c in R^H, h in R^P (P < H)."""

    def __init__(self, input_size, hidden_size, proj_size=None):
        proj_size = proj_size or hidden_size // 2
        nn.Module.__init__(self)
        self.H, self.P = hidden_size, proj_size
        self.out_size = proj_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size)
        self.h2g = nn.Linear(proj_size, 4 * hidden_size, bias=False)  # recurrent input is h in R^P
        self.proj = nn.Linear(hidden_size, proj_size)                 # W_p m + b_p
        self.reset_forget_bias()

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        i, f, o, g = sig(i), sig(f), sig(o), tanh(g)
        c = f * c + i * g
        m = o * tanh(c)
        h = self.proj(m)
        return h, c


CELLS = {"lstm": VanillaLSTMCell, "cifg": CIFGCell, "wmc": WMCCell, "lstmp": LSTMPCell}


class StackedRNN(nn.Module):
    """Stack of identical cells with dropout between layers."""

    def __init__(self, cell_name, input_size, hidden, layers, dropout, proj=None):
        super().__init__()
        cls = CELLS[cell_name]
        self.cells = nn.ModuleList()
        for l in range(layers):
            in_sz = input_size if l == 0 else self.cells[-1].out_size
            self.cells.append(cls(in_sz, hidden, proj))
        self.drop = nn.Dropout(dropout)
        self.out_size = self.cells[-1].out_size

    def init_state(self, B, device):
        return [c.init_state(B, device) for c in self.cells]

    def forward(self, x, states):
        new = []
        for cell, st in zip(self.cells, states):
            h, c = cell(x, st)
            new.append((h, c))
            x = self.drop(h)
        return x, new
