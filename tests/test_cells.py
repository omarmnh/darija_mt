"""Sanity tests: run with `python -m pytest tests` or `python tests/test_cells.py`."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
from cells import VanillaLSTMCell, CIFGCell, WMCCell, LSTMPCell


def test_vanilla_matches_torch_lstmcell():
    """Our vanilla cell must equal nn.LSTMCell once weights are copied (gate order i,f,g,o -> i,f,o,g)."""
    torch.manual_seed(0)
    I, H, B = 7, 5, 3
    ref, ours = nn.LSTMCell(I, H), VanillaLSTMCell(I, H)
    perm = lambda w: torch.cat([w[0:H], w[H:2*H], w[3*H:4*H], w[2*H:3*H]])   # i,f,g,o -> i,f,o,g
    with torch.no_grad():
        ours.x2g.weight.copy_(perm(ref.weight_ih)); ours.h2g.weight.copy_(perm(ref.weight_hh))
        ours.x2g.bias.copy_(perm(ref.bias_ih + ref.bias_hh))
    x, h, c = torch.randn(B, I), torch.randn(B, H), torch.randn(B, H)
    h1, c1 = ref(x, (h, c)); h2, c2 = ours(x, (h, c))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_shapes_and_param_counts():
    I, H, P, B = 16, 32, 8, 4
    cells = {"lstm": VanillaLSTMCell(I, H), "cifg": CIFGCell(I, H), "wmc": WMCCell(I, H), "lstmp": LSTMPCell(I, H, P)}
    n = {k: sum(p.numel() for p in v.parameters()) for k, v in cells.items()}
    for k, cell in cells.items():
        h, c = cell.init_state(B, "cpu")
        h2, c2 = cell(torch.randn(B, I), (h, c))
        assert h2.shape == (B, cell.out_size) and c2.shape == (B, H), k
    assert cells["lstmp"].out_size == P
    assert n["cifg"] < n["lstm"] < n["wmc"]          # CIFG removes a gate, WMC adds 3 HxH peepholes


def test_cifg_input_gate_is_one_minus_forget():
    torch.manual_seed(1)
    cell = CIFGCell(4, 6)
    x, h, c = torch.randn(2, 4), torch.zeros(2, 6), torch.ones(2, 6)
    f, o, g = (cell.x2g(x) + cell.h2g(h)).chunk(3, -1)
    f, o, g = torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
    c_exp = f * c + (1 - f) * g
    _, c_new = cell(x, (h, c))
    assert torch.allclose(c_new, c_exp, atol=1e-6)


if __name__ == "__main__":
    test_vanilla_matches_torch_lstmcell(); test_shapes_and_param_counts(); test_cifg_input_gate_is_one_minus_forget()
    print("all tests passed")
