"""Readable, small Transformer reference for Attention Is All You Need.
Original post-LN layout; synthetic data only. Requires Python 3 and PyTorch.
Run: python transformer_reference.py
This is NOT a trained translator or a reproduction of the paper's BLEU.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiHead(nn.Module):
    """Batch-first; Boolean masks use True = blocked."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.dk = heads, d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dk).transpose(1, 2)

    def forward(self, query, key, value, *, causal=False, key_pad=None):
        # query [B,m,d], key/value [B,n,d]; m can differ from n.
        q, k, v = self.split(self.q_proj(query)), self.split(self.k_proj(key)), self.split(self.v_proj(value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        m, n = scores.shape[-2:]
        if causal:
            if m != n:
                raise ValueError("This simple causal path expects equal lengths; no KV cache.")
            future = torch.ones(m, n, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        if key_pad is not None:
            scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
        # Do not silently generate NaN for an entirely masked query.
        if torch.isneginf(scores).all(dim=-1).any():
            raise ValueError("Every query must have at least one allowed key.")
        weights = scores.softmax(dim=-1)
        # Kept (pre-dropout) only so training scripts can plot attention heatmaps.
        self.last_weights = weights.detach()
        out = self.dropout(weights) @ v
        b = out.size(0)
        out = out.transpose(1, 2).contiguous().reshape(b, m, self.heads * self.dk)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.first = nn.Linear(d_model, d_ff)
        self.second = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.second(F.relu(self.first(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHead(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_pad):
        x = self.norm1(x + self.drop(self.attn(x, x, x, key_pad=src_pad)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHead(d_model, heads, dropout)
        self.cross_attn = MultiHead(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_pad, tgt_pad):
        x = self.norm1(x + self.drop(self.self_attn(
            x, x, x, causal=True, key_pad=tgt_pad)))
        x = self.norm2(x + self.drop(self.cross_attn(
            x, memory, memory, key_pad=src_pad)))
        return self.norm3(x + self.drop(self.ffn(x)))


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        if d_model % 2:
            raise ValueError("This educational implementation uses even d_model.")
        pos = torch.arange(max_len).float().unsqueeze(1)
        inverse_scale = torch.exp(torch.arange(0, d_model, 2).float()
                                  * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * inverse_scale)
        pe[:, 1::2] = torch.cos(pos * inverse_scale)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds the configured positional buffer.")
        return x + self.pe[:, :x.size(1)].to(dtype=x.dtype)


class Transformer(nn.Module):
    """Shared source/target vocabulary. PAD=0; each source must be nonempty."""
    def __init__(self, vocab_size, d_model=32, heads=4, d_ff=64,
                 layers=2, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id, self.scale = pad_id, math.sqrt(d_model)
        # Intentionally no padding_idx: tied output weights include all tokens.
        # Padding keys are masked and PAD labels are ignored in the loss.
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPosition(d_model)
        self.drop = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        # This initialization is a teaching choice, not a full training recipe.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, ids):
        return self.drop(self.position(self.embedding(ids) * self.scale))

    def forward(self, src, tgt_in):
        src_pad, tgt_pad = src.eq(self.pad_id), tgt_in.eq(self.pad_id)
        memory = self.embed(src)
        for layer in self.encoders:
            memory = layer(memory, src_pad)
        x = self.embed(tgt_in)
        for layer in self.decoders:
            x = layer(x, memory, src_pad, tgt_pad)
        return self.output(x)  # logits, NOT softmax probabilities


def learning_rate(step, d_model=512, warmup=4000):
    if step < 1 or warmup < 1:
        raise ValueError("step and warmup must be positive")
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def smoke_test():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = Transformer(vocab_size=20)
    src = torch.tensor([[3, 4, 5, 0], [7, 8, 9, 10]])
    # BOS=1, EOS=2. Right-padded; BOS guarantees a valid first key.
    tgt = torch.tensor([[1, 11, 12, 2, 0], [1, 13, 14, 15, 2]])
    inp, labels = tgt[:, :-1], tgt[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(1, 32),
                                 betas=(0.9, 0.98), eps=1e-9)
    model.train()
    optimizer.zero_grad()
    logits = model(src, inp)
    # PyTorch smoothing mixes with a uniform distribution over all classes.
    loss = F.cross_entropy(logits.reshape(-1, 20), labels.reshape(-1),
                           ignore_index=0, label_smoothing=0.1)
    assert logits.shape == (2, 4, 20)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        before = model(src, inp)
        changed = inp.clone()
        changed[:, 2:] = torch.tensor([[16, 17], [18, 19]])
        after = model(src, changed)
        torch.testing.assert_close(before[:, :2], after[:, :2],
                                   rtol=1e-5, atol=1e-6)
    print("PASS logits shape:", tuple(logits.shape))
    print("PASS finite loss and gradients; one optimizer step:", float(loss.detach()))
    print("PASS future-token perturbation does not change earlier logits")
    print("Synthetic smoke test only; translation quality is not evaluated.")


if __name__ == "__main__":
    smoke_test()

