"""Train the small Transformer on a real (toy) task: reversing digit sequences.
Example: source 3 7 1 9 -> target 9 1 7 3.
Run: python train_reverse.py
Outputs: outputs/loss_curve.png, outputs/attention_heatmap.png, printed accuracy.
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from transformer import Transformer, learning_rate

PAD, BOS, EOS = 0, 1, 2
DIGIT0 = 3                      # token id of digit 0; digits 0-9 -> ids 3-12
VOCAB = DIGIT0 + 10
MIN_LEN, MAX_LEN = 3, 10
D_MODEL, STEPS, BATCH, WARMUP = 64, 3000, 64, 400
OUT = Path(__file__).parent / "outputs"


def make_batch(n, rng):
    """Right-padded source and target (BOS + reversed + EOS) tensors."""
    src = torch.full((n, MAX_LEN), PAD)
    tgt = torch.full((n, MAX_LEN + 2), PAD)
    for i in range(n):
        seq = [rng.randrange(10) + DIGIT0 for _ in range(rng.randint(MIN_LEN, MAX_LEN))]
        src[i, :len(seq)] = torch.tensor(seq)
        tgt[i, :len(seq) + 2] = torch.tensor([BOS] + seq[::-1] + [EOS])
    return src, tgt


def loss_on(model, src, tgt):
    logits = model(src, tgt[:, :-1])
    return F.cross_entropy(logits.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1),
                           ignore_index=PAD, label_smoothing=0.1)


@torch.no_grad()
def greedy_decode(model, src):
    out = torch.full((src.size(0), 1), BOS, device=src.device)
    for _ in range(MAX_LEN + 1):
        next_tok = model(src, out)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out[:, 1:]


def sequence_accuracy(model, src, tgt):
    """Exact match of every token up to and including EOS."""
    pred = greedy_decode(model, src)
    labels = tgt[:, 1:]
    mask = labels.ne(PAD)
    return ((pred == labels) | ~mask).all(dim=1).float().mean().item()


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(VOCAB, d_model=D_MODEL, heads=4, d_ff=128, layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # LambdaLR multiplies lr=1.0 by the paper's warmup schedule.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: learning_rate(s + 1, D_MODEL, WARMUP))
    val_src, val_tgt = (t.to(device) for t in make_batch(1000, random.Random(123)))

    train_hist, val_hist = [], []
    snapshots = [parameters_to_vector(model.parameters()).detach().clone()]
    for step in range(1, STEPS + 1):
        model.train()
        src, tgt = (t.to(device) for t in make_batch(BATCH, rng))
        loss = loss_on(model, src, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        train_hist.append(loss.item())
        if step % 100 == 0:
            snapshots.append(parameters_to_vector(model.parameters()).detach().clone())
            model.eval()
            with torch.no_grad():
                val_hist.append((step, loss_on(model, val_src, val_tgt).item()))
            print(f"step {step:5d}  train {loss.item():.3f}  val {val_hist[-1][1]:.3f}")

    model.eval()
    acc = sequence_accuracy(model, val_src, val_tgt)
    print(f"Held-out exact-sequence accuracy (1000 sequences): {acc:.1%}")

    OUT.mkdir(exist_ok=True)
    plot_loss(train_hist, val_hist, acc)
    plot_attention(model, device)
    plot_loss_landscape(model, snapshots, val_src, val_tgt)


def plot_loss(train_hist, val_hist, acc):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train_hist) + 1), train_hist, alpha=0.35, label="train (per batch)")
    ax.plot(*zip(*val_hist), marker="o", ms=3, label="validation")
    ax.set(xlabel="step", ylabel="cross-entropy (label smoothing 0.1)",
           title=f"Sequence reversal - final held-out accuracy {acc:.1%}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_attention(model, device):
    digits = [3, 7, 1, 9, 4, 6, 0, 8]
    src = torch.tensor([[d + DIGIT0 for d in digits]], device=device)
    tgt_in = torch.cat([torch.tensor([[BOS]], device=device), greedy_decode(model, src)], 1)
    tgt_in = tgt_in[:, :len(digits) + 1]          # BOS + predicted digits, drop trailing EOS
    model(src, tgt_in)
    w = model.decoders[-1].cross_attn.last_weights[0].cpu()   # [heads, tgt, src]

    def name(t):
        return {BOS: "BOS", EOS: "EOS"}.get(t, str(t - DIGIT0))
    out_tokens = greedy_decode(model, src)[0, :len(digits) + 1].tolist()
    ylabels = [f"-> {name(t)}" for t in out_tokens]   # what each query row predicts
    xlabels = [str(d) for d in digits]

    panels = [("mean of heads", w.mean(0))] + [(f"head {h}", w[h]) for h in range(w.size(0))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (title, m) in zip(axes, panels):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_yticks(range(len(ylabels)), ylabels if ax is axes[0] else [])
        ax.set_xlabel("source token (key)")
    axes[0].set_ylabel("decoder step (query)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    fig.suptitle(f"Last-layer cross-attention, input {' '.join(xlabels)}")
    fig.savefig(OUT / "attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_loss_landscape(model, snapshots, val_src, val_tgt, n=31):
    """Loss surface on the plane of the top-2 PCA directions of the training trajectory
    (Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets"), with the path drawn on it."""
    params = list(model.parameters())
    final = parameters_to_vector(params).clone()
    offsets = torch.stack(snapshots) - final                  # [T, P], relative to final weights
    _, s, vh = torch.linalg.svd(offsets, full_matrices=False)
    dirs = vh[:2]                                              # two orthonormal directions
    explained = (s[:2] ** 2 / (s ** 2).sum()).tolist()
    path = offsets @ dirs.T                                    # trajectory projected onto the plane

    def loss_at(a, b):
        vector_to_parameters(final + a * dirs[0] + b * dirs[1], params)
        return loss_on(model, val_src, val_tgt).item()

    pad = 0.25
    lo, hi = path.min(0).values, path.max(0).values
    span = hi - lo
    xs = torch.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], n)
    ys = torch.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], n)
    zs = torch.tensor([[loss_at(x, y) for x in xs] for y in ys])
    path_z = [loss_at(a, b) for a, b in path.tolist()]
    vector_to_parameters(final, params)                       # restore trained weights

    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    px, py = path[:, 0].cpu(), path[:, 1].cpu()
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, zs, cmap="viridis", alpha=0.75, linewidth=0)
    ax.plot(px, py, path_z, color="red", marker="o", ms=3, lw=2, label="training path")
    ax.scatter(px[0], py[0], path_z[0], color="white", edgecolor="k", s=60, label="init")
    ax.scatter(px[-1], py[-1], path_z[-1], color="red", edgecolor="k", s=80, marker="*", label="final")
    ax.set(xlabel=f"PC1 ({explained[0]:.0%} of path variance)",
           ylabel=f"PC2 ({explained[1]:.0%})", zlabel="validation loss")
    ax.view_init(elev=30, azim=-130)
    ax.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(X, Y, zs, levels=30, cmap="viridis")
    ax2.plot(px, py, color="red", marker="o", ms=3, lw=2)
    ax2.scatter(px[0], py[0], color="white", edgecolor="k", s=60, zorder=3)
    ax2.scatter(px[-1], py[-1], color="red", edgecolor="k", s=120, marker="*", zorder=3)
    ax2.set(xlabel="PC1", ylabel="PC2", title="Top view (contours)")
    fig.colorbar(cs, ax=ax2, label="validation loss")
    fig.suptitle("Loss landscape and gradient-descent trajectory (projected to 2D)")
    fig.savefig(OUT / "loss_landscape_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
