This is almost certainly about residual *connections* / residual learning
(ResNet, He et al. 2015) given the deep-learning framing of "core issue... show
how residuals address them" and "seminal papers." Let me build a runnable,
verbose teaching notebook — but first I'll verify the environment and tune a
small experiment so the degradation phenomenon actually shows up
reproducibly.torch installed. Now let me tune the degradation experiment so it
reproduces reliably and quickly on CPU.The degradation reproduces beautifully —
plain nets collapse to the mean-predictor (train MSE ≈ 1.0) at depth 20+ while
residual nets stay at ≈0.001. Tanh makes the saturation/vanishing-gradient
effect stark. Let me time it and also prepare a gradient-norm
demonstration.~3.8 min is a touch long for embedded execution. Let me trim
epochs and confirm degradation still appears clearly, plus prototype the
gradient-flow demo.Both demonstrations are clean: the plain net collapses to
the mean-predictor at depth ≥24 while residual stays at ~0.001, and at init the
plain net's gradients vanish ~40,000× from output to input while the residual
net keeps them uniform. That's the whole story. Now I'll build the notebook.
Let me install the Jupyter execution tooling first.Now I'll write the notebook
builder. This assembles verbose teaching markdown and simple, well-named code
cells.Now execute it end-to-end to embed all the plots and outputs (raising the
per-cell timeout for the training sweeps).The docstrings got mangled by
quote-escaping inside the raw strings. Let me fix the delimiters and
rebuild.Executed cleanly (412 KB with embedded figures). Let me verify the key
results landed correctly and that no cell errored.Everything validates: 0
errors, 8 embedded figures, and the phenomenon reproduces cleanly — plain nets
collapse to the mean-predictor at depth ≥24 (MSE ≈ 0.88–0.92) while residual
nets stay at ≈0.001–0.003, with a ~44,000× gradient attenuation in the plain
net. Let me place it in outputs.I read "residual" as **residual learning / skip
connections** (ResNet) rather than statistical residuals, since the prompt asks
how residuals *address a core issue* — that's the ResNet story. The notebook
bridges both meanings so the etymology lands.

The notebook is fully executed, so every figure and number is already embedded (you can read it without running anything). It walks through:

1. **What "residual" means** — starts from the statistical residual (`observed − predicted`), then rearranges it to `observed = predicted + residual` and reuses that exact shape as `H(x) = F(x) + x`.
2. **A visible 1-D regression problem** — small, seeded, plottable.
3. **The degradation problem (He et al. 2015)** — plain deep nets are *re-run at increasing depth* and you watch training MSE fall off a cliff: depth 2→0.0009, depth 12→0.004, then depth 24→0.88 and depth 40→0.92 (i.e. collapse to the mean-predictor). The "deeper can always emulate shallower via identity layers" argument sets up *why this is surprising* — it's an optimization failure, not capacity.
4. **The mechanism** — a single backward pass shows gradients vanishing ~44,000× from output to input through the `tanh` stack.
5. **The derivation** — `F(x) = H(x) − x`, the `+ x` shortcut, the `∂/∂x = ∂F/∂x + 1` gradient highway, plus a demo that zeroing the residual function gives an exact identity (the "easy default" plain blocks lack).
6. **The payoff** — same depths with residual blocks stay at ~0.001–0.003, and a combined bar chart shows the plain-vs-residual gap.

It cites He et al. (1512.03385), Highway Networks (1505.00387), Identity Mappings (1603.05027), the vanishing-gradient lineage (Hochreiter; Bengio et al. 1994), and Transformers (1706.03762), and ends with four hands-on exercises — including deleting the `x +` to watch the residual net regress back into a collapsing plain net.

Variable names are long and the Python is kept flat (explicit loops, `nn.Sequential`, helper functions) for readability. One thing worth flagging: I used `tanh` deliberately because it makes the degradation dramatic and fast — with ReLU the cliff appears later and shallower, which is exactly what exercise 1 has you explore.
