# Hopfield Networks: Classical, Storkey, and Modern

A NumPy project comparing three associative-memory approaches: classical Hopfield with Hebbian learning, classical Hopfield with Storkey learning, and continuous modern Hopfield with softmax retrieval.

## What is a Hopfield network?

A Hopfield network is a **content-addressable memory**: you provide a noisy version of a stored pattern, and the network attempts to recover the original. You retrieve by similarity to the content rather than by supplying a memory address.

In this project, each stored pattern is a vector of `n` values, each either `-1` or `+1`. Noise is introduced by flipping selected signs. For example:

```python
original = np.array([ 1,  1, -1, -1])
noisy    = np.array([-1,  1, -1, -1])
```

The goal is to recover `original` from `noisy`. Recovery is not guaranteed: it depends on the stored patterns, noise, model, and settings. This benchmark demonstrates noise removal. It does not separately implement an experiment with missing coordinates.

Notation:

| Symbol | Meaning |
|---|---|
| `n` | Number of coordinates / neurons per pattern |
| `P_count` | Number of stored patterns |
| `W` | Classical or Storkey connection matrix, shape `(n, n)` |
| `P` | Modern memory matrix, shape `(P_count, n)` |
| `x` | Current state or noisy query, shape `(n,)` |
| `beta` | Selectivity of modern softmax retrieval |

Storkey is a different learning rule for the classical architecture, rather than a completely separate recall architecture. “Modern” here means the continuous softmax model implemented in this project; other modern Hopfield variants also exist.

## 1. Classical Hopfield with Hebbian learning

### Definition and storage

The network represents memories through shared pairwise connection weights. Each pattern contributes its outer product:

$$
W = \frac{1}{n}\sum_{\mu=1}^{P_{count}} p^\mu(p^\mu)^\top,
\qquad W_{ii}=0.
$$

In code:

```python
self.W.fill(0)
for p in patterns:
    self.W += np.outer(p, p)
np.fill_diagonal(self.W, 0)
self.W /= self.n
```

`np.outer(p, p)` creates a matrix with entry `p[i] * p[j]`. Positive weights encourage matching signs; negative weights encourage opposite signs. Zero weights contribute no input. Training calculates these weights; it does not use backpropagation or class labels.

### Recall and energy

Each neuron receives a weighted sum:

$$
h_i=\sum_j W_{ij}x_j.
$$

The implementation updates one neuron at a time, in a random order, using its latest available neighbors:

```python
x[i] = 1 if self.W[i] @ x >= 0 else -1
```

A sweep updates all neurons once. The default is ten sweeps. Zero input is resolved to `+1`.

The energy is a scalar score for the whole state:

$$
E(x)=-\frac12 x^\top W x.
$$

```python
inputs = self.W @ x          # one input per neuron
energy = -0.5 * (x @ inputs) # one score for the entire state
```

For symmetric weights with zero diagonal, asynchronous sign updates cannot increase this energy. Lower energy does **not** guarantee the intended memory: unwanted stable states can also have low energy.

### Capacity and limitation

The familiar retrieval threshold is approximately:

$$
P_{count}\approx0.138n.
$$

This concerns large networks with independent, unbiased random binary patterns and a retrieval criterion allowing a small fraction of errors [1]. It is not a hard finite-size cutoff or an exact-recovery guarantee. For `n=100`, “about 14 patterns” is a theoretical reference, not a promise of perfect recall with 20% noise.

Capacity is limited by **cross-talk**: every memory modifies the same connections, so other memories can overturn the weighted input supporting the target. Even an initially correct bit can flip.

## 2. Hopfield with Storkey learning

### Definition

Storkey retains the classical energy and recall update, but corrects the Hebbian weight additions using the network's existing inputs. The aim is to reduce interference while keeping learning local and incremental in principle [2].

For each new pattern `p`, using the weights before that pattern's update:

$$
h_{ij}=\sum_{k\ne i,j}W_{ik}p_k,
$$

$$
C_{ij}=h_{ij}p_j+p_i h_{ji},
$$

$$
\Delta W_{ij}=\frac{p_ip_j-C_{ij}}{n},\qquad i\ne j.
$$

The implementation follows those steps directly:

```python
field = self.W @ p
h = field[:, None] - self.W * p[None, :]
correction = h * p[None, :] + h.T * p[:, None]
delta_W = (np.outer(p, p) - correction) / self.n
np.fill_diagonal(delta_W, 0)
self.W += delta_W
```

| Expression | Meaning |
|---|---|
| `field[i]` | Total input to neuron `i` when `p` is presented |
| `h[i, j]` | That input excluding neuron `j` |
| `h * p[None, :]` | Multiply column `j` by `p[j]`, element by element |
| `h.T * p[:, None]` | Contribution from the other end of each connection |
| `correction` | Matrix adjusting the ordinary outer-product addition |
| `delta_W` | Matrix added to the old weights |

All off-diagonal corrections are computed before changing `W`. Diagonal intermediate values are irrelevant because the diagonal update is set to zero.

The first pattern sees zero existing weights, so its correction is zero. Later corrections can strengthen, weaken, or reverse particular additions. Storkey is not guaranteed to preserve every old memory. Unlike the Hebbian sum, sequential Storkey training can depend on pattern order.

`StorkeyHopfield` inherits from `Hopfield`; only `train()` is overridden. Both implementations reset their weights at the start of a training call. Although the Storkey rule supports incremental learning, this API treats each `train()` call as a fresh run.

### Capacity

Under the absolute stability criterion used in Storkey's capacity analysis, reported asymptotic scalings are [2]:

| Learning rule | Absolute capacity scaling |
|---|---|
| Hebbian | $n/(2\ln n)$ |
| Storkey | $n/\sqrt{2\ln n}$ |

This criterion concerns exact stability of stored patterns with high probability as the network grows. It is different from the retrieval criterion behind `0.138n`, and from recovering a noisy query. Do not compare these formulas as if they were the same measurement or use them as exact small-network limits.

## 3. Continuous modern Hopfield

### Definition and storage

This implementation keeps each pattern as a row of `P`, rather than combining them into a single pairwise matrix:

```python
self.P = np.array(patterns, dtype=float)
```

The softmax Hopfield formulation has exponential capacity in pattern dimension under suitable separation and parameter conditions [3]. This is not a universal guarantee for arbitrary patterns.

### Recall in the code

```python
scores = self.beta * self.P @ x
w = softmax(scores)
x = w @ self.P
```

1. Dot products compare the query with every memory.
2. Softmax converts scores into positive weights summing to one.
3. Each scalar weight multiplies its corresponding pattern vector; the vectors are summed.

Explicitly, the last operation is:

```python
x = (w[:, None] * self.P).sum(axis=0)
```

Softmax uses `exp(score) / sum(exp(scores))`. Subtracting the maximum first improves numerical stability without changing the weights. They form a normalized distribution over memories, but are not automatically calibrated probabilities that a memory is correct.

Larger `beta` concentrates influence on better matches; it cannot correct a similarity ranking that already favors the wrong memory. The state remains continuous. The benchmark applies `np.sign` afterward for binary scoring; this thresholding is outside the continuous update.

The implemented energy is:

$$
E(x)=-\frac1\beta\log\sum_\mu\exp(\beta p^{\mu\top}x)+\frac12x^\top x.
$$

Pattern-independent constants are omitted. This energy should not be compared numerically against classical energy to rank the models.

### Capacity and resource tradeoff

Exponential retrieval capacity does not mean exponential compression into `n*n` weights. This implementation stores `P_count*n` values and compares the query against all memories. Its storage and retrieval costs grow with the memory count. Correlated or insufficiently separated patterns can still cause ambiguous retrieval.

## Model overview

| Property | Classical Hebbian | Storkey | Modern softmax |
|---|---|---|---|
| Stored representation | Shared `W` | Shared `W` | Explicit pattern rows `P` |
| State during recall | Binary | Binary | Continuous |
| Main mechanism | Weighted sign updates | Same recall, corrected learning | Similarity, softmax, weighted sum |
| Storage in this implementation | $O(n^2)$ | $O(n^2)$ | $O(P_{count}n)$ |
| Learning / storing work | $O(P_{count}n^2)$ | $O(P_{count}n^2)$, more operations | $O(P_{count}n)$ copy |
| Recall work | $O(sn^2)$ for `s` sweeps | $O(sn^2)$ for `s` sweeps | $O(tP_{count}n)$ for `t` updates |

## Files and usage

| File | Purpose |
|---|---|
| `hopfield.py` | Classical Hebbian training, energy, asynchronous recall |
| `storkey_hopfield.py` | Storkey training; inherits classical energy and recall |
| `modern_hopfield.py` | Explicit memory storage, softmax, continuous recall and energy |
| `compare.py` | Shared noisy-query experiment and success table |
| `README.md` | Model definitions, capacity context, usage, and findings |

Requires Python 3 and NumPy. Run inside the project folder:

```bash
python -m pip install numpy
python compare.py
```

There are no command-line arguments. Edit the settings at the top of `main()` in `compare.py`:

```python
n = 100
flip = 20
trials = 20
seed = 0
pattern_counts = [5, 10, 14, 20, 50, 100, 200, 500,
                  1000, 10000]
```

The pattern-count list matches the original comparison, including ten thousand patterns. `flip` is a count of flipped bits: `flip = 20` means 20% only when `n = 100`.

For a shorter run, temporarily change `pattern_counts` to `[5, 14, 50]` and `trials` to `3`, then run the same command:

```bash
python compare.py
```

The full list can take a long time, especially for sequential Storkey training at 100,000 and 1,000,000 patterns. The larger sizes also need more memory because the generated patterns and modern model's pattern copy coexist.

## Experiment design

For each pattern count, the script repeats these steps:

1. Generate a fresh set of random binary patterns.
2. Select a random stored target, sampling different positions in Storkey's training order.
3. Flip exactly `flip` distinct bits of that target.
4. Give all three models the same memories and noisy query.
5. Run classical and Storkey recall for ten sweeps with identical random update orders.
6. Run modern recall for one update with `beta=4.0`, then apply `np.sign`.
7. Count success only when **every output bit** matches the target.

Each table entry is successful trials divided by total trials. The benchmark samples one target per fresh network; it does not test all memories or measure basin sizes.

## Observed findings

The following output was obtained in the earlier benchmark with `n=100`, 20 flipped bits, 20 trials per count, and seed 0. That run used `[5, 10, 14, 20, 30, 50, 100, 200]`; the current script uses the original list shown above instead. The table has not been regenerated for the expanded list, so the row for 30 is historical and no results for 500 through 1,000,000 are claimed:

| Stored patterns | Classical | Storkey | Modern |
|---:|---:|---:|---:|
| 5 | 1.00 | 1.00 | 1.00 |
| 10 | 1.00 | 1.00 | 1.00 |
| 14 | 0.55 | 0.95 | 1.00 |
| 20 | 0.00 | 1.00 | 1.00 |
| 30 | 0.00 | 0.50 | 1.00 |
| 50 | 0.00 | 0.00 | 1.00 |
| 100 | 0.00 | 0.00 | 1.00 |
| 200 | 0.00 | 0.00 | 1.00 |

Interpretation:

- At low load, all three methods recovered all sampled targets.
- Storkey improved exact recovery at intermediate loads: at 20 patterns, it succeeded in 20/20 trials versus 0/20 for Hebbian learning.
- Storkey still failed at higher loads in this test; correction reduces interference without eliminating it.
- Modern retrieval succeeded on all sampled targets through 200 patterns, using explicit memory storage and selective retrieval.

These results describe this experiment. They do **not** prove a capacity formula, establish a universal ranking, or show equal resource efficiency. Twenty trials give only 0.05 resolution; 1.00 means 20/20 observed successes, not guaranteed future success. Different seeds, corruption, correlations, and update budgets can change the results. No runtime or peak-memory benchmark is included.

Testing a million patterns with Storkey across 20 trials can be expensive because learning is sequential and performs matrix-sized operations for each pattern. Increase counts gradually.

## Implementation checks

The Storkey implementation was checked against a direct connection-by-connection reference, and against the manually calculated six-neuron A/B/C example. Symmetry and zero diagonal were checked. The previous comparison settings produced the table above. After removing the argument parser and restoring the original pattern-count list, a reduced smoke run was checked; the full million-pattern configuration was not executed.

## References


1. Amos Storkey (1997), [Increasing the Capacity of a Hopfield Network without Sacrificing Functionality](https://homepages.inf.ed.ac.uk/amos/publications/Storkey1997IncreasingtheCapacityoftheHopfieldNetworkwithoutSacrificingFunctionality.pdf). See also the author's [learning-rule overview](https://homepages.inf.ed.ac.uk/amos/learning.html) for the absolute-capacity comparison.
2. Hubert Ramsauer et al. (2021), [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217). Continuous modern Hopfield networks, softmax retrieval, and conditional exponential capacity.
3. Mihailo Stojnic (2024), [Capacity of the Hebbian-Hopfield network associative memory](https://arxiv.org/abs/2403.01907). Retrieval capacity and the approximately 0.138 load threshold.