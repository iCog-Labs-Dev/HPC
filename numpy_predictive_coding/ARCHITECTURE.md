# Predictive coding network architecture and mathematical foundations

## Architecture

The default MNIST network has four layers counting the input, with three
trainable affine transformations and two hidden layers. A batch of $B$ images
is represented by $X \in \mathbb{R}^{784 \times B}$, with each standardized
$28 \times 28$ image flattened into a column. The one-hot label matrix
$Y \in \mathbb{R}^{10 \times B}$ uses the same column order. Bias columns are
broadcast across the batch.

| Layer | Width | Weight shape | Bias shape | Forward activation |
| --- | ---: | --- | --- | --- |
| Input | 784 | — | — | Standardized pixel values |
| Hidden 1 | 256 | `W1` (256 × 784) | `b1` (256 × 1) | Sigmoid |
| Hidden 2 | 64 | `W2` (64 × 256) | `b2` (64 × 1) | Sigmoid |
| Output | 10 | `W3` (10 × 64) | `b3` (10 × 1) | Softmax over classes |

The default model has 217,728 weights and 330 biases, for 218,058 trainable
scalars. The hidden widths can be set with `--h1` and `--h2`. In the general
training API, the feature and label arrays determine the input and output widths.
Each weight matrix has shape `(next_width, previous_width)`.

With $A_0 = X$, the forward pass computes

$$
\begin{aligned}
Z_\ell &= W_\ell A_{\ell-1} + b_\ell \quad (\ell=1,2,3), \\
A_1 &= \sigma(Z_1), \qquad A_2 = \sigma(Z_2), \qquad
A_3 = \operatorname{softmax}(Z_3).
\end{aligned}
$$

The sigmoid is $\sigma(z) = 1/(1+e^{-z})$, with derivative
$\sigma'(z) = \sigma(z)(1-\sigma(z))$. Softmax is applied independently to
each sample column. For a logit vector $z$, the implementation computes it
with $c=\max_k z_k$ for numerical stability.

$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i-c)}{\sum_k \exp(z_k-c)}.
$$

## Weight and hidden-state initialization

By default, `initialize` uses a NumPy random generator seeded with `0` and
draws the weights with Xavier (Glorot) normal initialization. Let $n_{\mathrm{in}}$
and $n_{\mathrm{out}}$ be the layer's input and output widths, and $g$ its
initialization gain, which defaults to $1$. Each weight is drawn from a normal
distribution with zero mean and standard deviation
$g\sqrt{2/(n_{\mathrm{in}}+n_{\mathrm{out}})}$. The optional uniform variant
draws from $[-a,a]$, where $a=g\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})}$.
Weights are stored as `float32` and biases start as `float32` zeros. The training
API can also start from a `float32` copy of supplied parameters.

At the start of every batch, `inference` copies the current forward hidden
activations into $H_1^{(0)}=A_1$ and $H_2^{(0)}=A_2$. These copies are the
starting point for the batch's hidden-state updates. Both hidden prediction
errors are zero at initialization, while the output error signal can already
be nonzero.

## Prediction errors and energy

During training inference, $X$, $Y$, and the parameters are held fixed. At the
current hidden states, `compute_errors` computes the layer predictions
$\mu_1,\mu_2$, the hidden residuals $e_1,e_2$, and the output logit derivative
$\delta_3$ (`dZ3` in the code).

$$
\begin{aligned}
\mu_1 &= \sigma(W_1 X + b_1), & e_1 &= H_1 - \mu_1, \\
\mu_2 &= \sigma(W_2 H_1 + b_2), & e_2 &= H_2 - \mu_2, \\
\widetilde Z_3 &= W_3 H_2 + b_3, & p &= \operatorname{softmax}(\widetilde Z_3), \\
\delta_3 &= p - Y.
\end{aligned}
$$

For sample column $j$, let $e_{1,j}$ and $e_{2,j}$ be column $j$ of the two
hidden residual matrices. The energy combines two squared prediction penalties
and a softmax cross-entropy term. Its unclipped form underlies the training
derivatives.

$$
E_j = \frac{1}{2}\left(\lVert e_{1,j}\rVert_2^2 + \lVert e_{2,j}\rVert_2^2\right)
      - \sum_{k=1}^{10} Y_{kj}\log p_{kj}.
$$

`compute_energy` averages the energy over samples and clips the probabilities
inside the logarithm to $[10^{-7},1]$. Training uses the standard derivative
of unclipped softmax cross-entropy, $p-Y$, independently of the clipping in
the reported loss.

## Hidden-state inference

The hidden states are updated by gradient descent on the sum of the per-sample
energies. Because each column is a separate sample, the activity updates have
no batch-size divisor. With $\odot$ denoting elementwise multiplication, the
hidden-state gradients used by `inference` are

$$
\begin{aligned}
G_{H_1} &= e_1 - W_2^\top\left[e_2 \odot \mu_2 \odot (1-\mu_2)\right], \\
G_{H_2} &= e_2 + W_3^\top\delta_3.
\end{aligned}
$$

Each step computes both gradients from the same current states and then applies
the two updates simultaneously. The default is 20 steps at activity learning
rate $\eta_H=0.05$.

$$
H_1 \leftarrow H_1 - \eta_H G_{H_1}, \qquad
H_2 \leftarrow H_2 - \eta_H G_{H_2}.
$$

The updated states are unconstrained, so they can leave the sigmoid's output
range. Inference recomputes all errors at the final states before learning.

## Weight and bias gradients

`parameter_gradients` differentiates the mean energy with the final inferred
states held fixed. The local preactivation derivatives for the first two layers
come from their squared prediction errors. The output uses $\delta_3=p-Y$
from those same final states.

$$
\delta_1 = -e_1 \odot \mu_1 \odot (1-\mu_1), \qquad
\delta_2 = -e_2 \odot \mu_2 \odot (1-\mu_2).
$$

Let $U_0=X$, $U_1=H_1$, and $U_2=H_2$. For each layer $\ell \in \{1,2,3\}$,
the parameter gradients are

$$
\nabla_{W_\ell}\overline E = \frac{\delta_\ell U_{\ell-1}^{\top}}{B}, \qquad
\nabla_{b_\ell}\overline E = \frac{\delta_\ell \mathbf{1}_B}{B}, \qquad
\overline E = \frac{1}{B}\sum_{j=1}^B E_j,
$$

where $\mathbf{1}_B$ is a column vector of ones. Each gradient has the same
shape as its parameter. The divisor is the actual number of samples, including
for the final partial batch.

## AdamW parameter update

`weight_update` applies Adam with decoupled weight decay (AdamW) to all three
weight matrices and all three biases. For each parameter tensor $\theta$, let
$g_t$ be its batch gradient and $m_t,v_t$ its elementwise first and second
moments. Moments start at zero and persist across batches and epochs. The
shared step counter $t$ advances once per batch, starting at $t=1$ for the
first update.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^{\odot 2}, \\
\widehat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad
\widehat v_t = \frac{v_t}{1-\beta_2^t}, \\
\theta_t &= \theta_{t-1} - \eta_W\left(
  \frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon} + \lambda\theta_{t-1}\right).
\end{aligned}
$$

Here $g_t^{\odot 2}$ is the elementwise square. The defaults are
$\eta_W=0.001$, $\lambda=0.1$, $\beta_1=0.9$, $\beta_2=0.999$, and
$\epsilon=10^{-8}$. The decay term is applied directly to the parameters,
outside the moment calculations. The default MNIST run trains for 20 epochs
with shuffled batches of up to 200 samples and shuffle seed `42`.

## Evaluation and prediction

`evaluate` runs a forward pass on each batch with the current parameters. The
predictions are the digit indices with the largest forward probabilities
$\widehat y_j = \arg\max_k A_{3,kj}$. Evaluation uses labels to compute the
metrics, while `predict` returns the same class indices from inputs alone.
For $N$ examples, the reported metrics are

$$
\begin{aligned}
\mathrm{CE} &= -\frac{1}{N}\sum_{j=1}^{N}\sum_{k=1}^{10}
Y_{kj}\log\left(\operatorname{clip}(A_{3,kj},10^{-7},1)\right), \\
\mathrm{Accuracy} &= \frac{1}{N}\sum_{j=1}^{N}
\mathbf{1}\!\left[\widehat y_j = \arg\max_k Y_{kj}\right].
\end{aligned}
$$

Each example contributes equally. The evaluator weights each batch's mean loss
by its size and sums correct predictions before dividing by $N$. Training records
forward metrics on the full training split before the first update (epoch zero)
and after each epoch. The MNIST CLI evaluates the held-out test split once after
training. Both splits use the fixed preprocessing $(x/255-0.1307)/0.3081$.

## Implementation references

| Source | Details |
| --- | --- |
| [network.py](numpy_predictive_coding/network.py) | Initialization, forward pass, and class prediction |
| [core.py](numpy_predictive_coding/core.py) | Prediction errors, energy, hidden-state inference, and parameter gradients |
| [optimization.py](numpy_predictive_coding/optimization.py) | AdamW moments and updates |
| [training.py](numpy_predictive_coding/training.py) | Mini-batch training, metric aggregation, and history |
| [data.py](numpy_predictive_coding/data.py), [train_mnist.py](train_mnist.py) | MNIST preprocessing, CLI defaults, and test evaluation |
