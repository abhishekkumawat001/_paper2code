"""
Vanilla Recurrent Neural Network — Built from Scratch with NumPy
================================================================

Architecture:
    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
    y_t = W_hy · h_t + b_y

Training:
    - Backpropagation Through Time (BPTT)
    - Gradient clipping to prevent exploding gradients
    - Cross-entropy loss for character-level language modelling

Demo:
    Trains on a small text corpus and generates new text character-by-character.
"""

import numpy as np
from typing import Dict, Tuple, List


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """Average cross-entropy over a sequence of one-hot targets."""
    # probs  : (vocab_size, seq_len)
    # targets: list of integer indices, length = seq_len
    return -np.mean(np.log(probs[targets, np.arange(len(targets))] + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla RNN class
# ──────────────────────────────────────────────────────────────────────────────

class VanillaRNN:
    """
    A character-level Vanilla RNN built entirely with NumPy.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens (characters).
    hidden_size : int
        Dimensionality of the hidden state.
    learning_rate : float
        Step size for Adagrad updates.
    seq_length : int
        Number of time-steps to unroll during BPTT.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-2,
        seq_length: int = 25,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.seq_length = seq_length

        # ── Weight initialisation (Xavier / sqrt scaling) ─────────────────
        scale_xh = np.sqrt(1.0 / vocab_size)
        scale_hh = np.sqrt(1.0 / hidden_size)

        self.W_xh = np.random.randn(hidden_size, vocab_size) * scale_xh   # input  → hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh  # hidden → hidden
        self.W_hy = np.random.randn(vocab_size, hidden_size) * scale_hh   # hidden → output
        self.b_h  = np.zeros((hidden_size, 1))                            # hidden bias
        self.b_y  = np.zeros((vocab_size, 1))                             # output bias

        # Adagrad memory (accumulated squared gradients)
        self._mem = {k: np.zeros_like(v) for k, v in self._params().items()}

    # ── convenience: collect params into a dict ───────────────────────────
    def _params(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh,
            "W_hh": self.W_hh,
            "W_hy": self.W_hy,
            "b_h":  self.b_h,
            "b_y":  self.b_y,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Forward pass  (one full sequence)
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: List[int],
        targets: List[int],
        h_prev: np.ndarray,
    ) -> Tuple[float, Dict, np.ndarray]:
        """
        Run a forward pass over *one* training sequence.

        Parameters
        ----------
        inputs  : list[int]  — input character indices  (length T)
        targets : list[int]  — target character indices  (length T)
        h_prev  : (hidden_size, 1) — hidden state carried from previous chunk

        Returns
        -------
        loss    : scalar cross-entropy loss
        cache   : dict of intermediate values needed for backward pass
        h_last  : final hidden state (to carry forward)
        """
        T = len(inputs)
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0

        for t in range(T):
            # One-hot encode the input character
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1.0

            # Hidden state: h_t = tanh(W_xh · x_t  +  W_hh · h_{t-1}  +  b_h)
            hs[t] = np.tanh(
                self.W_xh @ xs[t] + self.W_hh @ hs[t - 1] + self.b_h
            )

            # Output logits:  y_t = W_hy · h_t  +  b_y
            ys[t] = self.W_hy @ hs[t] + self.b_y

            # Probabilities via softmax
            ps[t] = softmax(ys[t])

            # Accumulate cross-entropy loss: -log P(target)
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        cache = {"xs": xs, "hs": hs, "ps": ps}
        return loss, cache, hs[T - 1]

    # ──────────────────────────────────────────────────────────────────────
    # Backward pass  (BPTT)
    # ──────────────────────────────────────────────────────────────────────

    def backward(
        self,
        targets: List[int],
        cache: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Backpropagation Through Time.

        Returns a dict of gradients with the same keys as `_params()`.
        """
        xs, hs, ps = cache["xs"], cache["hs"], cache["ps"]
        T = len(targets)

        # Initialise gradient accumulators
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h  = np.zeros_like(self.b_h)
        db_y  = np.zeros_like(self.b_y)

        dh_next = np.zeros_like(hs[0])  # gradient flowing from future time-step

        for t in reversed(range(T)):
            # ── Output gradient ───────────────────────────────────────────
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0            # ∂L/∂y  =  p - one_hot(target)

            dW_hy += dy @ hs[t].T
            db_y  += dy

            # ── Hidden-state gradient ─────────────────────────────────────
            dh = self.W_hy.T @ dy + dh_next  # from output + from future step

            # Back-propagate through tanh:  dtanh = (1 - h²) * dh
            dh_raw = (1.0 - hs[t] ** 2) * dh

            dW_xh += dh_raw @ xs[t].T
            dW_hh += dh_raw @ hs[t - 1].T
            db_h  += dh_raw

            # Pass gradient to previous time-step
            dh_next = self.W_hh.T @ dh_raw

        grads = {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "W_hy": dW_hy,
            "b_h":  db_h,
            "b_y":  db_y,
        }

        # ── Gradient clipping (prevent exploding gradients) ───────────────
        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads

    # ──────────────────────────────────────────────────────────────────────
    # Parameter update  (Adagrad)
    # ──────────────────────────────────────────────────────────────────────

    def update(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply Adagrad updates to all parameters."""
        params = self._params()
        for k in params:
            self._mem[k] += grads[k] ** 2
            params[k]   -= self.lr * grads[k] / (np.sqrt(self._mem[k]) + 1e-8)

    # ──────────────────────────────────────────────────────────────────────
    # Text generation  (sampling)
    # ──────────────────────────────────────────────────────────────────────

    def sample(
        self,
        seed_idx: int,
        h: np.ndarray,
        length: int = 200,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate a sequence of character indices by sampling from the model.

        Parameters
        ----------
        seed_idx    : int — index of the seed character
        h           : (hidden_size, 1) — initial hidden state
        length      : int — how many characters to generate
        temperature : float — <1 sharper, >1 more random
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1.0
        indices = []

        for _ in range(length):
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y
            y = y / temperature

            p = softmax(y).ravel()
            idx = np.random.choice(self.vocab_size, p=p)

            # Prepare next input
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0
            indices.append(idx)

        return indices

    # ──────────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        data: str,
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
        epochs: int = 3,
        print_every: int = 500,
        sample_every: int = 2000,
        sample_length: int = 200,
    ) -> List[float]:
        """
        Train the RNN on a character-level corpus.

        Parameters
        ----------
        data         : str — full training text
        char_to_idx  : dict mapping characters → integer indices
        idx_to_char  : dict mapping integer indices → characters
        epochs       : int — number of full passes over the data
        print_every  : int — print loss every N steps
        sample_every : int — generate sample text every N steps
        sample_length: int — length of generated samples

        Returns
        -------
        loss_history : list of smoothed loss values
        """
        data_size = len(data)
        loss_history: List[float] = []
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # initial guess

        print("=" * 70)
        print(f"  Vanilla RNN  |  vocab={self.vocab_size}  hidden={self.hidden_size}"
              f"  seq_len={self.seq_length}  lr={self.lr}")
        print(f"  Corpus size : {data_size:,} characters")
        print("=" * 70)

        step = 0
        for epoch in range(1, epochs + 1):
            h_prev = np.zeros((self.hidden_size, 1))  # reset hidden state each epoch
            pointer = 0

            print(f"\n-- Epoch {epoch}/{epochs} {'-' * 50}")

            while pointer + self.seq_length + 1 < data_size:
                # ── Prepare mini-batch ────────────────────────────────────
                inputs  = [char_to_idx[ch] for ch in data[pointer : pointer + self.seq_length]]
                targets = [char_to_idx[ch] for ch in data[pointer + 1 : pointer + self.seq_length + 1]]

                # ── Forward → Backward → Update ──────────────────────────
                loss, cache, h_prev = self.forward(inputs, targets, h_prev)
                grads = self.backward(targets, cache)
                self.update(grads)

                # ── Exponential moving average of loss ────────────────────
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                loss_history.append(smooth_loss)

                # ── Logging ───────────────────────────────────────────────
                if step % print_every == 0:
                    print(f"  step {step:>7,}  |  loss = {smooth_loss:.4f}")

                if step % sample_every == 0:
                    sample_ids = self.sample(inputs[0], h_prev, length=sample_length)
                    sample_text = "".join(idx_to_char[i] for i in sample_ids)
                    print(f"\n  +- Sample (step {step}) {'-' * 40}")
                    for line in sample_text.split("\n"):
                        print(f"  | {line}")
                    print(f"  +{'-' * 58}\n")

                pointer += self.seq_length
                step += 1

        print(f"\n{'=' * 70}")
        print(f"  Training complete — final smoothed loss: {smooth_loss:.4f}")
        print(f"{'=' * 70}")
        return loss_history


# ──────────────────────────────────────────────────────────────────────────────
# Demo: Character-level language model
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Sample corpus ─────────────────────────────────────────────────────
    corpus = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    In the middle of difficulty lies opportunity.
    The only way to do great work is to love what you do.
    Life is what happens when you are busy making other plans.
    The greatest glory in living lies not in never falling,
    but in rising every time we fall.
    It does not matter how slowly you go as long as you do not stop.
    The future belongs to those who believe in the beauty of their dreams.
    Imagination is more important than knowledge.
    Knowledge is limited. Imagination encircles the world.
    Strive not to be a success, but rather to be of value.
    The mind is everything. What you think you become.
    An unexamined life is not worth living.
    The only true wisdom is in knowing you know nothing.
    Happiness is not something ready made. It comes from your own actions.
    In three words I can sum up everything I learned about life: it goes on.
    Do not go where the path may lead, go instead where there is no path
    and leave a trail. The best time to plant a tree was twenty years ago.
    The second best time is now. You miss one hundred percent of the shots
    you do not take. Whether you think you can or think you cannot, you are
    right. I have not failed. I have just found ten thousand ways that will
    not work. Everything you have ever wanted is on the other side of fear.
    """

    # ── Build vocabulary ──────────────────────────────────────────────────
    chars = sorted(set(corpus))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary ({vocab_size} unique characters): {''.join(chars)}\n")

    # ── Instantiate & train ───────────────────────────────────────────────
    rnn = VanillaRNN(
        vocab_size=vocab_size,
        hidden_size=128,
        learning_rate=1e-2,
        seq_length=25,
    )

    loss_history = rnn.train(
        data=corpus,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        epochs=50,
        print_every=200,
        sample_every=1000,
        sample_length=200,
    )

    # ── Final generation ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL GENERATED TEXT")
    print("=" * 70)

    h0 = np.zeros((rnn.hidden_size, 1))
    seed = char_to_idx["T"]

    for temp in [0.5, 0.8, 1.0]:
        ids = rnn.sample(seed, h0, length=300, temperature=temp)
        text = "".join(idx_to_char[i] for i in ids)
        print(f"\n  Temperature = {temp}")
        print(f"  {'-' * 60}")
        for line in text.split("\n"):
            print(f"  {line}")

    # ── Optional: plot loss curve ─────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(loss_history, color="#6C5CE7", linewidth=0.8, alpha=0.85)
        plt.title("Training Loss (Smoothed)", fontsize=14, fontweight="bold")
        plt.xlabel("Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("rnn_loss_curve.png", dpi=150)
        plt.show()
        print("\n  Loss curve saved → rnn_loss_curve.png")
    except ImportError:
        print("\n  (matplotlib not found — skipping loss plot)")


if __name__ == "__main__":
    main()
