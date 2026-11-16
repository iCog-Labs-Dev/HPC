"""Empirical recall success at 20% bit corruption, not a capacity proof."""
import numpy as np
from hopfield import Hopfield
from storkey_hopfield import StorkeyHopfield
from modern_hopfield import ModernHopfield


def main():
    # Edit these settings directly; no command-line arguments are needed.
    n = 100
    flip = 20
    trials = 20
    seed = 0
    pattern_counts = [5, 10, 14, 20, 50, 100, 200, 500,
                      1000, 10000]

    rng = np.random.default_rng(seed)
    print(f'n={n}, flipped bits={flip}, trials={trials}', flush=True)
    print('patterns | classical | storkey | modern', flush=True)
    for count in pattern_counts:
        ok_c = ok_s = ok_m = 0
        for _ in range(trials):
            patterns = rng.choice([-1, 1], size=(count, n))
            # Storkey is sequential: vary the tested memory's position in the list.
            target_index = rng.integers(count)
            target = patterns[target_index]
            noisy = target.copy()
            noisy[rng.choice(n, size=flip, replace=False)] *= -1
            update_seed = int(rng.integers(0, 2**32 - 1))

            c = Hopfield(n)
            c.train(patterns)
            np.random.seed(update_seed)
            ok_c += np.array_equal(c.recall(noisy), target)

            s = StorkeyHopfield(n)
            s.train(patterns)
            # Same update order and sweeps as classical.
            np.random.seed(update_seed)
            ok_s += np.array_equal(s.recall(noisy), target)

            m = ModernHopfield(beta=4.0)
            m.store(patterns)
            ok_m += np.array_equal(np.sign(m.recall(noisy)), target)

        print(f'{count:8d} | {ok_c/trials:9.2f} | '
              f'{ok_s/trials:7.2f} | {ok_m/trials:6.2f}', flush=True)


if __name__ == '__main__':
    main()
