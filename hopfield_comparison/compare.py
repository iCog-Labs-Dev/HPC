"""Empirical recall of noisy binary and continuous-valued memories."""
import numpy as np
from hopfield import Hopfield
from storkey_hopfield import StorkeyHopfield
from modern_hopfield import ModernHopfield


def compare_binary():
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


def compare_continuous(n=100, pattern_counts=(5, 50, 500), trials=20,
                       noise_std=0.2, atol=1e-3, seed=0):
    """Compare recall of real-valued memories corrupted by Gaussian noise."""
    rng = np.random.default_rng(seed)
    print(f'\nContinuous recall (n={n}, noise std={noise_std}, '
          f'absolute tolerance={atol}, trials={trials})', flush=True)
    print('patterns | classical | storkey | modern', flush=True)
    for count in pattern_counts:
        ok_c = ok_s = ok_m = 0
        storkey_rejected = False
        for _ in range(trials):
            patterns = rng.uniform(-1.0, 1.0, size=(count, n))
            target = patterns[rng.integers(count)]
            noisy = target + rng.normal(0.0, noise_std, size=n)
            update_seed = int(rng.integers(0, 2**32 - 1))

            c = Hopfield(n)
            c.train(patterns)
            np.random.seed(update_seed)
            ok_c += np.allclose(c.recall(noisy), target, atol=atol, rtol=0)

            s = StorkeyHopfield(n)
            try:
                s.train(patterns)
            except ValueError:
                # The generated array has the required shape but non-binary values.
                storkey_rejected = True
            else:
                np.random.seed(update_seed)
                ok_s += np.allclose(s.recall(noisy), target, atol=atol, rtol=0)

            m = ModernHopfield(beta=4.0)
            m.store(patterns)
            ok_m += np.allclose(m.recall(noisy), target, atol=atol, rtol=0)

        storkey_result = 'N/A' if storkey_rejected else f'{ok_s/trials:.2f}'
        print(f'{count:8d} | {ok_c/trials:9.2f} | '
              f'{storkey_result:>7} | {ok_m/trials:6.2f}', flush=True)

    print('N/A = Storkey training rejected the continuous-valued memories.', flush=True)


def main():
    compare_binary()
    compare_continuous()


if __name__ == '__main__':
    main()
