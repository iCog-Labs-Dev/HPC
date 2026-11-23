"""Train the reusable predictive-coding network on MNIST."""

import argparse

from numpy_predictive_coding import XavierInitializer, evaluate, load_mnist, pc_model


def main(argv=None):
    """Parse training options, load MNIST, train, and report held-out metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", help="Existing MNIST .npz archive or download destination")
    parser.add_argument("--h1", type=int, default=256)
    parser.add_argument("--h2", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20, help="Inference steps per batch")
    parser.add_argument("--activity-rate", type=float, default=0.05)
    parser.add_argument("--weight-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--distribution", choices=("normal", "uniform"), default="normal")
    parser.add_argument("--gain", type=float, default=1.0)
    args = parser.parse_args(argv)
    X_train, Y_train, X_test, Y_test = load_mnist(args.cache_path)
    parameters, _ = pc_model(
        X_train, Y_train, h1=args.h1, h2=args.h2, epochs=args.epochs,
        batch_size=args.batch_size, steps=args.steps, activity_rate=args.activity_rate,
        weight_rate=args.weight_rate, weight_decay=args.weight_decay, seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        weight_init=XavierInitializer(args.distribution, args.gain),
    )
    loss, accuracy = evaluate(X_test, Y_test, parameters, batch_size=args.batch_size)
    print(f"Test: CE = {loss:.5f}, accuracy = {accuracy:.2%}")


if __name__ == "__main__":
    main()
