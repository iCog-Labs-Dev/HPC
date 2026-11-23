# Hopfield Networks

A NumPy comparison of classical Hopfield networks with Hebbian and Storkey
learning rules and a continuous modern Hopfield network with softmax retrieval.
The benchmark measures how well each model recovers stored patterns from noisy
inputs.

## Quick start

From the repository root, install NumPy and run the comparison.

```sh
python -m pip install numpy
python hopfield_comparison/compare.py
```

The script runs both binary and continuous-valued recall experiments. The
largest binary runs can take a while.

## Tests

```sh
python -m pip install pytest
python -m pytest hopfield_comparison/tests.py
```

## More details

See the [technical comparison in Google Docs](https://docs.google.com/document/d/1t_cAGRlsZNw9Y6JU7wS0tE-804rcFW5KG4Dz0dZI8Cg/edit?usp=sharing) for the model definitions, equations, experiment setup, and results.
