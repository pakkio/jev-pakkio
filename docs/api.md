# API reference

## Cross-platform inference

`OptionScorer` accepts `backend="auto"` and `device="auto"`. Automatic backend
selection uses MLX on Apple silicon and PyTorch elsewhere. For PyTorch, automatic
device selection prefers CUDA, then MPS, then CPU.

```python
from jev_pakkio import OptionScorer

scorer = OptionScorer("google/gemma-3-4b-it", backend="torch", device="cuda", batch_size=2)
results = scorer.score("The capital of France is", [" Paris", " Berlin"])
```

Use `device="cpu"` for CPU inference. The server uses the same options:
`uv run jev_pakkio serve --backend torch --device auto`. Both `/score` and
`/v1/systemone` retain the same request and response contracts across backends.
See [Getting started](getting-started.md) for GPU installation instructions.


The `jev_pakkio` package. `OptionScorer` is re-exported from the top level:

```python
from jev_pakkio import OptionScorer
```

## `jev_pakkio`

::: jev_pakkio

## `jev_pakkio.scorer`

::: jev_pakkio.scorer

## `jev_pakkio.systemone`

::: jev_pakkio.systemone

## `jev_pakkio.server`

::: jev_pakkio.server

## `jev_pakkio.features`

::: jev_pakkio.features

## `jev_pakkio.head`

::: jev_pakkio.head

## `jev_pakkio.train`

::: jev_pakkio.train

## `jev_pakkio.cli`

::: jev_pakkio.cli
