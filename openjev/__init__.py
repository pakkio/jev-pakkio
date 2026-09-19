"""openjev: one-pass option scoring with a local Gemma model (MLX on Apple silicon, torch elsewhere)."""

__all__ = ["DEFAULT_MODEL", "OptionScore", "OptionScorer"]


def __getattr__(name: str):
    if name in __all__:
        from . import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
