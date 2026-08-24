# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Look inside any vision transformer backbone in one command.

    from vit_xray import inspect
    result = inspect("photo.jpg")
    result.artifact_score   # fraction of high-norm outlier tokens
    result.plot()           # matplotlib Figure

See LICENSE for the full MIT licence text.
"""

from vit_xray.defaults import (
    ARTIFACT_MAX_FRACTION,
    ARTIFACT_OVERLAP_GATE,
    DEFAULT_MODEL,
    MODEL_ZOO,
    SHOWCASE_MODEL,
)

__version__ = "0.1.0"
__all__ = [
    "inspect",
    "XrayResult",
    "artifact_score",
    "DEFAULT_MODEL",
    "SHOWCASE_MODEL",
    "MODEL_ZOO",
    "ARTIFACT_OVERLAP_GATE",
    "ARTIFACT_MAX_FRACTION",
    "__version__",
]

# Everything below lives in core.py, which imports torch and timm and takes ~6 seconds.
# Resolve it on first use (PEP 562) so `vit-xray --help` doesn't pay for it.
_LAZY = {"inspect", "XrayResult", "artifact_score"}


def __getattr__(name):
    if name in _LAZY:
        from vit_xray import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
