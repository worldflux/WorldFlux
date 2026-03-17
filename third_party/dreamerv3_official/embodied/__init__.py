__version__ = "2.0.0"

try:
    import colored_traceback

    colored_traceback.add_hook(colors="terminal")
except ImportError:
    pass

from . import envs, jax, run
from .core import *
