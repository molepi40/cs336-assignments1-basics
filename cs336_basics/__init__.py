import importlib.metadata
from .tokenizer.bpe_tokenizer import BPETokenizer

__version__ = importlib.metadata.version("cs336_basics")

__all__ = [
    'BPETokenizer'
]
