"""
Analysis modules for requirement similarity
"""

__all__ = ['SentenceTransformerAnalyzer']

def __getattr__(name):
    if name == 'SentenceTransformerAnalyzer':
        from .sentence_transformer import SentenceTransformerAnalyzer
        return SentenceTransformerAnalyzer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 