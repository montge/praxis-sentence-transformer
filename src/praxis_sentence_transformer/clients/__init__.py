"""
Client modules for external API integrations
"""

# Import Claude client when available
def __getattr__(name):
    if name in ['ClaudeClient']:
        from .claude import ClaudeClient
        return ClaudeClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ClaudeClient'
] 