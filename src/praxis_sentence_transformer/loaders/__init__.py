"""
Loaders package for handling document and requirements loading
"""

__all__ = [
    'Document',
    'Section', 
    'Subsection',
    'Requirement',
    'DocumentHierarchyLoader',
    'RequirementsLoader',
    'RequirementsPreprocessor'
]

def __getattr__(name):
    if name in ['Document', 'Section', 'Subsection', 'Requirement', 'DocumentHierarchyLoader']:
        from .document_loader import Document, Section, Subsection, Requirement, DocumentHierarchyLoader
        return locals()[name]
    elif name == 'RequirementsLoader':
        from .requirements_loader import RequirementsLoader
        return RequirementsLoader
    elif name == 'RequirementsPreprocessor':
        from .requirements_preprocessor import RequirementsPreprocessor
        return RequirementsPreprocessor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 