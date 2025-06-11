# MIT License
#
# Copyright (c) 2024-2025 Evan Montgomery-Recht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Praxis Sentence Transformer Library
"""

__version__ = "0.2.0"

# Core functionality
from .logger import setup_logging, handle_exception, DebugTimer

# Import Neo4j operations
from .neo4j_operations import (
    Neo4jClient,
    RequirementsTraceabilityGraph,
    RequirementNode
)

# Import models
from .neo4j_operations.models import (
    Project,
    Document,
    Section,
    Requirement
)

# Loaders
from .loaders.document_loader import DocumentHierarchyLoader
from .loaders.requirements_loader import RequirementsLoader
from .loaders.requirements_preprocessor import RequirementsPreprocessor

# Utils
from .utils.cuda import initialize_cuda, cleanup_cuda, cleanup_resources
from .utils.file_handler import create_results_directory

__all__ = [
    # Core
    'setup_logging',
    'handle_exception',
    'DebugTimer',
    
    # Neo4j
    'Neo4jClient',
    'RequirementsTraceabilityGraph',
    'RequirementNode',
    
    # Models
    'Project',
    'Document',
    'Section',
    'Requirement',
    
    # Loaders
    'DocumentHierarchyLoader',
    'RequirementsLoader',
    'RequirementsPreprocessor',
    
    # Utils
    'initialize_cuda',
    'cleanup_cuda',
    'cleanup_resources',
    'create_results_directory'
] 