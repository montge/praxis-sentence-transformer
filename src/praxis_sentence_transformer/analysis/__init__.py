"""
This module initializes the analysis package which contains components for
analyzing requirements and their relationships.

Available Classes:
    - RequirementsAnalyzer: Handles the analysis and processing of requirements
    - RequirementsLoader: Handles loading and parsing requirements from XML files
    - RequirementsManager: Manages requirements and their relationships in Neo4j
    - RequirementValidator: Validates requirement analysis results against ground truth
    - ValidationMetrics: Data class for holding validation metrics
    - SimilarityModelTrainer: Handles training and evaluation of similarity models
    - FeatureImportanceAnalyzer: Analyzes feature importance in similarity models
    - ErrorAnalyzer: Analyzes error cases in similarity predictions
    - EffortAnalyzer: Analyzes impact and effort estimation of requirement analysis
    - CostAnalysis: Data class for holding cost analysis parameters
"""

from .analyzer import RequirementsAnalyzer
from .manager import RequirementsManager
from .validation import RequirementValidator, ValidationMetrics
from .model_trainer import SimilarityModelTrainer
from .feature_analyzer import FeatureImportanceAnalyzer
from .error_analysis import ErrorAnalyzer
from .effort_analyzer import EffortAnalyzer, CostAnalysis

__all__ = [
    'RequirementsAnalyzer',
    'RequirementsManager',
    'RequirementValidator',
    'ValidationMetrics',
    'SimilarityModelTrainer',
    'FeatureImportanceAnalyzer',
    'ErrorAnalyzer',
    'EffortAnalyzer',
    'CostAnalysis'
] 