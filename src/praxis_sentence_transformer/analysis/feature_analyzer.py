"""
Module for analyzing feature importance in similarity models
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..logger import setup_logging, handle_exception

logger = setup_logging(__name__)

class FeatureImportanceAnalyzer:
    """Analyzes and visualizes feature importance in similarity models"""
    
    def __init__(self, visualization_path: Path):
        """
        Initialize analyzer
        
        Args:
            visualization_path: Path to save visualizations
        """
        self.visualization_path = Path(visualization_path)
        self.visualization_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized FeatureImportanceAnalyzer with path: {self.visualization_path}")

    @handle_exception
    def analyze_feature_importance(self, 
                                 feature_importance: pd.DataFrame,
                                 model_name: str = "Random Forest") -> Dict[str, float]:
        """
        Analyze and visualize feature importance
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            model_name: Name of the model being analyzed
            
        Returns:
            Dict containing analysis results
        """
        logger.info(f"Analyzing feature importance for {model_name}")
        
        try:
            # Sort features by importance
            sorted_features = feature_importance.sort_values('importance', ascending=True)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar plot
            bars = plt.barh(range(len(sorted_features)), sorted_features['importance'])
            
            # Customize plot
            plt.yticks(range(len(sorted_features)), sorted_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance Analysis - {model_name}')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', 
                        ha='left', va='center', fontweight='bold')
            
            # Save plot
            plot_path = self.visualization_path / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Feature importance plot saved to {plot_path}")
            
            # Calculate summary statistics
            results = {
                'top_feature': sorted_features.iloc[-1]['feature'],
                'top_importance': float(sorted_features.iloc[-1]['importance']),
                'mean_importance': float(sorted_features['importance'].mean()),
                'std_importance': float(sorted_features['importance'].std()),
                'feature_count': len(sorted_features)
            }
            
            # Log results
            logger.info("Feature Importance Analysis Results:")
            logger.info(f"Top Feature: {results['top_feature']} ({results['top_importance']:.3f})")
            logger.info(f"Mean Importance: {results['mean_importance']:.3f}")
            logger.info(f"Std Importance: {results['std_importance']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise 