"""
Module for creating and saving model analysis visualizations
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, 
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from typing import Tuple, Dict, Any, Union

from ..logger import setup_logging, handle_exception
from ..utils.file_handler import create_results_directory

logger = setup_logging(__name__)

class ModelVisualizer:
    """Creates and saves visualizations for model analysis"""
    
    def __init__(self, results_dir: Union[str, Path] = None):
        """
        Initialize visualizer
        
        Args:
            results_dir: Directory to save visualizations (optional)
                Can be string or Path object
        """
        # Convert results_dir to Path if it's a string
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = create_results_directory(
                model_name="similarity-analysis",
                dataset_name=os.getenv('PROJECT_NAME', 'default')
            )
        
        # Create visualizations subdirectory
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        logger.debug(f"Visualization directory: {self.viz_dir}")

    @handle_exception
    def create_visualizations(self, 
                            model: Any,
                            feature_importance: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_test: np.ndarray,
                            y_pred: np.ndarray,
                            timestamp: str = None) -> Tuple[plt.Figure, Dict[str, float]]:
        """
        Create and save visualizations for model analysis
        
        Args:
            model: Trained model instance
            feature_importance: DataFrame with feature importance scores
            X_test: Test feature data
            y_test: True test labels
            y_pred: Predicted test labels
            timestamp: Optional timestamp for filenames
            
        Returns:
            Tuple of (figure, metrics_dict)
        """
        project_name = os.getenv('PROJECT_NAME', 'default')
        logger.info(f"Creating visualizations for project: {project_name}")
        
        try:
            # Set up the figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Model Analysis Visualizations - Project: {project_name}', fontsize=16)
            
            # 1. Confusion Matrix Heatmap
            logger.debug("Creating confusion matrix heatmap")
            conf_matrix = self._create_confusion_matrix_plot(y_test, y_pred, ax1)
            
            # 2. Feature Importance Plot
            logger.debug("Creating feature importance plot")
            self._create_feature_importance_plot(feature_importance, ax2)
            
            # 3. ROC Curve
            logger.debug("Creating ROC curve")
            roc_auc = self._create_roc_curve_plot(model, X_test, y_test, ax3)
            
            # 4. Precision-Recall Curve
            logger.debug("Creating precision-recall curve")
            avg_precision = self._create_precision_recall_plot(model, X_test, y_test, ax4)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save plot
            if timestamp:
                plot_path = self.viz_dir / f"model_analysis_{timestamp}.png"
            else:
                plot_path = self.viz_dir / f"model_analysis_{project_name}.png"
                
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to {plot_path}")
            
            metrics = {
                'roc_auc': roc_auc,
                'avg_precision': avg_precision
            }
            
            return fig, metrics
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def _create_confusion_matrix_plot(self, y_test: np.ndarray, y_pred: np.ndarray, ax: plt.Axes) -> np.ndarray:
        """Create confusion matrix heatmap"""
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Not Related', 'Related'],
            yticklabels=['Not Related', 'Related'],
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        return conf_matrix

    def _create_feature_importance_plot(self, feature_importance: pd.DataFrame, ax: plt.Axes):
        """Create feature importance barplot"""
        feature_importance.plot(
            kind='barh',
            x='feature',
            y='importance',
            ax=ax,
            color='skyblue'
        )
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance Score')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _create_roc_curve_plot(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray, ax: plt.Axes) -> float:
        """Create ROC curve plot"""
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return roc_auc

    def _create_precision_recall_plot(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray, ax: plt.Axes) -> float:
        """Create precision-recall curve plot"""
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve\n(AP = {avg_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        return avg_precision 