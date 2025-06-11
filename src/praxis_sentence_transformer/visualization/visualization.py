"""
This module provides visualization functions for requirements analysis results,
including correlation matrices and statistical measure plots.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..logger.logger import setup_logging, handle_exception

logger = setup_logging(__name__, logging.INFO)

class RequirementsVisualizer:
    """
    Handles visualization of requirements analysis results including
    correlation matrices and statistical measures.
    """
    
    def __init__(self, 
                 figsize: tuple = (10, 6), 
                 st_model: Optional[str] = None, 
                 llm_model: Optional[str] = None):
        """
        Initialize visualizer
        
        Parameters:
            figsize: Default figure size for plots (width, height)
            st_model: Name of sentence transformer model being used
            llm_model: Name of LLM model being used
        """
        self.figsize = figsize
        self.st_model = st_model
        self.llm_model = llm_model
        logger.info("RequirementsVisualizer initialized")

    def _add_model_info_to_title(self, title: str) -> str:
        """Helper method to add model information to plot titles"""
        full_title = title
        if self.st_model:
            full_title += f"\nSentence Transformer: {self.st_model}"
        if self.llm_model:
            full_title += f"\nLLM Model: {self.llm_model}"
        return full_title

    @handle_exception
    def plot_correlation_matrix(self, 
                              correlation_data: Dict[str, float],
                              title: str = "Correlation Matrix",
                              save_path: str = None,
                              max_size: int = 20) -> None:
        """
        Plot correlation matrix heatmap with size limits
        
        Parameters:
            correlation_data: Dictionary of correlation values
            title: Plot title
            save_path: Path to save the plot
            max_size: Maximum number of items to show in correlation matrix
        """
        logger.debug("Creating correlation matrix plot")
        try:
            # Convert to DataFrame and limit size
            df = pd.DataFrame(correlation_data)
            if len(df) > max_size:
                logger.warning(f"Correlation matrix too large ({len(df)} items), showing top {max_size}")
                # Select top correlations based on absolute values
                df = df.iloc[:max_size, :max_size]
            
            plt.figure(figsize=self.figsize)
            sns.heatmap(df, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       vmin=-1, 
                       vmax=1,
                       fmt='.2f')
            
            plt.title(self._add_model_info_to_title(title))
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved correlation matrix plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix plot: {str(e)}", exc_info=True)
            raise

    @handle_exception
    def plot_statistical_measures(self,
                                metrics: Dict[str, float],
                                title: str = "Statistical Measures",
                                save_path: str = None) -> None:
        """Create bar plot of statistical measures"""
        logger.debug("Creating statistical measures plot")
        try:
            plt.figure(figsize=self.figsize)
            measures = list(metrics.keys())
            values = list(metrics.values())
            bars = plt.bar(measures, values)
            
            plt.title(self._add_model_info_to_title(title))
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.1)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved statistical measures plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating statistical measures plot: {str(e)}", exc_info=True)
            raise

    def plot_confusion_matrix(self,
                            conf_matrix: np.ndarray,
                            title: str = "Confusion Matrix",
                            save_path: str = None,
                            dpi: int = 300) -> None:
        """
        Plot confusion matrix
        
        Parameters:
            conf_matrix: 2x2 numpy array containing [[tp, fp], [fn, tn]]
            title: Plot title
            save_path: Path to save the plot
            dpi: Dots per inch for the output image (default: 300)
        """
        logger.debug("Creating confusion matrix plot")
        try:
            plt.figure(figsize=self.figsize)
            sns.heatmap(conf_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'])
            
            plt.title(self._add_model_info_to_title(title))
            plt.ylabel('Predicted Label')
            plt.xlabel('True Label')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved confusion matrix plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {str(e)}", exc_info=True)
            raise
