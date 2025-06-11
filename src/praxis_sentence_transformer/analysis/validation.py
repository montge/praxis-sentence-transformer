# src/analysis/validation.py

"""
This module provides classes for validating requirement analysis results
against ground truth data and computing validation metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..logger.logger import setup_logging, handle_exception
import logging

logger = setup_logging(__name__, logging.INFO)

@dataclass
class ValidationMetrics:
    """Class to hold validation metrics for requirement analysis"""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray
    
    def __str__(self) -> str:
        return f"""
Validation Metrics:
------------------
True Positives: {self.true_positives}
False Positives: {self.false_positives}
True Negatives: {self.true_negatives}
False Negatives: {self.false_negatives}

Precision: {self.precision:.3f}
Recall: {self.recall:.3f}
F1 Score: {self.f1_score:.3f}
Accuracy: {self.accuracy:.3f}
"""

class RequirementValidator:
    """Class to validate requirement analysis results against ground truth"""
    
    def __init__(self, neo4j_client):
        """Initialize validator with Neo4j client"""
        logger.debug("Initializing RequirementValidator")
        self.neo4j_client = neo4j_client
        logger.debug("RequirementValidator initialized")
        
    @handle_exception
    def get_ground_truth_links(self) -> Set[Tuple[str, str]]:
        """Retrieve ground truth requirement links from Neo4j"""
        logger.debug("Retrieving ground truth links from Neo4j")
        query = """
        MATCH (s:Requirement)-[r:GROUND_TRUTH]->(t:Requirement)
        RETURN s.id as high_level_id, t.id as low_level_id
        """
        try:
            with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                result = session.run(query)
                links = {(r['high_level_id'], r['low_level_id']) for r in result}
                logger.info(f"Retrieved {len(links)} ground truth links")
                logger.debug(f"First few links: {list(links)[:3]}")
                return links
        except Exception as e:
            logger.error(f"Error retrieving ground truth links: {str(e)}", exc_info=True)
            return set()
    
    @handle_exception
    def validate_results(self, analysis_results: List, threshold: float = 0.5) -> ValidationMetrics:
        """
        Validate analysis results against ground truth links
        
        Args:
            analysis_results: List of analysis results containing requirement pairs and scores
            threshold: Minimum score to consider a pair as associated
            
        Returns:
            ValidationMetrics object containing various metrics
        """
        logger.debug(f"Starting validation with threshold {threshold}")
        
        # Get ground truth links
        ground_truth = self.get_ground_truth_links()
        logger.debug(f"Retrieved {len(ground_truth)} ground truth links for validation")
        
        # Convert analysis results to predicted links
        predicted_links = {
            (r.high_level_req_id, r.low_level_req_id) 
            for r in analysis_results 
            if r.association_probability >= threshold
        }
        logger.debug(f"Found {len(predicted_links)} predicted links above threshold {threshold}")
        
        # Calculate metrics
        tp = len(ground_truth.intersection(predicted_links))
        fp = len(predicted_links - ground_truth)
        fn = len(ground_truth - predicted_links)
        logger.debug(f"Initial metrics - TP: {tp}, FP: {fp}, FN: {fn}")
        
        # For TN, we need all possible pairs minus the others
        all_high_level = {pair[0] for pair in ground_truth.union(predicted_links)}
        all_low_level = {pair[1] for pair in ground_truth.union(predicted_links)}
        total_possible = len(all_high_level) * len(all_low_level)
        tn = total_possible - (tp + fp + fn)
        
        # Calculate derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Create confusion matrix
        y_true = []
        y_pred = []
        
        # Add all possible pairs to arrays
        for h in all_high_level:
            for l in all_low_level:
                y_true.append(1 if (h, l) in ground_truth else 0)
                y_pred.append(1 if (h, l) in predicted_links else 0)
                
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics = ValidationMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            confusion_matrix=conf_matrix
        )
        
        logger.info(f"Validation complete - F1 Score: {metrics.f1_score:.3f}")
        return metrics
    
    @handle_exception
    def plot_confusion_matrix(self, metrics: ValidationMetrics, save_path: str = None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics.confusion_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Associated', 'Associated'],
            yticklabels=['Not Associated', 'Associated']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()
        else:
            plt.show()