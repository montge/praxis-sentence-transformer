"""
Module for analyzing error cases in similarity predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from ..logger import setup_logging, handle_exception

logger = setup_logging(__name__)

class ErrorAnalyzer:
    """Analyzes error cases in similarity predictions"""
    
    def __init__(self, model: RandomForestClassifier, neo4j_client=None):
        """
        Initialize error analyzer
        
        Args:
            model: Trained RandomForestClassifier model
            neo4j_client: Optional Neo4j client for retrieving requirement texts
        """
        self.model = model
        self.neo4j_client = neo4j_client
        
    @handle_exception
    def analyze_false_positives(self, X_test: pd.DataFrame, y_test: pd.Series, 
                              predictions: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze false positive predictions
        
        Args:
            X_test: Test feature matrix
            y_test: True labels
            predictions: Model predictions
            threshold: Classification threshold
            
        Returns:
            DataFrame containing false positive analysis
        """
        # Get probabilities for positive class
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Find false positive indices
        fp_indices = np.where((predictions == 1) & (y_test == 0))[0]
        
        # Create DataFrame with false positives
        fp_data = pd.DataFrame({
            'Actual': y_test.iloc[fp_indices],
            'Predicted': predictions[fp_indices],
            'Probability': y_prob[fp_indices]
        })
        
        # Add model scores
        for i, col in enumerate(X_test.columns):
            fp_data[f'Model {i+1}'] = X_test.iloc[fp_indices][col]
            
        # Sort by probability
        fp_data = fp_data.sort_values('Probability', ascending=False)
        
        logger.info(f"Found {len(fp_indices)} false positives")
        logger.info(f"Average probability: {fp_data['Probability'].mean():.3f}")
        
        return fp_data
        
    @handle_exception
    def analyze_false_negatives(self, X_test: pd.DataFrame, y_test: pd.Series, 
                              predictions: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze false negative predictions
        
        Args:
            X_test: Test feature matrix
            y_test: True labels
            predictions: Model predictions
            threshold: Classification threshold
            
        Returns:
            DataFrame containing false negative analysis
        """
        # Get probabilities for positive class
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Find false negative indices
        fn_indices = np.where((predictions == 0) & (y_test == 1))[0]
        
        # Create DataFrame with false negatives
        fn_data = pd.DataFrame({
            'Actual': y_test.iloc[fn_indices],
            'Predicted': predictions[fn_indices],
            'Probability': y_prob[fn_indices]
        })
        
        # Add model scores
        for i, col in enumerate(X_test.columns):
            fn_data[f'Model {i+1}'] = X_test.iloc[fn_indices][col]
            
        # Sort by probability
        fn_data = fn_data.sort_values('Probability', ascending=True)
        
        logger.info(f"Found {len(fn_indices)} false negatives")
        logger.info(f"Average probability: {fn_data['Probability'].mean():.3f}")
        
        return fn_data
        
    @handle_exception
    def analyze_true_positives(self, X_test: pd.DataFrame, y_test: pd.Series, 
                             predictions: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze true positive predictions
        
        Args:
            X_test: Test feature matrix
            y_test: True labels
            predictions: Model predictions
            threshold: Classification threshold
            
        Returns:
            DataFrame containing true positive analysis
        """
        # Get probabilities for positive class
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Find true positive indices
        tp_indices = np.where((predictions == 1) & (y_test == 1))[0]
        
        # Create DataFrame with true positives
        tp_data = pd.DataFrame({
            'Actual': y_test.iloc[tp_indices],
            'Predicted': predictions[tp_indices],
            'Probability': y_prob[tp_indices]
        })
        
        # Add model scores
        for i, col in enumerate(X_test.columns):
            tp_data[f'Model {i+1}'] = X_test.iloc[tp_indices][col]
            
        # Sort by probability
        tp_data = tp_data.sort_values('Probability', ascending=False)
        
        logger.info(f"Found {len(tp_indices)} true positives")
        logger.info(f"Average probability: {tp_data['Probability'].mean():.3f}")
        
        return tp_data
        
    @handle_exception
    def analyze_true_negatives(self, X_test: pd.DataFrame, y_test: pd.Series, 
                             predictions: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze true negative predictions
        
        Args:
            X_test: Test feature matrix
            y_test: True labels
            predictions: Model predictions
            threshold: Classification threshold
            
        Returns:
            DataFrame containing true negative analysis
        """
        # Get probabilities for positive class
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Find true negative indices
        tn_indices = np.where((predictions == 0) & (y_test == 0))[0]
        
        # Create DataFrame with true negatives
        tn_data = pd.DataFrame({
            'Actual': y_test.iloc[tn_indices],
            'Predicted': predictions[tn_indices],
            'Probability': y_prob[tn_indices]
        })
        
        # Add model scores
        for i, col in enumerate(X_test.columns):
            tn_data[f'Model {i+1}'] = X_test.iloc[tn_indices][col]
            
        # Sort by probability
        tn_data = tn_data.sort_values('Probability', ascending=True)
        
        logger.info(f"Found {len(tn_indices)} true negatives")
        logger.info(f"Average probability: {tn_data['Probability'].mean():.3f}")
        
        return tn_data 