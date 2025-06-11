"""
Module for training and evaluating similarity models
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    fbeta_score, 
    roc_auc_score, 
    confusion_matrix,
    precision_score,
    recall_score
)

from ..logger import setup_logging, handle_exception

logger = setup_logging(__name__)

class SimilarityModelTrainer:
    """Handles training and evaluation of similarity models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        
    @handle_exception
    def prepare_and_train_model(self, data: pd.DataFrame) -> tuple:
        """
        Prepare balanced dataset and train Random Forest model
        
        Args:
            data: DataFrame containing similarity scores and labels
            
        Returns:
            tuple: (trained model, feature importance DataFrame, X_test, y_test, predictions, X_full, y_full)
        """
        logger.info(f"Starting data preparation and model training")
        
        try:
            # Prepare features and target
            feature_columns = [col for col in data.columns 
                             if col not in ['source_id', 'source_content', 'target_id', 
                                          'target_content', 'is_related', 'model_name']]
            
            # Create full dataset
            X_full = data[feature_columns]
            y_full = data['is_related']
            
            # Prepare balanced dataset
            logger.debug("Preparing balanced dataset")
            positive_samples = data[data['is_related'] == 1]
            negative_samples = data[data['is_related'] == 0].sample(
                n=len(positive_samples), 
                random_state=self.random_state
            )
            balanced_data = pd.concat([positive_samples, negative_samples])
            
            logger.debug(f"Created balanced dataset with {len(balanced_data)} total samples")
            
            # Prepare features and target for balanced data
            X = balanced_data[feature_columns]
            y = balanced_data['is_related']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            logger.debug(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Initialize and train model
            logger.debug("Initializing RandomForestClassifier")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            )
            
            logger.debug("Training RandomForestClassifier")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            logger.debug("Evaluating model performance")
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            classification_rep = classification_report(y_test, y_pred)
            f2 = fbeta_score(y_test, y_pred, beta=2)
            roc_auc = roc_auc_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Log results
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_rep}")
            logger.info(f"F2 Score: {f2:.3f}")
            logger.info(f"ROC AUC Score: {roc_auc:.3f}")
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return (
                self.model,
                self.feature_importance,
                X_test,
                y_test,
                y_pred,
                X_full,
                y_full
            )
            
        except Exception as e:
            logger.error(f"Error in prepare_and_train_model: {str(e)}")
            raise 