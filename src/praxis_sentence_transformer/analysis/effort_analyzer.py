"""
Module for analyzing impact and effort estimation of requirement analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

from ..logger import setup_logging, handle_exception

logger = setup_logging(__name__)

@dataclass
class CostAnalysis:
    """Cost analysis parameters"""
    review_cost_per_hour: float
    engineering_cost_per_hour: float
    rework_hours: float
    integration_cost: float
    delay_cost_per_day: float
    typical_delay_days: float
    quick_review_minutes: float
    detailed_review_hours: float
    fp_review_hours: float
    fp_overhead_cost: float

class EffortAnalyzer:
    """Analyzes complexity and effort estimation for requirements"""
    
    def __init__(self, model: RandomForestClassifier, neo4j_client=None):
        """
        Initialize effort analyzer
        
        Args:
            model: Trained RandomForestClassifier model
            neo4j_client: Optional Neo4j client for retrieving requirement texts
        """
        self.model = model
        self.neo4j_client = neo4j_client
        
    @handle_exception
    def analyze_requirement_complexity(self, df_analysis: pd.DataFrame, category: str) -> Dict:
        """
        Analyze complexity metrics for requirements in different prediction categories
        
        Args:
            df_analysis: DataFrame containing analysis results
            category: Category being analyzed (TP, FP, FN, TN)
            
        Returns:
            Dict containing complexity metrics
        """
        try:
            # Calculate text length statistics
            text_lengths = {
                'source': df_analysis['source_text'].str.len().mean() if 'source_text' in df_analysis else 0,
                'target': df_analysis['target_text'].str.len().mean() if 'target_text' in df_analysis else 0
            }
            
            # Calculate average similarity scores
            similarity_scores = {}
            for col in df_analysis.columns:
                if col.startswith('Model'):
                    similarity_scores[col.lower()] = df_analysis[col].mean()
            
            # Calculate complexity metrics
            metrics = {
                'category': category,
                'sample_count': len(df_analysis),
                'avg_text_length': text_lengths,
                'avg_similarity_scores': similarity_scores,
                'avg_probability': df_analysis['Probability'].mean() if 'Probability' in df_analysis else 0
            }
            
            logger.info(f"\nComplexity Analysis for {category}:")
            logger.info(f"Sample Count: {metrics['sample_count']}")
            logger.info(f"Average Text Length - Source: {text_lengths['source']:.1f}, Target: {text_lengths['target']:.1f}")
            logger.info(f"Average Probability: {metrics['avg_probability']:.3f}")
            
            for model, score in similarity_scores.items():
                logger.info(f"Average {model} Score: {score:.3f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing complexity for {category}: {str(e)}")
            raise
            
    @handle_exception
    def analyze_effort_impact(self, tp_analysis: pd.DataFrame, fp_analysis: pd.DataFrame, 
                            fn_analysis: pd.DataFrame, cost_analysis: CostAnalysis) -> Dict:
        """
        Analyze impact on project effort estimation
        
        Args:
            tp_analysis: True positive analysis results
            fp_analysis: False positive analysis results  
            fn_analysis: False negative analysis results
            cost_analysis: Cost analysis parameters
            
        Returns:
            Dict containing effort impact analysis
        """
        try:
            # Analyze each category
            tp_metrics = self.analyze_requirement_complexity(tp_analysis, "True Positives")
            fp_metrics = self.analyze_requirement_complexity(fp_analysis, "False Positives")
            fn_metrics = self.analyze_requirement_complexity(fn_analysis, "False Negatives")
            
            # Calculate cost impacts
            cost_impacts = {
                'rework_cost': fn_metrics['sample_count'] * cost_analysis.rework_hours * cost_analysis.engineering_cost_per_hour,
                'integration_cost': fn_metrics['sample_count'] * cost_analysis.integration_cost,
                'delay_cost': fn_metrics['sample_count'] * cost_analysis.delay_cost_per_day * cost_analysis.typical_delay_days,
                'review_cost': (
                    (tp_metrics['sample_count'] + fp_metrics['sample_count']) * 
                    cost_analysis.detailed_review_hours * cost_analysis.review_cost_per_hour
                ),
                'fp_overhead': fp_metrics['sample_count'] * cost_analysis.fp_overhead_cost
            }
            
            total_impact = sum(cost_impacts.values())
            
            results = {
                'metrics': {
                    'true_positives': tp_metrics,
                    'false_positives': fp_metrics,
                    'false_negatives': fn_metrics
                },
                'cost_impacts': cost_impacts,
                'total_impact': total_impact
            }
            
            # Log results
            logger.info("\nEffort Impact Analysis:")
            logger.info(f"Total Cost Impact: ${total_impact:,.2f}")
            for impact, cost in cost_impacts.items():
                logger.info(f"{impact.replace('_', ' ').title()}: ${cost:,.2f}")
                
            return results
            
        except Exception as e:
            logger.error("Error analyzing effort impact: {str(e)}")
            raise 