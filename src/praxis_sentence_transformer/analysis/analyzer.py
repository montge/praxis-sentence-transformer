# src/analysis/analyzer.py

"""
This module provides the RequirementsAnalyzer class for analyzing requirements
and managing their relationships using sentence transformers and Neo4j.
"""

import logging
from typing import List, Tuple, Dict
from ..neo4j_operations.neo4j_client import Neo4jClient
from ..logger.logger import setup_logging, handle_exception

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)
import numpy as np

logger = setup_logging(__name__, logging.INFO)

class RequirementsAnalyzer:
    """
    Analyzes requirements and manages their relationships between source and target requirements
    using sentence transformers, LLMs, and Neo4j.
    """
    
    def __init__(self, 
                 neo4j_client: Neo4jClient, 
                 sentence_transformer_model_name: str = None,
                 llm_model_name: str = None):
        """
        Initialize requirements analyzer
        
        Parameters:
            neo4j_client: Neo4j client instance
            sentence_transformer_model_name: Name of the sentence transformer model (optional)
            llm_model_name: Name of the LLM model (e.g. Claude) for additional analysis (optional)
        """
        logger.debug(f"Initializing RequirementsAnalyzer with sentence_transformer={sentence_transformer_model_name}, "
                    f"llm_model={llm_model_name}")
        
        self.neo4j_client = neo4j_client
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.llm_model_name = llm_model_name
        logger.info("RequirementsAnalyzer initialized successfully")

    @handle_exception
    def analyze_requirements(self, results_dir: str) -> Dict[str, float]:
        """
        Analyze requirements and store results
        
        Parameters:
            results_dir: Directory to store results
            
        Returns:
            Dict containing analysis statistics
        """
        logger.debug(f"Starting requirements analysis, results will be stored in {results_dir}")
        try:
            # Get source and target requirements from Neo4j
            logger.debug("Fetching source requirement IDs from Neo4j")
            source_ids = self.neo4j_client.get_source_requirement_ids()
            logger.info(f"Found {len(source_ids)} source requirements")

            logger.debug("Fetching target requirement IDs from Neo4j")
            target_ids = self.neo4j_client.get_target_requirement_ids()
            logger.info(f"Found {len(target_ids)} target requirements")
            
            # Process requirements in batches
            all_matches = {}
            logger.debug("Processing requirements in batches")
            for source_id in source_ids:
                logger.debug(f"Processing source requirement: {source_id}")
                matches = self.neo4j_client.get_requirement_pairs(
                    source_id=source_id,
                    model_name=self.sentence_transformer_model_name,
                    alpha=self.alpha,
                    threshold=self.threshold
                )
                if matches:
                    logger.debug(f"Found {len(matches)} matches for source {source_id}")
                    all_matches[source_id] = matches
            
            # Calculate statistics
            logger.debug("Calculating analysis statistics")
            stats = self._calculate_statistics(source_ids, all_matches)
            logger.debug("Saving analysis results")
            self._save_results(all_matches, stats, results_dir)
            
            logger.info(f"Analysis complete. Processed {len(all_matches)} requirements with matches")
            return stats
            
        except Exception as e:
            logger.error(f"Error in analyzing requirements: {str(e)}", exc_info=True)
            raise

    def get_true_positive_count(self, model_name: str, sentence_transformer_model_name: str = None) -> int:
        """Get count of true positive matches"""
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[llm:LLM_REQUIREMENT_TRACE]->(target:Requirement {type: 'TARGET'}),
              (source)-[gt:GROUND_TRUTH]->(target)
        WHERE llm.llm_model_name = $model_name
          AND llm.is_associated = true
          AND (llm.model_name IS NULL OR llm.model_name = $sentence_transformer_model_name)
        RETURN COUNT(*) as count
        """
        logger.debug(f"Executing TP count query with params: model_name={model_name}, st_model={sentence_transformer_model_name}")
        logger.debug(f"Query: {query}")
        
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, 
                             model_name=model_name,
                             sentence_transformer_model_name=sentence_transformer_model_name).single()["count"]
            logger.debug(f"TP count query returned: {result}")
            return result

    def get_false_positive_count(self, model_name: str, sentence_transformer_model_name: str = None) -> int:
        """Get count of false positive matches"""
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[llm:LLM_REQUIREMENT_TRACE]->(target:Requirement {type: 'TARGET'})
        WHERE llm.llm_model_name = $model_name
          AND llm.is_associated = true
          AND NOT EXISTS((source)-[:GROUND_TRUTH]->(target))
          AND (llm.model_name IS NULL OR llm.model_name = $sentence_transformer_model_name)
        RETURN COUNT(*) as count
        """
        logger.debug(f"Executing FP count query with params: model_name={model_name}, st_model={sentence_transformer_model_name}")
        logger.debug(f"Query: {query}")
        
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, 
                             model_name=model_name,
                             sentence_transformer_model_name=sentence_transformer_model_name).single()["count"]
            logger.debug(f"FP count query returned: {result}")
            return result

    def get_false_negative_count(self, model_name: str, sentence_transformer_model_name: str = None) -> int:
        """Get count of false negative matches"""
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[gt:GROUND_TRUTH]->(target:Requirement {type: 'TARGET'})
        WHERE NOT EXISTS {
            MATCH (source)-[llm:LLM_REQUIREMENT_TRACE]->(target)
            WHERE llm.llm_model_name = $model_name
            AND llm.is_associated = true
            AND (llm.model_name IS NULL OR llm.model_name = $sentence_transformer_model_name)
        }
        RETURN COUNT(*) as count
        """
        logger.debug(f"Executing FN count query with params: model_name={model_name}, st_model={sentence_transformer_model_name}")
        logger.debug(f"Query: {query}")
        
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, 
                             model_name=model_name,
                             sentence_transformer_model_name=sentence_transformer_model_name).single()["count"]
            logger.debug(f"FN count query returned: {result}")
            return result

    def get_true_negative_count(self, model_name: str, sentence_transformer_model_name: str = None) -> int:
        """Get count of true negative matches"""
        query = """
        MATCH (source:Requirement {type: 'SOURCE'}), (target:Requirement {type: 'TARGET'})
        WHERE NOT EXISTS {
            MATCH (source)-[llm:LLM_REQUIREMENT_TRACE]->(target)
            WHERE llm.llm_model_name = $model_name
            AND llm.is_associated = true
            AND (llm.model_name IS NULL OR llm.model_name = $sentence_transformer_model_name)
        }
        AND NOT EXISTS((source)-[:GROUND_TRUTH]->(target))
        RETURN COUNT(*) as count
        """
        logger.debug(f"Executing TN count query with params: model_name={model_name}, st_model={sentence_transformer_model_name}")
        logger.debug(f"Query: {query}")
        
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, 
                             model_name=model_name,
                             sentence_transformer_model_name=sentence_transformer_model_name).single()["count"]
            logger.debug(f"TN count query returned: {result}")
            return result

    @handle_exception
    def calculate_metrics(self, llm_model_name: str = None, sentence_transformer_model_name: str = None) -> Dict[str, float]:
        """
        Calculate statistical metrics comparing LLM model predictions to ground truth.
        
        Parameters:
            llm_model_name: Name of the LLM model to analyze. If None, uses the model specified during initialization.
            sentence_transformer_model_name: Name of the sentence transformer model (optional)
            
        Returns:
            Dict containing metrics including:
            - true_positives: Count of true positive matches
            - false_positives: Count of false positive matches
            - true_negatives: Count of true negative matches
            - false_negatives: Count of false negative matches
            - precision: TP/(TP+FP)
            - recall: TP/(TP+FN)
            - specificity: TN/(TN+FP)
            - accuracy: (TP+TN)/(TP+TN+FP+FN)
            - balanced_accuracy: (recall + specificity)/2
            - f1_score: 2*(precision*recall)/(precision+recall)
        """
        model_name = llm_model_name or self.llm_model_name
        if not model_name:
            raise ValueError("No LLM model name provided for metrics calculation")
        
        st_model = sentence_transformer_model_name or self.sentence_transformer_model_name
            
        logger.debug(f"Calculating metrics for LLM model: {model_name}, ST: {st_model}")
        
        try:
            # Get counts using separate methods with all parameters
            tp = self.get_true_positive_count(model_name, st_model)
            fp = self.get_false_positive_count(model_name, st_model)
            fn = self.get_false_negative_count(model_name, st_model)
            tn = self.get_true_negative_count(model_name, st_model)

            # Calculate metrics
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            balanced_accuracy = (recall + specificity) / 2

            metrics = {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'f1_score': f1
            }
            
            logger.info(f"Calculated metrics for {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix in the format:
        [
            [TP, FP],
            [FN, TN]
        ]
        Returns:
            2x2 numpy array with confusion matrix values
        """
        metrics = self.calculate_metrics()
        
        # Create confusion matrix with TP in upper left
        conf_matrix = np.array([
            [metrics['true_positives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_negatives']]
        ])
        
        return conf_matrix

    def _calculate_statistics(self, source_ids: List[str], matches: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        """Calculate analysis statistics"""
        total_source = len(source_ids)
        total_matches = sum(len(m) for m in matches.values())
        requirements_with_matches = len(matches)
        
        return {
            'total_source_requirements': total_source,
            'total_matches': total_matches,
            'requirements_with_matches': requirements_with_matches,
            'average_matches_per_requirement': total_matches / total_source if total_source > 0 else 0,
            'coverage_percentage': (requirements_with_matches / total_source * 100) if total_source > 0 else 0
        }

    def _save_results(self, matches: Dict[str, Dict], stats: Dict[str, float], results_dir: str):
        """Save analysis results"""
        import json
        from pathlib import Path
        
        # Create results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save matches
        matches_file = results_path / f"matches_{self.sentence_transformer_model_name.split('/')[-1]}_a{self.alpha}_t{self.threshold}.json"
        with open(matches_file, 'w') as f:
            json.dump(matches, f, indent=2)
            
        # Save statistics
        stats_file = results_path / f"stats_{self.sentence_transformer_model_name.split('/')[-1]}_a{self.alpha}_t{self.threshold}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Results saved to {results_dir}")

    def get_true_positive_details(self, model_name: str, sentence_transformer_model_name: str = None) -> List[Dict]:
        """Get detailed information about true positive matches"""
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[llm:LLM_REQUIREMENT_TRACE]->(target:Requirement {type: 'TARGET'}),
              (source)-[gt:GROUND_TRUTH]->(target)
        WHERE llm.llm_model_name = $model_name
          AND llm.is_associated = true
          AND (llm.model_name IS NULL OR llm.model_name = $sentence_transformer_model_name)
        RETURN source.id as source_id,
               target.id as target_id,
               llm.association_probability as llm_confidence,
               gt.confidence as ground_truth_confidence
        """
        logger.debug(f"Executing TP details query with params: model_name={model_name}, st_model={sentence_transformer_model_name}")
        logger.debug(f"Query: {query}")
        
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = [dict(record) for record in session.run(query, 
                                                          model_name=model_name,
                                                          sentence_transformer_model_name=sentence_transformer_model_name)]
            logger.debug(f"TP details query returned {len(result)} records")
            return result

    def get_false_positive_details(self, model_name: str) -> List[Dict]:
        """
        Get detailed information about false positive matches.
        
        Parameters:
            model_name: Name of the LLM model to analyze
            
        Returns:
            List of dictionaries containing:
            - source_id: ID of source requirement
            - target_id: ID of target requirement
            - llm_confidence: Confidence score from LLM
        """
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[llm:LLM_REQUIREMENT_TRACE]->(target:Requirement {type: 'TARGET'})
        WHERE llm.llm_model_name = $model_name
          AND llm.is_associated = true
          AND NOT EXISTS((source)-[:GROUND_TRUTH]->(target))
        RETURN source.id as source_id, 
               target.id as target_id, 
               llm.association_probability as llm_confidence
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, model_name=model_name)
            return [dict(record) for record in result]

    def get_false_negative_details(self, model_name: str) -> List[Dict]:
        """
        Get detailed information about false negative matches.
        
        Parameters:
            model_name: Name of the LLM model to analyze
            
        Returns:
            List of dictionaries containing:
            - source_id: ID of source requirement
            - target_id: ID of target requirement
            - ground_truth_confidence: Confidence score from ground truth
        """
        query = """
        MATCH (source:Requirement {type: 'SOURCE'})-[gt:GROUND_TRUTH]->(target:Requirement {type: 'TARGET'})
        WHERE NOT EXISTS {
            MATCH (source)-[llm:LLM_REQUIREMENT_TRACE]->(target)
            WHERE llm.llm_model_name = $model_name
            AND llm.is_associated = true
        }
        RETURN source.id as source_id, 
               target.id as target_id, 
               gt.confidence as ground_truth_confidence
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, model_name=model_name)
            return [dict(record) for record in result]

    def get_true_negative_details(self, model_name: str) -> List[Dict]:
        """
        Get detailed information about true negative matches.
        
        Parameters:
            model_name: Name of the LLM model to analyze
            
        Returns:
            List of dictionaries containing:
            - source_id: ID of source requirement
            - target_id: ID of target requirement
        """
        query = """
        MATCH (source:Requirement {type: 'SOURCE'}), (target:Requirement {type: 'TARGET'})
        WHERE NOT EXISTS {
            MATCH (source)-[llm:LLM_REQUIREMENT_TRACE]->(target)
            WHERE llm.llm_model_name = $model_name
            AND llm.is_associated = true
        }
        AND NOT EXISTS((source)-[:GROUND_TRUTH]->(target))
        RETURN source.id as source_id, 
               target.id as target_id
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run(query, model_name=model_name)
            return [dict(record) for record in result]

    def get_all_classification_details(self, llm_model_name: str = None) -> Dict[str, List[Dict]]:
        """
        Get detailed information about all classifications (TP, FP, FN, TN).
        
        Parameters:
            llm_model_name: Name of the LLM model to analyze. If None, uses the model specified during initialization.
            
        Returns:
            Dictionary containing lists of details for each classification type:
            - true_positives: List of TP details
            - false_positives: List of FP details
            - false_negatives: List of FN details
            - true_negatives: List of TN details
        """
        model_name = llm_model_name or self.llm_model_name
        if not model_name:
            raise ValueError("No LLM model name provided for getting classification details")
            
        return {
            'true_positives': self.get_true_positive_details(model_name),
            'false_positives': self.get_false_positive_details(model_name),
            'false_negatives': self.get_false_negative_details(model_name),
            'true_negatives': self.get_true_negative_details(model_name)
        }