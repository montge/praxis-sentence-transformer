# src/analysis/manager.py

"""
This module provides the RequirementsManager class for managing requirements
and their relationships in Neo4j.
"""

import logging
from typing import List, Tuple, Dict
from ..neo4j_operations.neo4j_client import Neo4jClient
from ..logger.logger import setup_logging, handle_exception

logger = setup_logging(__name__, logging.INFO)

class RequirementsManager:
    """Manages requirements and their relationships"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize requirements manager
        
        Parameters:
            neo4j_client: Initialized Neo4jClient instance
        """
        logger.debug("Initializing RequirementsManager")
        self.neo4j_client = neo4j_client
        logger.info("RequirementsManager initialized successfully")

    @handle_exception
    def store_analysis_results(self, 
                             source_requirements: List[Tuple[str, str]],
                             target_requirements: List[Tuple[str, str]],
                             similarities: Dict[str, List[Tuple[str, float]]]):
        """
        Store requirements and their similarities in Neo4j
        
        Parameters:
            source_requirements: List of (id, content) tuples for source requirements
            target_requirements: List of (id, content) tuples for target requirements
            similarities: Dictionary mapping source IDs to list of (target_id, similarity) tuples
        """
        logger.debug("Starting to store analysis results in Neo4j")
        logger.debug(f"Processing {len(source_requirements)} source requirements")
        logger.debug(f"Processing {len(target_requirements)} target requirements")
        logger.debug(f"Processing similarities for {len(similarities)} source requirements")

        try:
            # Store requirements
            logger.debug("Storing source requirements")
            self.neo4j_client.store_requirements(source_requirements, "source")
            logger.debug("Storing target requirements")
            self.neo4j_client.store_requirements(target_requirements, "target")
            
            # Store similarities
            logger.debug("Storing similarity relationships")
            self.neo4j_client.store_similarities(similarities)
            
            logger.info("Successfully stored all analysis results in Neo4j")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}", exc_info=True)
            raise

    def get_similar_requirements(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Get requirements pairs above similarity threshold
        
        Parameters:
            threshold: Minimum similarity threshold
            
        Returns:
            List of (source_id, target_id, similarity) tuples
        """
        return self.neo4j_client.get_requirement_pairs(threshold)