# src/praxis_sentence_transformer/utils/file_handler.py

from typing import Dict, List, Tuple, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import os
import logging

from ..logger import setup_logging, handle_exception, DebugTimer

# Initialize logger with module name only - level comes from env
logger = setup_logging(__name__)

@handle_exception
def create_results_directory(model_name: str, dataset_name: str) -> Dict[str, str]:
    """
    Creates a directory structure for storing analysis results.
    
    Args:
        model_name (str): Name of the model being used
        dataset_name (str): Name of the dataset being analyzed
        
    Returns:
        Dict[str, str]: Dictionary containing paths to different results directories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    
    base_dir = os.path.join(
        'results', 
        dataset_name, 
        model_short_name, 
        timestamp
    )
    
    comparisons_dir = os.path.join(
        'results', 
        dataset_name, 
        'comparisons', 
        timestamp
    )
    
    paths = {
        'base': base_dir,
        'data': os.path.join(base_dir, 'data'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'results': os.path.join(base_dir, 'results'),
        'comparisons': comparisons_dir,
        'compressed': os.path.join('results', 'compressed_results')
    }
    
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
        
    return paths

class RequirementsFileHandler:
    """Handles file operations for requirements analysis"""
    
    def __init__(self, results_dir: Path = None):
        """
        Initialize file handler
        
        Args:
            results_dir: Optional base directory for results
        """
        self.results_dir = results_dir or Path("results")
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Set up directory structure for results"""
        self.data_dir = self.results_dir / "data"
        self.viz_dir = self.results_dir / "visualizations"
        
        for dir_path in [self.data_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    @handle_exception
    def parse_requirements(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Parse requirements from XML file, focusing on descriptions
        
        Args:
            file_path: Path to XML requirements file
            
        Returns:
            List of requirement tuples (id, description)
            where id is kept only for reference
        """
        logger.debug(f"Parsing requirements from {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        requirements = []
        
        for artifact in root.findall('.//artifact'):
            req_id = artifact.find('id')
            req_desc = artifact.find('content')
            
            if req_id is not None and req_desc is not None:
                desc_text = req_desc.text.strip()
                if desc_text:  # Only add if description is not empty
                    requirements.append((req_id.text.strip(), desc_text))
        
        logger.info(f"Successfully parsed {len(requirements)} requirements")
        return requirements

    @handle_exception
    def save_results(self, results: pd.DataFrame, timestamp: str) -> str:
        """
        Save analysis results with similarity scores
        
        Args:
            results: DataFrame with analysis results including similarity scores
            timestamp: Timestamp for file naming
            
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / f"sentence_transformer_results_{timestamp}.csv"
        results.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    @handle_exception
    def save_visualization(self, fig: Any, name: str) -> str:
        """
        Save visualization figure
        
        Args:
            fig: Matplotlib figure object
            name: Base name for the visualization
            
        Returns:
            Path to saved visualization
        """
        filepath = self.viz_dir / f"{name}.png"
        fig.savefig(filepath)
        logger.info(f"Visualization saved to {filepath}")
        return str(filepath)