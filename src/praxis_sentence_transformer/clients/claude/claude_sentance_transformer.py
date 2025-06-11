# src/praxis_sentence_transformer/clients/claude/claude_sentence_transformer.py

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ModelSpecialization:
    """Represents a model's specialization and characteristics"""
    name: str
    specialization: str
    strengths: List[str]
    architecture: str

class ClaudeSentenceTransformer:
    """
    A class to enhance Claude's analysis using sentence transformer scores.
    
    This class helps interpret and utilize sentence transformer scores to guide
    Claude's analysis of requirement relationships without relying on ground truth data.
    """
    
    def __init__(self):
        self.model_specs = {
            'qa_models': {
                'multi-qa-mpnet-dot': ModelSpecialization(
                    name='multi-qa-mpnet-base-dot-v1',
                    specialization='Cross-lingual question-answering',
                    strengths=['Semantic matching', 'Cross-language understanding'],
                    architecture='MPNet with dot-product similarity'
                ),
                'multi-qa-distilbert-cos': ModelSpecialization(
                    name='multi-qa-distilbert-cos-v1',
                    specialization='Lightweight QA matching',
                    strengths=['Fast processing', 'Memory efficient'],
                    architecture='DistilBERT with cosine similarity'
                )
            },
            'general_models': {
                'all-mpnet': ModelSpecialization(
                    name='all-mpnet-base-v2',
                    specialization='General semantic similarity',
                    strengths=['Robust semantic understanding', 'Good with technical text'],
                    architecture='MPNet base'
                ),
                'MiniLM': ModelSpecialization(
                    name='all-MiniLM-L6-v2',
                    specialization='Efficient general similarity',
                    strengths=['Fast processing', 'Good for short text'],
                    architecture='MiniLM base'
                )
            },
            'sts_models': {
                'stsb-bert-large': ModelSpecialization(
                    name='stsb-bert-large',
                    specialization='High-precision semantic similarity',
                    strengths=['Precise semantic matching', 'Good with nuanced differences'],
                    architecture='BERT large'
                ),
                'stsb-bert-base': ModelSpecialization(
                    name='stsb-bert-base',
                    specialization='Standard semantic similarity',
                    strengths=['Balance of precision and efficiency'],
                    architecture='BERT base'
                )
            }
        }

    def analyze_score_consensus(self, similarity_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze agreement patterns between different model scores.
        
        Args:
            similarity_scores: Dictionary of model names and their similarity scores
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            semantic_scores = [
                similarity_scores.get('multi-qa-mpnet-dot', 0),
                similarity_scores.get('all-mpnet', 0)
            ]
            
            technical_scores = [
                similarity_scores.get('stsb-bert-large', 0),
                similarity_scores.get('stsb-bert-base', 0)
            ]
            
            general_scores = [
                similarity_scores.get('MiniLM', 0),
                similarity_scores.get('distilroberta', 0)
            ]
            
            baseline_score = similarity_scores.get('tf-idf', 0)
            
            return {
                'semantic_consensus': np.mean(semantic_scores),
                'technical_consensus': np.mean(technical_scores),
                'general_consensus': np.mean(general_scores),
                'score_variance': np.var(list(similarity_scores.values())),
                'baseline_comparison': baseline_score
            }
        except Exception as e:
            logger.error(f"Error in analyze_score_consensus: {str(e)}")
            raise

    def get_focus_areas(self, analysis: Dict[str, float]) -> List[str]:
        """
        Determine areas that need special attention based on score patterns.
        
        Args:
            analysis: Dictionary containing score analysis results
            
        Returns:
            List of focus points for analysis
        """
        focus_points = []
        
        if analysis['score_variance'] > 0.2:
            focus_points.append(
                "Different models show significant disagreement. "
                "Consider multiple perspectives in your analysis."
            )
        
        if (analysis['semantic_consensus'] > 0.7 and 
            analysis['technical_consensus'] < 0.4):
            focus_points.append(
                "High semantic but low technical similarity. "
                "Verify if the relationship is functional rather than technical."
            )
        
        if (analysis['technical_consensus'] > 0.7 and 
            analysis['semantic_consensus'] < 0.4):
            focus_points.append(
                "High technical but low semantic similarity. "
                "Check if there's meaningful relationship beyond shared terminology."
            )
        
        if analysis['baseline_comparison'] > 0.8:
            focus_points.append(
                "High text similarity. "
                "Verify this isn't just similar wording with different intent."
            )
        
        return focus_points if focus_points else ["No specific patterns of concern identified."]

    def create_llm_guidance(self, similarity_scores: Dict[str, float]) -> str:
        """
        Create guidance for LLM based on score patterns.
        
        Args:
            similarity_scores: Dictionary of model names and their similarity scores
            
        Returns:
            Formatted prompt string for the LLM
        """
        try:
            analysis = self.analyze_score_consensus(similarity_scores)
            focus_areas = self.get_focus_areas(analysis)
            
            prompt = f"""
            When analyzing these requirements, consider these similarity patterns:

            1. Semantic Analysis:
            - Semantic similarity: {analysis['semantic_consensus']:.3f}
            - Technical similarity: {analysis['technical_consensus']:.3f}
            - General similarity: {analysis['general_consensus']:.3f}
            - Basic text similarity: {analysis['baseline_comparison']:.3f}

            2. Pattern Analysis:
            - Score agreement: {'High' if analysis['score_variance'] < 0.1 else 'Mixed'}
            - Semantic vs Technical gap: {abs(analysis['semantic_consensus'] - analysis['technical_consensus']):.3f}

            Focus Areas:
            {chr(10).join(f"- {point}" for point in focus_areas)}

            Please consider these patterns in your analysis, but make your own determination 
            based on the actual content. Explain how these patterns inform your thinking.
            """
            
            return prompt.strip()
        except Exception as e:
            logger.error(f"Error in create_llm_guidance: {str(e)}")
            raise

    def get_model_insights(self, model_name: str) -> Optional[ModelSpecialization]:
        """
        Get detailed insights about a specific model.
        
        Args:
            model_name: Name of the model to get insights for
            
        Returns:
            ModelSpecialization object if found, None otherwise
        """
        for category in self.model_specs.values():
            if model_name in category:
                return category[model_name]
        return None