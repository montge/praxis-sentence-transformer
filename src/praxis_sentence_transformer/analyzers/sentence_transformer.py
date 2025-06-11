"""
Module for analyzing requirements using sentence transformers
"""

import os
import gc
import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import math

from ..logger import setup_logging, handle_exception, DebugTimer
from ..loaders.requirements_loader import RequirementsLoader
from ..loaders.requirements_preprocessor import RequirementsPreprocessor
from ..utils.cuda import cleanup_cuda

logger = setup_logging("sentence-transformer-analyzer")

class SentenceTransformerAnalyzer:
    """Analyzes requirements using sentence transformers"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 alpha: float = 0.3,
                 device: Optional[torch.device] = None):
        """
        Initialize analyzer with model name and combination weight
        
        Parameters:
            model_name (str): Name of the sentence transformer model
            alpha (float): Weight for TF-IDF similarity (1-alpha for transformer-based similarity)
            device (Optional[torch.device]): Device to use for computation (cuda/cpu)
        """
        self.model_name = model_name
        self.model = None
        self.loader = RequirementsLoader()
        self.nlp = None
        self.preprocessor = None
        self.alpha = alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @handle_exception
    def initialize(self):
        """Initialize the sentence transformer model with proper CUDA configuration"""
        logger.info(f"Initializing sentence transformer model: {self.model_name}")
        
        # Verify CUDA configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
            logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Successfully loaded spaCy model")
        except OSError:
            logger.warning("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize model
        try:
            self.model = SentenceTransformer(self.model_name)
            
            if torch.cuda.is_available():
                logger.info("Moving model to GPU...")
                initial_memory = torch.cuda.memory_allocated()
                self.model = self.model.to(torch.device("cuda"))
                post_load_memory = torch.cuda.memory_allocated()
                logger.info(f"GPU Memory used by model: {(post_load_memory - initial_memory)/1e9:.2f} GB")
                
                # Test with minimal batch
                test_text = ["Test sentence"]
                with torch.no_grad():
                    self.model.encode(test_text)
                logger.info("Model successfully loaded on GPU")
                logger.info(f"Current GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                
            else:
                logger.info("CUDA not available, using CPU")
                self.model = self.model.to(torch.device("cpu"))
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            logger.warning("Falling back to CPU")
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(torch.device("cpu"))
            
        # Initialize preprocessor without passing nlp
        self.preprocessor = RequirementsPreprocessor()
        logger.info("Preprocessor initialized")

    @handle_exception
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing stop words and performing basic cleaning
        
        Parameters:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not self.nlp:
            logger.warning("spaCy model not initialized, returning original text")
            return text
            
        logger.debug(f"Original text: '{text}'")
        
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        processed_text = ' '.join(tokens)
        
        logger.debug(f"Processed text: '{processed_text}'")
        logger.debug(f"Removed {len(doc) - len(tokens)} tokens "
                    f"({len([t for t in doc if t.is_stop])} stop words, "
                    f"{len([t for t in doc if t.is_punct])} punctuation marks)")
        
        return processed_text

    @handle_exception
    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF similarity between two texts
        
        Parameters:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score between the TF-IDF vectors
            
        Raises:
            ValueError: If either text is empty/None
            RuntimeError: If preprocessor is not initialized
        """
        if not text1 or not text2:
            logger.error("Empty or None text provided for TF-IDF similarity")
            raise ValueError("Both texts must be non-empty")
        
        if not self.preprocessor:
            logger.error("Preprocessor not initialized")
            raise RuntimeError("Preprocessor must be initialized before computing similarities")
        
        try:
            logger.debug("Computing TF-IDF similarity")
            logger.debug(f"Text 1 (truncated): '{text1[:100]}...'")
            logger.debug(f"Text 2 (truncated): '{text2[:100]}...'")
            
            # Preprocess texts
            preprocessed_texts = [
                self.preprocessor.preprocess_text(text1),
                self.preprocessor.preprocess_text(text2)
            ]
            
            # Get TF-IDF vectors
            tfidf_vectors = self.preprocessor.get_tfidf_vectors(preprocessed_texts)
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]
            
            logger.debug(f"TF-IDF similarity score: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {str(e)}")
            logger.debug(f"Text 1 length: {len(text1)}")
            logger.debug(f"Text 2 length: {len(text2)}")
            raise

    def process_batch(self, texts: List[str], batch_idx: int, batch_size: int) -> Optional[torch.Tensor]:
        """Process a batch of texts with error handling"""
        try:
            # Convert texts to embeddings using the model
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=self.model.device
                )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            return None

    @handle_exception
    def get_embeddings(self, texts: List[str], batch_size: int = 32, desc: str = "") -> np.ndarray:
        """
        Get embeddings for a list of texts with batching and progress tracking
        
        Parameters:
            texts (List[str]): List of texts to encode
            batch_size (int): Size of batches for processing
            desc (str): Description for progress bar
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

    @handle_exception
    def analyze_requirements(self, 
                    source_file: str, 
                    target_file: str, 
                    answer_file: str = None,
                    batch_size: int = 32,
                    save_path: str = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze requirements and calculate similarity scores for all pairs
        
        Parameters:
            source_file (str): Path to source requirements file
            target_file (str): Path to target requirements file
            answer_file (str): Optional path to answer set file for evaluation
            batch_size (int): Batch size for processing
            save_path (str): Optional path to save results
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping source requirements to lists of (target_id, similarity) tuples
        """
        timer = DebugTimer("analyze_requirements")
        
        try:
            # Load requirements
            source_reqs = self.loader.parse_requirements(source_file)
            target_reqs = self.loader.parse_requirements(target_file)
            
            # Extract texts and IDs
            source_texts = []
            source_ids = []
            target_texts = []
            target_ids = []
            
            for req_id, desc in source_reqs:
                if desc and isinstance(desc, str):
                    source_texts.append(desc)
                    source_ids.append(req_id)
                    
            for req_id, desc in target_reqs:
                if desc and isinstance(desc, str):
                    target_texts.append(desc)
                    target_ids.append(req_id)
            
            # Generate embeddings
            source_embeddings = self.get_embeddings(
                texts=source_texts,
                batch_size=batch_size,
                desc="Generating source embeddings"
            )
            
            target_embeddings = self.get_embeddings(
                texts=target_texts,
                batch_size=batch_size,
                desc="Generating target embeddings"
            )
            
            # Calculate similarity matrix
            similarity_matrix = self.compute_similarity_matrix(source_embeddings, target_embeddings)
            
            # Create mapping of all pairs with similarity scores
            requirement_mapping = {}
            
            for i, src_id in enumerate(source_ids):
                matches = []
                for j, tgt_id in enumerate(target_ids):
                    similarity = float(similarity_matrix[i][j])
                    matches.append((tgt_id, similarity))
                
                # Store all matches sorted by similarity
                matches.sort(key=lambda x: x[1], reverse=True)
                requirement_mapping[src_id] = matches
            
            # Save results if path provided
            if save_path:
                self.save_detailed_results(
                    source_reqs=source_reqs,
                    target_reqs=target_reqs,
                    similarity_matrix=similarity_matrix,
                    requirement_mapping=requirement_mapping,
                    save_path=save_path
                )
            
            return requirement_mapping
            
        except Exception as e:
            logger.error(f"Error in analyze_requirements: {str(e)}")
            logger.exception("Detailed error trace:")
            return {}

    def find_optimal_threshold(self,
                           source_file: str,
                           target_file: str,
                           answer_set_file: str,
                           base_save_path: str,
                           threshold_range: List[float] = None) -> Dict[float, Dict[str, float]]:
        """
        Find optimal threshold by testing different values
        
        Parameters:
            source_file (str): Path to source requirements file
            target_file (str): Path to target requirements file
            answer_set_file (str): Path to answer set file
            base_save_path (str): Base path for saving results
            threshold_range (List[float]): List of thresholds to test
            
        Returns:
            Dict[float, Dict[str, float]]: Dictionary mapping thresholds to their evaluation metrics
        """
        if threshold_range is None:
            threshold_range = [round(x * 0.05, 2) for x in range(1, 13)]
                
        results = {}
        min_coverage = float(os.getenv('MIN_COVERAGE_THRESHOLD', '0.9'))
        
        # Sort thresholds to ensure we process them in ascending order
        sorted_thresholds = sorted(threshold_range)
        
        for threshold in sorted_thresholds:
            logger.info(f"Testing threshold: {threshold}")
            mappings = self.analyze_requirements(
                source_file, 
                target_file, 
                threshold=threshold,
                save_path=base_save_path
            )
            
            # Load answer set for evaluation
            answer_set = self.loader.parse_answer_set(answer_set_file)
            
            # Convert mappings to format needed for evaluation
            predicted_links = []
            for src_id, matches in mappings.items():
                for tgt_id, _ in matches:
                    predicted_links.append((src_id, tgt_id))
            
            # Evaluate results
            evaluation = self.evaluate_results(
                predicted_links=predicted_links,
                ground_truth=answer_set,
                threshold=threshold,
                save_dir=os.path.join(base_save_path, f'threshold_{threshold:.2f}')
            )
            
            # Store results regardless of recall threshold
            results[threshold] = evaluation
            
            # Calculate recall/coverage
            recall = evaluation['metrics']['recall']
            
            # Check if recall has dropped below minimum coverage threshold
            if recall < min_coverage:
                logger.info(f"Stopping threshold search as recall ({recall:.3f}) has dropped below "
                          f"minimum coverage threshold ({min_coverage})")
                break
                    
        return results

    def evaluate_results(self,
                      predicted_links: List[Tuple[str, str]],
                      ground_truth: List[Tuple[str, str]],
                      threshold: float,
                      save_dir: Optional[str] = None) -> Dict:
        """
        Evaluates the predicted requirement links against ground truth and saves detailed results
        
        Parameters:
            predicted_links (List[Tuple[str, str]]): List of predicted requirement links (source_id, target_id)
            ground_truth (List[Tuple[str, str]]): List of ground truth requirement links
            threshold (float): Similarity threshold used for prediction
            save_dir (Optional[str]): Directory to save evaluation results
            
        Returns:
            Dict: Dictionary containing evaluation metrics and counts
        """
        try:
            # Convert to sets for efficient comparison
            predicted_set = set(predicted_links)
            ground_truth_set = set(ground_truth)
            
            # Get unique source and target IDs from both predicted and ground truth
            source_ids = set([src for src, _ in predicted_links + ground_truth])
            target_ids = set([tgt for _, tgt in predicted_links + ground_truth])
            
            # Calculate total possible combinations
            total_source_reqs = len(source_ids)
            total_target_reqs = len(target_ids)
            total_possible = total_source_reqs * total_target_reqs
            
            logger.debug(f"Total source requirements: {total_source_reqs}")
            logger.debug(f"Total target requirements: {total_target_reqs}")
            logger.debug(f"Total possible combinations: {total_possible}")
            logger.info(f"Predicted links: {len(predicted_links)}")
            logger.info(f"Ground truth links: {len(ground_truth)}")
            
            # Calculate metrics
            true_positives = len(predicted_set.intersection(ground_truth_set))
            false_positives = len(predicted_set - ground_truth_set)
            false_negatives = len(ground_truth_set - predicted_set)
            true_negatives = total_possible - (true_positives + false_positives + false_negatives)
            
            logger.info(f"True Positives: {true_positives}")
            logger.info(f"False Positives: {false_positives}")
            logger.info(f"False Negatives: {false_negatives}")
            logger.info(f"True Negatives: {true_negatives}")
            
            # Calculate rates and scores
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / total_possible if total_possible > 0 else 0
            
            false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            true_negative_rate = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            balanced_accuracy = (recall + true_negative_rate) / 2
            
            # Create confusion matrix string
            confusion_matrix = (
                f"             Predicted Pos  Predicted Neg\n"
                f"Actual Pos   {true_positives:12d}  {false_negatives:12d}\n"
                f"Actual Neg   {false_positives:12d}  {true_negatives:12d}\n"
            )
            
            results = {
                "metrics": {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "false_negative_rate": false_negative_rate,
                    "true_negative_rate": true_negative_rate,
                    "confusion_matrix": confusion_matrix
                },
                "counts": {
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "true_negatives": true_negatives,
                    "false_negatives": false_negatives,
                    "total_samples": total_possible,
                    "total_predicted": len(predicted_links),
                    "total_ground_truth": len(ground_truth),
                    "total_source_reqs": total_source_reqs,
                    "total_target_reqs": total_target_reqs
                }
            }
            
            if save_dir:
                self._save_evaluation_details(
                    results=results,
                    true_positives=predicted_set.intersection(ground_truth_set),
                    false_positives=predicted_set - ground_truth_set,
                    false_negatives=ground_truth_set - predicted_set,
                    true_negatives=set(),  # We don't store all true negatives due to size
                    save_dir=save_dir
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluate_results: {str(e)}")
            logger.exception("Detailed error trace:")
            return self._create_empty_evaluation_result(threshold)

    def _create_empty_evaluation_result(self, threshold: float) -> Dict:
        """
        Create empty evaluation result structure
        
        Parameters:
            threshold (float): Threshold value used in evaluation
            
        Returns:
            Dict: Empty evaluation results structure
        """
        return {
            "metrics": {
                "threshold": threshold,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "true_negative_rate": 0.0
            },
            "counts": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "total_samples": 0,
                "total_predicted": 0,
                "total_ground_truth": 0,
                "total_source_reqs": 0,
                "total_target_reqs": 0
            },
            "error": "Evaluation failed"
        }

    def _save_evaluation_details(self,
                              results: Dict,
                              true_positives: set,
                              false_positives: set,
                              false_negatives: set,
                              true_negatives: set,
                              save_dir: str):
        """
        Save detailed evaluation results
        
        Parameters:
            results (Dict): Evaluation results dictionary
            true_positives (set): Set of true positive matches
            false_positives (set): Set of false positive matches
            false_negatives (set): Set of false negative matches
            true_negatives (set): Set of true negative matches
            save_dir (str): Directory to save results
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create detailed results with link information
            detailed_results = {
                **results,
                "detailed_links": {
                    "true_positives": sorted(list(true_positives)),
                    "false_positives": sorted(list(false_positives)),
                    "false_negatives": sorted(list(false_negatives)),
                    "true_negatives": sorted(list(true_negatives))[:1000]  # Limit TN to prevent huge files
                },
                "timestamp": datetime.now().isoformat(),
                "save_location": save_dir
            }
            
            # Save main results file
            save_path = os.path.join(save_dir, "detailed_evaluation.json")
            with open(save_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"Saved detailed evaluation results to {save_path}")
            
            # Save separate files for each category to avoid memory issues
            for category in ["true_positives", "false_positives", "false_negatives"]:
                category_path = os.path.join(save_dir, f"{category}.json")
                with open(category_path, 'w') as f:
                    json.dump(sorted(list(locals()[category])), f, indent=2)
                logger.debug(f"Saved {category} to {category_path}")
            
            # Save sample of true negatives
            tn_sample_path = os.path.join(save_dir, "true_negatives_sample.json")
            with open(tn_sample_path, 'w') as f:
                json.dump(sorted(list(true_negatives))[:1000], f, indent=2)
            logger.debug(f"Saved true negatives sample to {tn_sample_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation details: {str(e)}")
            logger.exception("Detailed error trace:")

    def compute_similarity_matrix(self, source_texts: List[str], target_texts: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between source and target texts by first encoding them
        into embeddings using the sentence transformer model.
        
        Parameters:
            source_texts: List of source requirement texts
            target_texts: List of target requirement texts
            
        Returns:
            np.ndarray: Similarity matrix
        """
        try:
            # First encode texts to embeddings using the model
            logger.debug("Encoding source texts to embeddings...")
            source_embeddings = self.model.encode(
                source_texts, 
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            logger.debug("Encoding target texts to embeddings...")
            target_embeddings = self.model.encode(
                target_texts, 
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Normalize embeddings
            source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.mm(source_embeddings, target_embeddings.t())
            
            return similarity_matrix.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error computing similarity matrix: {str(e)}")
            raise

    def process_requirements(self, requirements: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Process requirements in batches with progress bar
        
        Args:
            requirements: List of requirement texts
            batch_size: Size of batches for processing
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        embeddings_list = []
        
        for i in tqdm(range(0, len(requirements), batch_size),
                      desc="Processing requirements",
                      leave=True):
            batch = requirements[i:i + batch_size]
            batch_embeddings = self.process_batch(batch, i, batch_size)
            if batch_embeddings is not None:
                embeddings_list.append(batch_embeddings)
        
        return torch.cat(embeddings_list, dim=0)

    @handle_exception
    def compute_batch_tfidf_similarities(self, source_texts: List[str], target_texts: List[str]) -> np.ndarray:
        """
        Compute TF-IDF similarities between source and target texts in batches
        
        Args:
            source_texts: List of source requirement texts
            target_texts: List of target requirement texts
            
        Returns:
            np.ndarray: Similarity matrix of shape (len(source_texts), len(target_texts))
        """
        logger.info(f"Computing TF-IDF similarities between {len(source_texts)} source and {len(target_texts)} target requirements")
        logger.info(f"Using device: {self.device}")
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        
        # Process source texts with progress bar
        processed_source = []
        for i, text in enumerate(tqdm(source_texts, desc="Preprocessing source texts")):
            processed_text = self.preprocessor.preprocess_text(text)
            processed_source.append(processed_text)
            if (i + 1) % 10 == 0:  # Log every 10 requirements
                logger.debug(f"Preprocessed {i + 1}/{len(source_texts)} source requirements")
        
        # Process target texts with progress bar    
        processed_target = []
        for i, text in enumerate(tqdm(target_texts, desc="Preprocessing target texts")):
            processed_text = self.preprocessor.preprocess_text(text)
            processed_target.append(processed_text)
            if (i + 1) % 10 == 0:  # Log every 10 requirements
                logger.debug(f"Preprocessed {i + 1}/{len(target_texts)} target requirements")
        
        logger.info("Initializing TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(lowercase=True)
        
        # Fit on all texts to get common vocabulary
        logger.info("Fitting vectorizer on all texts...")
        all_texts = processed_source + processed_target
        vectorizer.fit(all_texts)
        
        # Transform texts
        logger.info("Transforming texts to TF-IDF vectors...")
        source_matrix = vectorizer.transform(processed_source)
        target_matrix = vectorizer.transform(processed_target)
        
        # Compute similarities in smaller batches
        batch_size = 16  # Smaller batch size for more frequent updates
        n_source = len(source_texts)
        similarities = np.zeros((n_source, len(target_texts)))
        
        logger.info(f"Computing similarities in batches of {batch_size}...")
        for i in tqdm(range(0, n_source, batch_size), desc="Computing TF-IDF similarities"):
            batch_end = min(i + batch_size, n_source)
            source_batch = source_matrix[i:batch_end]
            
            # Compute cosine similarity for the batch
            batch_similarities = cosine_similarity(source_batch, target_matrix)
            similarities[i:batch_end] = batch_similarities
            
            logger.debug(f"Processed {batch_end}/{n_source} source requirements")
        
        logger.info(f"Completed TF-IDF similarity computation")
        logger.info(f"Final similarity matrix shape: {similarities.shape}")
        
        return similarities

    def compute_sentence_transformer_similarities(self, 
                                               source_texts: List[str], 
                                               target_texts: List[str],
                                               batch_size: int = 32) -> np.ndarray:
        """
        Compute sentence transformer similarities using GPU acceleration
        
        Args:
            source_texts: List of source requirement texts
            target_texts: List of target requirement texts
            batch_size: Size of batches for processing
            
        Returns:
            np.ndarray: Normalized similarity matrix
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Computing sentence transformer similarities using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Process source texts in batches
        source_embeddings = []
        for i in tqdm(range(0, len(source_texts), batch_size), desc="Encoding source texts"):
            batch = source_texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_tensor=True)
                source_embeddings.append(embeddings)
        
        source_embeddings = torch.cat(source_embeddings, dim=0)
        
        # Process target texts in batches
        target_embeddings = []
        for i in tqdm(range(0, len(target_texts), batch_size), desc="Encoding target texts"):
            batch = target_texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_tensor=True)
                target_embeddings.append(embeddings)
        
        target_embeddings = torch.cat(target_embeddings, dim=0)
        
        # Compute similarities in batches
        similarities = torch.zeros((len(source_texts), len(target_texts)), device=device)
        
        for i in tqdm(range(0, len(source_texts), batch_size), desc="Computing similarities"):
            i_end = min(i + batch_size, len(source_texts))
            source_batch = source_embeddings[i:i_end]
            
            for j in range(0, len(target_texts), batch_size):
                j_end = min(j + batch_size, len(target_texts))
                target_batch = target_embeddings[j:j_end]
                
                # Compute cosine similarity
                sim_batch = F.cosine_similarity(
                    source_batch.unsqueeze(1), 
                    target_batch.unsqueeze(0),
                    dim=2
                )
                similarities[i:i_end, j:j_end] = sim_batch

        if device.type == 'cuda':
            logger.info(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        return similarities.cpu().numpy()