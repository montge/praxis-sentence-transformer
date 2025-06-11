"""
# src/clients/claude/requirement_analyzer.py

This module handles analysis of requirements using the Claude API. It provides functionality
to analyze relationships between software requirements and determine their associations.

Example:
    ```python
    analyzer = ClaudeRequirementAnalyzer()
    matches = analyzer.process_mappings(mapping_data)
    analyzed_results = analyzer.analyze_requirement_matches(matches)
    analyzer.save_results(analyzed_results, "output/dir")
    ```
"""

import os
import json
import logging
import anthropic
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from ...logger.logger import handle_exception, setup_logging, DebugTimer
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError
from multiprocessing import Pool, cpu_count
from functools import partial

logger = setup_logging(__name__, logging.INFO)

@dataclass
class RequirementMatch:
    """
    Represents a matched pair of requirements with Claude's analysis results.
    
    Attributes:
        source_id (str): Identifier of the source requirement
        source_content (str): Content of the source requirement
        target_id (str): Identifier of the target requirement
        target_content (str): Content of the target requirement
        similarity_score (float): Initial similarity score (0.0 to 1.0)
        association_probability (float): Probability of association from Claude analysis
        is_associated (bool): True if association probability exceeds threshold
        explanation (str): Claude's explanation of the association
        timestamp (datetime): When the match was analyzed
    
    Example:
        ```python
        match = RequirementMatch(
            source_id="REQ-001",
            source_content="The system shall...",
            target_id="REQ-002",
            target_content="Users must...",
            similarity_score=0.85
        )
        ```
    """
    
    source_id: str
    source_content: str
    target_id: str
    target_content: str
    similarity_score: Optional[float] = None
    association_probability: float = 0.0
    is_associated: bool = False
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_mapping(cls, source: Dict[str, Any], target: Dict[str, Any]) -> 'RequirementMatch':
        """Create RequirementMatch from mapping format"""
        return cls(
            source_id=source['id'],
            source_content=source['content'],
            target_id=target['id'],
            target_content=target['content'],
            similarity_score=target.get('similarity')
        )

@dataclass
class ClaudeAnalysisResults:
    """Container for storing all Claude analysis results"""
    processed_matches: List[RequirementMatch]
    total_source_requirements: int
    total_target_matches: int
    total_associated_matches: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert results to dictionary format"""
        return {
            "processed_matches": [asdict(match) for match in self.processed_matches],
            "total_source_requirements": self.total_source_requirements,
            "total_target_matches": self.total_target_matches, 
            "total_associated_matches": self.total_associated_matches,
            "timestamp": self.timestamp.isoformat()
        }

class Neo4jConfig(NamedTuple):
    """Configuration for Neo4j connection"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"

class ClaudeRequirementAnalyzer:
    """Analyzes requirements using Claude API"""
    
    def __init__(self, api_key: str = None, model_name: str = None, min_association_probability: float = 0.6, 
                 transformer_names: Optional[List[str]] = None):
        """
        Initialize the analyzer with Claude API client
        
        Args:
            api_key: Anthropic API key (defaults to env var ANTHROPIC_API_KEY)
            model_name: Name of Claude model to use (defaults to env var CLAUDE_3_5_MODEL)
            min_association_probability: Minimum probability threshold for associations
            transformer_names: Optional list of sentence transformer model names used for similarity scores
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and ANTHROPIC_API_KEY environment variable not set")
        
        # Get model name from parameter or environment
        self.model_name = model_name or os.getenv("CLAUDE_3_5_MODEL")
        if not self.model_name:
            raise ValueError("Model name not provided and CLAUDE_3_5_MODEL environment variable not set")
        
        self.min_association_probability = min_association_probability
        
        try:
            self.client = anthropic.Client(api_key=self.api_key)
            logger.debug("Successfully created Anthropic client")
        except Exception as e:
            logger.error(f"Failed to create Anthropic client: {str(e)}")
            raise
        
        self._last_raw_response = None
        self._system_prompt = None
        self._transformer_names = None

        logger.info(f"Initialized ClaudeRequirementAnalyzer with model: {self.model_name}")
        logger.info(f"Using minimum association probability: {self.min_association_probability}")

        # Set transformer names if provided
        self.transformer_names = transformer_names
        if self.transformer_names:
            logger.info(f"Using sentence transformers: {self.transformer_names}")
        else:
            logger.info("No sentence transformers specified")

    @property
    def system_prompt(self) -> str:
        """System prompt for analyzing requirement matches"""
        if self._system_prompt is None:
            # Base prompt with optional transformer description
            transformer_note = ""
            if self.transformer_names:
                transformer_desc = ", ".join(self.transformer_names)
                transformer_note = f"\nNote: The similarity_scores array contains semantic similarity scores (0.0 to 1.0) generated using these sentence transformer models: {transformer_desc}. Higher scores indicate greater semantic similarity between requirements."

            self._system_prompt = f"""You are an AI assistant specialized in analyzing software requirements. Your task is to analyze a source requirement against potential matching requirements provided in JSON format.

The input will be a JSON object with this structure:
{{
    "source_requirement": {{
        "id": "REQ-001",
        "content": "The system shall..."
    }},
    "potential_matches": [
        {{
            "id": "REQ-002",
            "content": "Users must...",
            "similarity_scores": [0.85, 0.82, 0.78]  // Array of similarity scores
        }}
    ]
}}{transformer_note}

IMPORTANT: Your response must contain ONLY a valid JSON object in this exact format, with no additional text, explanations, or markdown formatting:

{{
    "source_requirement": {{
        "id": "<source_id>",
        "content": "<source_content>"
    }},
    "associations": [
        {{
            "target_id": "<target_id>",
            "similarity_scores": [<score1>, <score2>, ...],  // Original similarity scores if provided
            "association_probability": <float_between_0_and_1>,
            "is_associated": <true_if_probability_over_{self.min_association_probability}>,
            "explanation": "<brief_explanation_under_45_words>"
        }}
    ]
}}

Guidelines:
1. Return all matches, even if they are not associated
2. association_probability should be between 0 and 1
3. is_associated should be true if probability > {self.min_association_probability}
4. The explanation should be under 45 words
5. include similarity_scores array only if provided in input
6. Consider the following:
   - Direct functional relationships
   - Implementation dependencies
   - Shared system components
   - Technical constraints
   - Requirements hierarchy
   - Provides a partial implementation of a requirement
   - Semantic similarity scores from multiple transformer models, if provided
7. Although the inputs may contain a foreign language, your response MUST ONLY be in English.
8. CRITCAL: Responses MUST ONLY include the JSON content. Do not include any other text, explanations, markdown formatting, or additional content in your response."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt"""
        self._system_prompt = value

    def _create_claude_client(self):
        """
        Creates a new Claude client instance.
        
        Returns:
            anthropic.Client: New Claude client instance
        """
        try:
            return anthropic.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to create new Anthropic client: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=3, max=30),
        retry=retry_if_exception_type((HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying Claude API call in {retry_state.next_action.sleep} seconds..."
        )
    )
    def _make_claude_request(self, input_json: dict) -> str:
        """
        Makes a request to Claude API with retry logic.
        
        Args:
            input_json: JSON input for Claude
            
        Returns:
            str: Claude's response text
            
        Raises:
            Exception: If all retries fail
        """
        try:
            # Create new client for each attempt
            client = self._create_claude_client()
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=self.system_prompt,
                messages=[{
                    "role": "user", 
                    "content": json.dumps(input_json, indent=2)
                }]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API request failed: {str(e)}")
            raise

    def _create_process_client(self):
        """Creates a new Claude client instance for a process."""
        return anthropic.Client(api_key=self.api_key)

    def _process_single_source(self, source_data: tuple, batch_size: int = 5, neo4j_client=None) -> tuple:
        """
        Process a single source requirement and its targets.
        
        Args:
            source_data (tuple): Tuple of (source_id, source_group DataFrame)
            batch_size (int): Number of targets to process in each batch
            neo4j_client: Optional Neo4j client for writing results
            
        Returns:
            tuple: (requirement_matches, target_count, associated_count)
        """
        # Create a new client for this process
        process_client = self._create_process_client()
        
        source_id, source_group = source_data
        requirement_matches = []
        target_matches = 0
        associated_matches = 0
        
        source = {
            'id': source_id,
            'content': source_group['source_content'].iloc[0]
        }
        
        targets = [
            {
                'id': row['target_id'],
                'content': row['target_content']
            }
            for _, row in source_group.iterrows()
        ]
        
        logger.info(f"Processing source requirement: {source_id} with {len(targets)} targets")
        
        for i in range(0, len(targets), batch_size):
            batch_targets = targets[i:i + batch_size]
            input_json = {
                "source_requirement": {
                    "id": source['id'],
                    "content": source['content']
                },
                "potential_matches": batch_targets
            }
            
            try:
                # Use process_client instead of self._make_claude_request
                response = process_client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=[{
                        "role": "user", 
                        "content": json.dumps(input_json, indent=2)
                    }]
                )
                raw_response = response.content[0].text
                
                json_content = self._extract_json_from_response(raw_response)
                analysis = json.loads(json_content)
                
                batch_matches = []
                for target in batch_targets:
                    target_analysis = next(
                        (a for a in analysis.get('associations', []) 
                         if a['target_id'] == target['id']),
                        None
                    )
                    
                    match = RequirementMatch(
                        source_id=source['id'],
                        source_content=source['content'],
                        target_id=target['id'],
                        target_content=target['content'],
                        similarity_score=target.get('similarity'),
                        association_probability=target_analysis['association_probability'] if target_analysis else 0.0,
                        is_associated=target_analysis['is_associated'] if target_analysis else False,
                        explanation=target_analysis['explanation'] if target_analysis else "No analysis provided",
                        timestamp=datetime.now()
                    )
                    batch_matches.append(match)
                    requirement_matches.append(match)
                
                if neo4j_client and batch_matches:
                    self._write_batch_to_neo4j(batch_matches, neo4j_client)
                
                target_matches += len(batch_targets)
                associated_matches += len([m for m in batch_matches if m.is_associated])
                
            except Exception as e:
                logger.error(f"Error processing source {source_id} batch {i//batch_size + 1}: {str(e)}")
                continue
        
        return requirement_matches, target_matches, associated_matches

    def process_mappings(self, mappings: Dict[str, Any], neo4j_client=None) -> ClaudeAnalysisResults:
        """Process source requirements in parallel through Claude analysis."""
        logger.info("Starting parallel processing of mappings with Claude...")
        
        df = mappings['data']
        source_groups = list(df.groupby('source_id'))
        
        # Determine number of processes (max concurrent)
        concurrent_processes = os.getenv('CONCURRENT_PROCESSES')
        if concurrent_processes is not None:
            try:
                concurrent_processes = int(concurrent_processes)
            except ValueError:
                logger.warning(f"Invalid CONCURRENT_PROCESSES value: {concurrent_processes}. Using CPU count.")
                concurrent_processes = cpu_count()
        else:
            logger.debug("CONCURRENT_PROCESSES not set. Using CPU count.")
            concurrent_processes = cpu_count()

        n_processes = min(concurrent_processes, len(source_groups), cpu_count())
        logger.info(f"Processing {len(source_groups)} source requirements using {n_processes} processes")
        
        # Create Neo4j config if client is provided
        neo4j_config = None
        if neo4j_client:
            neo4j_config = Neo4jConfig(
                uri=neo4j_client.uri,
                username=neo4j_client.username,
                password=neo4j_client.password,
                database=neo4j_client.database
            )
        
        # Prepare arguments for each process
        process_args = [
            (
                source_data,
                self.api_key,
                self.model_name,
                self.min_association_probability,
                self.system_prompt,
                5,  # batch_size
                neo4j_config
            )
            for source_data in source_groups
        ]
        
        all_requirement_matches = []
        total_target_matches = 0
        total_associated_matches = 0
        
        # Process sources in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_source_standalone, process_args)
            
            # Combine results
            for matches, target_count, associated_count in results:
                all_requirement_matches.extend(matches)
                total_target_matches += target_count
                total_associated_matches += associated_count
        
        # Create results container
        results = ClaudeAnalysisResults(
            processed_matches=all_requirement_matches,
            total_source_requirements=len(source_groups),
            total_target_matches=total_target_matches,
            total_associated_matches=total_associated_matches
        )
        
        logger.info(f"Successfully processed {len(all_requirement_matches)} total requirement matches")
        logger.info(f"Found {total_associated_matches} associated matches")
        
        return results

    def process_mappings_with_sentence_transformers(self, mappings: Dict[str, Any], sentence_transformers: List[str], neo4j_client=None) -> ClaudeAnalysisResults:
        """
        Process source requirements with sentence transformer scores through Claude analysis.
        
        Args:
            mappings (Dict[str, Any]): Mapping data containing requirements and similarity scores
            sentence_transformers (List[str]): List of sentence transformer column names to use
            neo4j_client: Optional Neo4j client for writing results
            
        Returns:
            ClaudeAnalysisResults: Analysis results including processed matches
            
        Example:
            ```python
            transformers = ['all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-v2']
            results = analyzer.process_mappings_with_sentence_transformers(
                mappings=mapping_data,
                sentence_transformers=transformers
            )
            ```
        """
        logger.info(f"Starting parallel processing with sentence transformers: {sentence_transformers}")
        
        df = mappings['data']
        
        # Validate that all transformer columns exist
        missing_cols = [col for col in sentence_transformers if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing transformer columns in data: {missing_cols}")
        
        source_groups = list(df.groupby('source_id'))
        
        # Determine number of processes (max concurrent)
        concurrent_processes = os.getenv('CONCURRENT_PROCESSES')
        if concurrent_processes is not None:
            try:
                concurrent_processes = int(concurrent_processes)
            except ValueError:
                logger.warning(f"Invalid CONCURRENT_PROCESSES value: {concurrent_processes}. Using CPU count.")
                concurrent_processes = cpu_count()
        else:
            logger.debug("CONCURRENT_PROCESSES not set. Using CPU count.")
            concurrent_processes = cpu_count()

        n_processes = min(concurrent_processes, len(source_groups), cpu_count())
        logger.info(f"Processing {len(source_groups)} source requirements using {n_processes} processes")
        
        # Create Neo4j config if client is provided
        neo4j_config = None
        if neo4j_client:
            neo4j_config = Neo4jConfig(
                uri=neo4j_client.uri,
                username=neo4j_client.username,
                password=neo4j_client.password,
                database=neo4j_client.database
            )
        
        # Prepare arguments for each process
        process_args = [
            (
                (source_id, group),  # source_data
                self.api_key,
                self.model_name,
                self.min_association_probability,
                self.system_prompt,
                5,  # batch_size
                neo4j_config,
                sentence_transformers  # Add transformer names
            )
            for source_id, group in source_groups
        ]
        
        all_requirement_matches = []
        total_target_matches = 0
        total_associated_matches = 0
        
        # Process sources in parallel using modified standalone function
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_source_with_transformers_standalone, process_args)
            
            # Combine results
            for matches, target_count, associated_count in results:
                all_requirement_matches.extend(matches)
                total_target_matches += target_count
                total_associated_matches += associated_count
        
        # Create results container
        results = ClaudeAnalysisResults(
            processed_matches=all_requirement_matches,
            total_source_requirements=len(source_groups),
            total_target_matches=total_target_matches,
            total_associated_matches=total_associated_matches
        )
        
        logger.info(f"Successfully processed {len(all_requirement_matches)} total requirement matches")
        logger.info(f"Found {total_associated_matches} associated matches")
        
        return results

    def _extract_json_from_response(self, raw_response: str) -> str:
        """Extract JSON content from Claude response."""
        json_block_start = raw_response.find("```json")
        if json_block_start != -1:
            json_block_end = raw_response.find("```", json_block_start + 6)
            if json_block_end != -1:
                return raw_response[json_block_start + 7:json_block_end].strip()
            else:
                json_start = raw_response.find('{', json_block_start)
                json_end = raw_response.rindex('}') + 1
                return raw_response[json_start:json_end]
        else:
            json_start = raw_response.find('{')
            json_end = raw_response.rindex('}') + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in response")
            return raw_response[json_start:json_end]

    def _write_batch_to_neo4j(self, batch_matches: List[RequirementMatch], neo4j_client, transformers_utilized: List[str] = None) -> None:
        """
        Write batch of matches to Neo4j.
        
        Args:
            batch_matches: List of requirement matches to write
            neo4j_client: Neo4j client instance
            transformers_utilized: Optional list of transformer names used in analysis
        """
        temp_results = ClaudeAnalysisResults(
            processed_matches=batch_matches,
            total_source_requirements=1,
            total_target_matches=len(batch_matches),
            total_associated_matches=len([m for m in batch_matches if m.is_associated])
        )
        
        try:
            success_count, failure_count = neo4j_client.create_llm_traces_from_results(
                results_set=temp_results,
                llm_model_name=self.model_name,
                remove_existing=False,
                transformers_utilized=transformers_utilized  # This will determine the relationship type
            )
            
            rel_type = "LLM_RESULT_WITH_TRANSFORMERS" if transformers_utilized else "LLM_RESULT"
            logger.info(f"Wrote {success_count} {rel_type} traces to Neo4j")
            if failure_count > 0:
                logger.warning(f"Failed to write {failure_count} {rel_type} traces")
        except Exception as e:
            logger.error(f"Error writing traces to Neo4j: {str(e)}")

    def analyze_requirement_matches(self, requirement_matches: List[RequirementMatch]) -> List[RequirementMatch]:
        """
        Analyzes requirement matches using Claude API to determine associations.
        
        Args:
            requirement_matches (List[RequirementMatch]): List of requirement matches to analyze
        
        Returns:
            List[RequirementMatch]: Analyzed matches with updated fields:
                - association_probability: Float between 0 and 1
                - is_associated: True if probability > min_association_probability
                - explanation: Claude's analysis explanation
                
        Raises:
            Exception: If Claude API call fails or response parsing fails
            
        Example:
            ```python
            matches = [
                RequirementMatch(
                    source_id="REQ-001",
                    source_content="The system shall authenticate users",
                    target_id="REQ-002", 
                    target_content="Users must provide valid credentials",
                    similarity_score=0.85
                )
            ]
            analyzed = analyzer.analyze_requirement_matches(matches)
            # Returns updated matches with:
            # - association_probability = 0.95
            # - is_associated = True
            # - explanation = "Direct functional relationship between authentication requirement and credentials"
            ```
        """
        # Group matches by source requirement to minimize API calls
        source_groups = {}
        for match in requirement_matches:
            if match.source_id not in source_groups:
                source_groups[match.source_id] = []
            source_groups[match.source_id].append(match)

        analyzed_matches = []
        total_sources = len(source_groups)
        
        logger.info(f"Starting analysis of {total_sources} source requirements with {len(requirement_matches)} total matches")

        for i, (source_id, matches) in enumerate(source_groups.items(), 1):
            logger.info(f"Processing source requirement {i}/{total_sources}: {source_id} with {len(matches)} matches")
            
            try:
                # Create prompt for Claude
                prompt = self.create_analysis_prompt(source_id, matches)

                # Get analysis from Claude with retry logic
                self._last_raw_response = self._make_claude_request(prompt)
                
                # Parse response and update matches
                updated_matches = self._parse_claude_response(self._last_raw_response, matches)
                analyzed_matches.extend(updated_matches)
                
                logger.info(f"Updated {len(updated_matches)} matches for source {source_id}")
                
            except Exception as e:
                logger.error(f"Error analyzing source {source_id}: {str(e)}")
                continue

        successful_matches = [m for m in analyzed_matches if m.is_associated]
        logger.info(f"Analysis complete. Found {len(successful_matches)}/{len(requirement_matches)} associated matches")
        
        return analyzed_matches

    def save_results(self, results: List[RequirementMatch], output_dir: str) -> bool:
        """
        Save analysis results to JSON file in the specified directory.
        
        Args:
            results (List[RequirementMatch]): Analyzed requirement matches to save
            output_dir (str): Directory path to save results
            
        Returns:
            bool: True if save successful, False otherwise
            
        Raises:
            OSError: If directory creation or file writing fails
            
        Example:
            ```python
            success = analyzer.save_results(analyzed_matches, "output/results")
            # Creates file: output/results/claude_analysis_20240315_143022.json
            # With content:
            # [
            #   {
            #     "source_id": "REQ-001",
            #     "source_content": "The system shall authenticate users",
            #     "target_id": "REQ-002",
            #     "target_content": "Users must provide valid credentials",
            #     "similarity_score": 0.85,
            #     "association_probability": 0.95,
            #     "is_associated": true,
            #     "explanation": "Direct functional relationship...",
            #     "timestamp": "2024-03-15T14:30:22"
            #   }
            # ]
            ```
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"claude_analysis_{timestamp}.json")
            
            results_dict = [
                {
                    "source_id": r.source_id,
                    "source_content": r.source_content,
                    "target_id": r.target_id,
                    "target_content": r.target_content,
                    **({"similarity_score": r.similarity_score} if r.similarity_score is not None else {}),
                    "association_probability": r.association_probability,
                    "is_associated": r.is_associated,
                    "explanation": r.explanation,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            return False

    def _sanitize_content(self, content: str) -> str:
        """
        Remove or replace problematic characters in content.
        
        Args:
            content (str): Content string to sanitize
            
        Returns:
            str: Sanitized content string with control characters removed/replaced
            
        Example:
            ```python
            content = "Some text\x00with\x01control\x02chars"
            clean = analyzer._sanitize_content(content)
            # Returns: "Some text with control chars"
            ```
        """
        if content is None:
            return ""
        
        # Remove null bytes
        content = content.replace('\x00', '')
        
        # Replace other control characters
        content = ''.join(char if ord(char) >= 32 else ' ' for char in content)
        
        # Log if we found any control chars
        if content != content.encode('utf-8', 'ignore').decode('utf-8'):
            logger.warning(f"Found and sanitized control characters in content: {repr(content)}")
        
        return content.strip()

    def _parse_claude_response(self, response: str, matches: List[RequirementMatch]) -> List[RequirementMatch]:
        """
        Parses Claude API response and updates matches. Handles cases where response contains
        text before/after the JSON content and markdown code blocks.
        
        Args:
            response: Response text from Claude API
            matches: Original requirement matches to update
        
        Returns:
            List of updated requirement matches
        """
        try:
            # Look for JSON content within markdown code blocks first
            json_block_start = response.find("```json")
            if json_block_start != -1:
                # Find the end of the code block
                json_block_end = response.find("```", json_block_start + 6)
                if json_block_end != -1:
                    # Extract just the JSON content from within the code block
                    json_content = response[json_block_start + 7:json_block_end].strip()
                else:
                    # Fallback to looking for just the JSON object if no end marker
                    json_start = response.find('{', json_block_start)
                    json_end = response.rindex('}') + 1
                    json_content = response[json_start:json_end]
            else:
                # Fallback to looking for just the JSON object
                json_start = response.find('{')
                json_end = response.rindex('}') + 1
                if json_start == -1 or json_end == -1:
                    raise ValueError("No JSON object found in response")
                json_content = response[json_start:json_end]

            try:
                parsed = json.loads(json_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON content: {str(e)}")
                logger.debug(f"Extracted JSON content: {json_content}")
                raise
            
            # Handle case where no associations were found
            if not parsed.get("associations"):
                logger.info(f"No valid associations found for source requirement: {parsed['source_requirement']['id']}")
                # Mark all matches as unassociated
                for match in matches:
                    match.association_probability = 0.0
                    match.is_associated = False
                    match.explanation = "No valid association found by analysis"
                return matches
            
            # Create a mapping of target_ids to their analysis results
            analysis_map = {
                analysis["target_id"]: analysis 
                for analysis in parsed["associations"]
            }
            
            # Update each match with its corresponding analysis
            updated_matches = []
            for match in matches:
                if match.target_id in analysis_map:
                    analysis = analysis_map[match.target_id]
                    match.association_probability = analysis["association_probability"]
                    match.is_associated = analysis["is_associated"]
                    match.explanation = analysis["explanation"]
                    updated_matches.append(match)
                else:
                    logger.debug(f"No analysis found for target_id: {match.target_id}")
                    # Keep the match but mark it as unassociated
                    match.association_probability = 0.0
                    match.is_associated = False
                    match.explanation = "Not identified as a valid match"
                    updated_matches.append(match)
                
            return updated_matches
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Claude response: {str(e)}")
            logger.debug(f"Full response content: {repr(response)}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected field in Claude response: {str(e)}")
            logger.debug(f"Response content: {repr(response)}")
            raise

    @property
    def transformer_names(self) -> Optional[List[str]]:
        """Get the list of transformer names"""
        return self._transformer_names

    @transformer_names.setter
    def transformer_names(self, value: Optional[List[str]]):
        """
        Set the list of transformer names
        
        Args:
            value: List of transformer names or None
        """
        self._transformer_names = value
        # Reset system prompt when transformer names change
        self._system_prompt = None
        if value:
            logger.info(f"Updated transformer names to: {value}")
        else:
            logger.info("Cleared transformer names")

def _process_source_standalone(args):
    """
    Standalone function for processing a single source requirement.
    
    Args:
        args (tuple): (source_data, api_key, model_name, min_prob, system_prompt, batch_size, neo4j_config)
        
    Returns:
        tuple: (requirement_matches, target_count, associated_count)
    """
    source_data, api_key, model_name, min_prob, system_prompt, batch_size, neo4j_config = args
    
    # Create a new analyzer instance for this process
    analyzer = ClaudeRequirementAnalyzer(
        api_key=api_key,
        model_name=model_name,
        min_association_probability=min_prob
    )
    analyzer.system_prompt = system_prompt
    
    # Create new Neo4j client if config is provided
    neo4j_client = None
    if neo4j_config:
        from ...neo4j_operations import Neo4jClient
        neo4j_client = Neo4jClient(
            uri=neo4j_config.uri,
            username=neo4j_config.username,
            password=neo4j_config.password,
            database=neo4j_config.database
        )
        neo4j_client.connect()
    
    try:
        return analyzer._process_single_source(source_data, batch_size=batch_size, neo4j_client=neo4j_client)
    finally:
        if neo4j_client:
            neo4j_client.close()

def _process_source_with_transformers_standalone(args):
    """
    Standalone function for processing a single source requirement with transformer scores.
    
    Args:
        args (tuple): (source_data, api_key, model_name, min_prob, system_prompt, 
                      batch_size, neo4j_config, transformer_names)
        
    Returns:
        tuple: (requirement_matches, target_count, associated_count)
    """
    (source_data, api_key, model_name, min_prob, system_prompt, 
     batch_size, neo4j_config, transformer_names) = args
    
    # Create a new analyzer instance for this process
    analyzer = ClaudeRequirementAnalyzer(
        api_key=api_key,
        model_name=model_name,
        min_association_probability=min_prob,
        transformer_names=transformer_names  # Pass transformer names to analyzer
    )
    analyzer.system_prompt = system_prompt
    
    # Create new Neo4j client if config is provided
    neo4j_client = None
    if neo4j_config:
        from ...neo4j_operations import Neo4jClient
        neo4j_client = Neo4jClient(
            uri=neo4j_config.uri,
            username=neo4j_config.username,
            password=neo4j_config.password,
            database=neo4j_config.database
        )
        neo4j_client.connect()
    
    try:
        source_id, source_group = source_data
        requirement_matches = []
        target_matches = 0
        associated_matches = 0
        
        source = {
            'id': source_id,
            'content': source_group['source_content'].iloc[0]
        }
        
        # Create targets with similarity scores array
        targets = []
        for _, row in source_group.iterrows():
            target = {
                'id': row['target_id'],
                'content': row['target_content'],
                'similarity_scores': [float(row[col]) for col in transformer_names]  # Get scores for selected transformers
            }
            targets.append(target)
        
        logger.info(f"Processing source requirement: {source_id} with {len(targets)} targets")
        
        for i in range(0, len(targets), batch_size):
            batch_targets = targets[i:i + batch_size]
            input_json = {
                "source_requirement": source,
                "potential_matches": batch_targets
            }
            
            try:
                # Use process_client instead of self._make_claude_request
                process_client = analyzer._create_claude_client()
                response = process_client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{
                        "role": "user", 
                        "content": json.dumps(input_json, indent=2)
                    }]
                )
                raw_response = response.content[0].text
                
                json_content = analyzer._extract_json_from_response(raw_response)
                analysis = json.loads(json_content)
                
                batch_matches = []
                for target in batch_targets:
                    target_analysis = next(
                        (a for a in analysis.get('associations', []) 
                         if a['target_id'] == target['id']),
                        None
                    )
                    
                    match = RequirementMatch(
                        source_id=source['id'],
                        source_content=source['content'],
                        target_id=target['id'],
                        target_content=target['content'],
                        similarity_score=None,  # Don't use single score since we have array
                        association_probability=target_analysis['association_probability'] if target_analysis else 0.0,
                        is_associated=target_analysis['is_associated'] if target_analysis else False,
                        explanation=target_analysis['explanation'] if target_analysis else "No analysis provided",
                        timestamp=datetime.now()
                    )
                    batch_matches.append(match)
                    requirement_matches.append(match)
                
                if neo4j_client and batch_matches:
                    analyzer._write_batch_to_neo4j(
                        batch_matches, 
                        neo4j_client, 
                        transformers_utilized=transformer_names
                    )
                
                target_matches += len(batch_targets)
                associated_matches += len([m for m in batch_matches if m.is_associated])
                
            except Exception as e:
                logger.error(f"Error processing source {source_id} batch {i//batch_size + 1}: {str(e)}")
                continue
        
        return requirement_matches, target_matches, associated_matches
        
    finally:
        if neo4j_client:
            neo4j_client.close()