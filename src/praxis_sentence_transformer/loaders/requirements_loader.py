"""
Module for loading and managing requirements
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any
from ..logger import setup_logging, handle_exception
from .requirements_preprocessor import RequirementsPreprocessor
from ..neo4j_operations.neo4j_client import Neo4jClient
import os

class RequirementsLoader:
    """Handles loading and storing requirements"""
    
    def __init__(self, neo4j_client: Neo4jClient = None):
        """
        Initialize RequirementsLoader
        
        Args:
            neo4j_client: Neo4j client instance for database operations
        """
        self.logger = setup_logging("requirements-loader")
        self.preprocessor = RequirementsPreprocessor()
        self.client = neo4j_client
        self.project_name = os.getenv('PROJECT_NAME', 'default')
        
    @handle_exception
    def create_requirement(self, parent_id: str, req_id: str, content: str, 
                         req_type: str, level: int = 1, project_name: str = None) -> Dict:
        """
        Create requirement node in Neo4j
        
        Args:
            parent_id: ID of parent node (document or section)
            req_id: Requirement ID
            content: Requirement content
            req_type: Type of requirement (source/target)
            level: Hierarchy level
            project_name: Name of the project (defaults to environment variable)
            
        Returns:
            Dict: Created requirement node data
        """
        if not self.client:
            raise ValueError("Neo4j client not initialized. Pass client in constructor.")
            
        # Use provided project name or default from environment
        project_name = project_name or self.project_name
            
        with self.client.driver.session(database=self.client.database) as session:
            # Preprocess content with case preservation for CC docs
            processed_content = self.preprocessor.preprocess_text(
                content, 
                preserve_case=req_id.startswith('CC')
            )
            
            # First check if requirement exists
            existing = session.run("""
                MATCH (r:Requirement {id: $req_id, project: $project_name})
                RETURN r.id as id, r.type as type, r.level as level
            """, {
                'req_id': req_id,
                'project_name': project_name
            }).single()
            
            if existing:
                self.logger.debug(f"Requirement {req_id} already exists, updating content and ensuring link to parent")
                # Update existing requirement and ensure link to parent
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH (parent {id: $parent_id, project: $project_name})
                    MATCH (r:Requirement {id: $req_id, project: $project_name})
                    SET r.content = $content,
                        r.type = $type,
                        r.level = $level
                    MERGE (parent)-[:CONTAINS]->(r)
                    RETURN r.id as id, r.type as type, r.level as level
                """, {
                    'parent_id': parent_id,
                    'req_id': req_id,
                    'content': processed_content,
                    'type': req_type,
                    'level': level,
                    'project_name': project_name
                }).single()
            else:
                # Create new requirement
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH (parent {id: $parent_id, project: $project_name})
                    CREATE (r:Requirement {
                        id: $req_id,
                        project: $project_name,
                        content: $content,
                        type: $type,
                        level: $level
                    })
                    CREATE (parent)-[:CONTAINS]->(r)
                    RETURN r.id as id, r.type as type, r.level as level
                """, {
                    'parent_id': parent_id,
                    'req_id': req_id,
                    'content': processed_content,
                    'type': req_type,
                    'level': level,
                    'project_name': project_name
                }).single()
            
            if not result:
                self.logger.error(f"Failed to create/update requirement {req_id}")
                raise ValueError(f"Failed to create/update requirement {req_id}")
            
            self.logger.debug(f"Created/updated requirement: {result['id']} (level {result['level']}) for project {project_name}")
            return result
    
    @handle_exception
    def parse_requirements(self, file_path: str) -> List[Dict[str, str]]:
        """Parse requirements from XML file."""
        self.logger.debug(f"Parsing requirements from {file_path}")
        requirements = []
        seen_ids = set()
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Log the raw count of requirement elements
            raw_req_count = len(root.findall('.//artifact'))
            self.logger.debug(f"Found {raw_req_count} raw requirement elements in XML")
            
            for req in root.findall('.//artifact'):
                req_id = req.findtext('id', '').strip()
                content = req.findtext('content', '').strip()
                
                if req_id and content:
                    if req_id in seen_ids:
                        self.logger.warning(f"Duplicate requirement ID found: {req_id}")
                    else:
                        seen_ids.add(req_id)
                        requirements.append({
                            'id': req_id,
                            'content': content
                        })
                else:
                    if not req_id:
                        self.logger.warning("Skipping requirement with missing id")
                    if not content:
                        self.logger.warning(f"Skipping requirement with missing content for id: {req_id}")
            
            self.logger.debug(f"Raw requirements count: {raw_req_count}")
            self.logger.debug(f"Valid unique requirements parsed: {len(requirements)}")
            if raw_req_count != len(requirements):
                self.logger.warning(f"Discrepancy found: {raw_req_count - len(requirements)} requirements were either duplicates or invalid")
            self.logger.debug(f"First requirement: {requirements[0] if requirements else 'None'}")
            self.logger.debug(f"Last requirement: {requirements[-1] if requirements else 'None'}")
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Error parsing requirements from {file_path}: {str(e)}")
            raise
    
    @handle_exception
    def parse_answer_set(self, file_path: str) -> List[Tuple[str, str]]:
        """Parse answer set from XML file"""
        self.logger.debug(f"Parsing answer set from {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        mappings = []
        for link in root.findall('.//link'):
            source = link.findtext('source_artifact_id', '').strip()  # Updated XML path
            target = link.findtext('target_artifact_id', '').strip()  # Updated XML path
            
            if source and target:
                mappings.append((source, target))
                
        self.logger.info(f"Successfully parsed {len(mappings)} reference mappings")
        return mappings
    
    @handle_exception
    def validate_requirements(self, source_reqs: List[Dict], target_reqs: List[Dict], 
                            answer_set: List[Tuple[str, str]]) -> bool:
        """
        Validate requirements against answer set
        
        Args:
            source_reqs: List of source requirements
            target_reqs: List of target requirements
            answer_set: List of (source_id, target_id) tuples
            
        Returns:
            bool: True if validation passes
        """
        # Create sets of IDs
        source_ids = {req['id'] for req in source_reqs}
        target_ids = {req['id'] for req in target_reqs}
        answer_source_ids = {src for src, _ in answer_set}
        answer_target_ids = {tgt for _, tgt in answer_set}
        
        # Check for missing requirements
        missing_source = answer_source_ids - source_ids
        missing_target = answer_target_ids - target_ids
        
        # Log validation results
        self.logger.info("Requirements Validation Summary:")
        self.logger.info(f"Source requirements: {len(source_ids)}")
        self.logger.info(f"Target requirements: {len(target_ids)}")
        self.logger.info(f"Answer set mappings: {len(answer_set)}")
        
        if missing_source:
            self.logger.error(f"Missing source requirements: {sorted(missing_source)}")
        if missing_target:
            self.logger.error(f"Missing target requirements: {sorted(missing_target)}")
            
        is_valid = not (missing_source or missing_target)
        self.logger.info(f"Validation {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid