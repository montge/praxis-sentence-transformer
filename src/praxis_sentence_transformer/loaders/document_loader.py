"""
Module for loading and managing document hierarchies in Neo4j
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from ..logger import setup_logging, handle_exception
from ..neo4j_operations.neo4j_client import Neo4jClient
from ..neo4j_operations.models import Project, Document, Section, Requirement
import xml.etree.ElementTree as ET
from .requirements_loader import RequirementsLoader

class DocumentHierarchyLoader:
    """Loads and stores document hierarchies in Neo4j"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize DocumentHierarchyLoader
        
        Args:
            neo4j_client: Neo4j client instance for database operations
        """
        self.logger = setup_logging(__name__)
        self.neo4j_client = neo4j_client
        self.requirements_loader = RequirementsLoader(neo4j_client=neo4j_client)
        self.project_name = os.getenv('PROJECT_NAME', 'default')
        
    @handle_exception
    def create_project(self) -> Project:
        """Create project in Neo4j"""
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run("""
                MERGE (p:Project {name: $name})
                SET p.description = $description,
                    p.created_at = datetime()
                RETURN p.name as name, p.description as description
            """, {
                'name': self.project_name,
                'description': f"Project {self.project_name}"
            }).single()
            
            self.logger.info(f"Created/verified project: {result['name']}")
            return Project(name=result['name'], description=result['description'])
    
    @handle_exception
    def create_document(self, doc_id: str, title: str, doc_type: str) -> Document:
        """Create document node in Neo4j"""
        try:
            self.logger.debug(f"Creating document with ID: {doc_id}, title: {title}, type: {doc_type}")
            self.logger.debug(f"Project context: {self.project_name}")
            
            with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                # First verify project exists
                project_check = session.run("""
                    MATCH (p:Project {name: $project_name})
                    RETURN p.name as name
                """, {'project_name': self.project_name}).single()
                
                if not project_check:
                    self.logger.error(f"Project {self.project_name} not found")
                    raise ValueError(f"Project {self.project_name} not found")
                
                self.logger.debug(f"Found project: {project_check['name']}")
                
                # Create document with detailed logging
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MERGE (d:Document {id: $id, project: $project_name})
                    SET d.title = $title,
                        d.type = $type,
                        d.created_at = datetime()
                    MERGE (p)-[:CONTAINS]->(d)
                    RETURN d.id as id, d.title as title, d.type as type
                """, {
                    'project_name': self.project_name,
                    'id': doc_id,
                    'title': title,
                    'type': doc_type
                }).single()
                
                if not result:
                    self.logger.error(f"Failed to create document {doc_id}")
                    raise ValueError(f"Failed to create document {doc_id}")
                
                self.logger.info(f"Created/verified document: {result['id']} ({result['type']})")
                self.logger.debug(f"Document details: {result}")
                
                # Verify the relationship was created
                rel_check = session.run("""
                    MATCH (p:Project {name: $project_name})-[:CONTAINS]->(d:Document {id: $doc_id})
                    RETURN count(*) as count
                """, {
                    'project_name': self.project_name,
                    'doc_id': doc_id
                }).single()
                
                if rel_check['count'] == 0:
                    self.logger.error(f"Failed to create CONTAINS relationship for document {doc_id}")
                    raise ValueError(f"Failed to create CONTAINS relationship for document {doc_id}")
                
                self.logger.debug(f"Verified CONTAINS relationship for document {doc_id}")
                
                return Document(
                    id=result['id'],
                    title=result['title'],
                    content="",
                    type=result['type'],
                    project_name=self.project_name
                )
                
        except Exception as e:
            self.logger.error(f"Error creating document {doc_id}: {str(e)}")
            self.logger.exception("Detailed error trace:")
            raise
    
    @handle_exception
    def create_section(self, doc_id: str, section_id: str, title: str, level: int) -> Optional[Section]:
        """
        Create section node in Neo4j
        
        Args:
            doc_id: Parent document or section ID
            section_id: ID for the new section
            title: Section title
            level: Hierarchy level
            
        Returns:
            Optional[Section]: Created section object or None if creation fails
        """
        try:
            with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                # Create section with CONTAINS relationship from parent
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH (parent {id: $doc_id, project: $project_name})
                    MERGE (s:Section {id: $section_id, project: $project_name})
                    SET s.title = $title,
                        s.level = $level,
                        s.name = $section_id,
                        s.created_at = datetime()
                    MERGE (parent)-[:CONTAINS]->(s)
                    RETURN s.id as id, s.title as title, s.level as level
                """, {
                    'project_name': self.project_name,
                    'doc_id': doc_id,
                    'section_id': section_id,
                    'title': title,
                    'level': level
                })
                
                record = result.single()
                if record:
                    self.logger.info(f"Created/verified section: {record['id']} (level {record['level']})")
                    return Section(
                        id=record['id'],
                        number=section_id.split('.')[-1],
                        level=record['level']
                    )
                else:
                    self.logger.warning(f"Failed to create section {section_id}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Error creating section {section_id}: {str(e)}")
            return None
    
    @handle_exception
    def load_document_hierarchy(self, file_path: str, doc_type: str) -> Optional[Document]:
        """
        Load document hierarchy from XML file.
        
        Args:
            file_path: Path to XML file
            doc_type: Type of document (SOURCE or TARGET)
            
        Returns:
            Document object if successful, None otherwise
        """
        try:
            self.logger.info(f"Loading {doc_type} document...")
            self.logger.info(f"Loading document hierarchy from {file_path}")
            
            # Create project if it doesn't exist
            project = self.create_project()
            self.logger.info(f"Created/verified project: {project.name}")
            
            # Parse requirements first
            requirements = self.requirements_loader.parse_requirements(file_path)
            if not requirements:
                raise ValueError("No requirements found in file")
            
            self.logger.debug(f"Found {len(requirements)} requirements in file")
            
            # Determine document ID based on type
            doc_id = f"{self.project_name}_{doc_type.upper()}"
            
            # Create document with CONTAINS relationship to project
            document = self.create_document(
                doc_id=doc_id,
                title=f"{doc_type.upper()} Document for {self.project_name}",
                doc_type=doc_type.upper()
            )
            
            if not document:
                raise ValueError(f"Failed to create document {doc_id}")
            
            # Process each requirement
            for req in requirements:
                self._process_requirement(document, req, doc_type.upper())
                
            return document
            
        except Exception as e:
            self.logger.error(f"Error in load_document_hierarchy: {str(e)}")
            self.logger.exception("Detailed error trace:")
            raise
    
    @handle_exception
    def _process_requirement(self, document: Document, requirement: Dict, doc_type: str):
        """
        Process requirement and create it under the document.
        
        Args:
            document: Parent document
            requirement: Requirement data dictionary
            doc_type: Document type (SOURCE/TARGET)
        """
        try:
            # Create requirement directly under document
            self.requirements_loader.create_requirement(
                parent_id=document.id,
                content=requirement['content'],
                req_id=requirement['id'],
                req_type=doc_type,
                level=1,
                project_name=self.project_name
            )
            
        except Exception as e:
            self.logger.error(f"Error processing requirement {requirement['id']}: {str(e)}")
            raise
    
    @handle_exception
    def load_and_store_documents(self, source_file: str, target_file: str) -> Tuple[Document, Document]:
        """
        Load and store both source and target documents.
        
        Args:
            source_file: Path to source document file
            target_file: Path to target document file
            
        Returns:
            Tuple of (source_document, target_document)
        """
        try:
            # Create project if it doesn't exist
            project = self.create_project()
            self.logger.info(f"Created/verified project: {project.name}")
            
            # Load source document
            source_doc = self.load_document_hierarchy(source_file, 'SOURCE')
            if not source_doc:
                raise ValueError("Failed to load source document")
            
            # Load target document
            target_doc = self.load_document_hierarchy(target_file, 'TARGET')
            if not target_doc:
                raise ValueError("Failed to load target document")
            
            return source_doc, target_doc
            
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise
    
    @handle_exception
    def create_ground_truth_links(self, answer_set: List[Tuple[str, str]]) -> None:
        """
        Create ground truth links between requirements based on answer set.
        
        Args:
            answer_set: List of (source_id, target_id) tuples
        """
        try:
            with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                for source_id, target_id in answer_set:
                    session.run("""
                        MATCH (p:Project {name: $project_name})
                        MATCH (s:Requirement {id: $source_id, project: $project_name})
                        MATCH (t:Requirement {id: $target_id, project: $project_name})
                        MERGE (s)-[r:GROUND_TRUTH]->(t)
                        SET r.project = $project_name,
                            r.timestamp = datetime()
                    """, {
                        'project_name': self.project_name,
                        'source_id': source_id,
                        'target_id': target_id
                    })
                self.logger.info(f"Created {len(answer_set)} ground truth links")
                
        except Exception as e:
            self.logger.error(f"Error creating ground truth links: {str(e)}")
            raise
    
    @handle_exception
    def _parse_document_id(self, req_id: str) -> Dict[str, Any]:
        """
        Parse requirement ID to extract document structure.
        
        Args:
            req_id: Requirement ID to parse
            
        Returns:
            Dict containing parsed components
        """
        try:
            self.logger.debug(f"Parsing requirement ID: {req_id}")
            
            # Split ID into components
            parts = req_id.split('.')
            
            if not parts:
                raise ValueError(f"Empty requirement ID: {req_id}")
            
            # First part is document type
            doc_type = parts[0]
            
            if not doc_type:
                raise ValueError(f"Empty document type in ID: {req_id}")
            
            self.logger.debug(f"Document type: {doc_type}")
            
            # Check if it's a flat structure (no dots)
            if len(parts) == 1:
                self.logger.debug(f"Flat structure detected for ID: {req_id}")
                return {
                    'doc_type': doc_type,
                    'is_flat': True,
                    'section_id': None,
                    'subsection_ids': []
                }
            
            # For hierarchical structure
            section_id = parts[1]
            subsection_ids = parts[2:-1] if len(parts) > 3 else []
            
            self.logger.debug(f"Hierarchical structure detected for ID: {req_id}")
            self.logger.debug(f"Section ID: {section_id}")
            self.logger.debug(f"Subsection IDs: {subsection_ids}")
            
            return {
                'doc_type': doc_type,
                'is_flat': False,
                'section_id': section_id,
                'subsection_ids': subsection_ids
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing requirement ID {req_id}: {str(e)}")
            self.logger.exception("Detailed error trace:")
            raise
    
    @handle_exception
    def _create_document_hierarchy(self, file_path: str, doc_type: str) -> Document:
        """
        Create document hierarchy from XML file.
        
        Args:
            file_path: Path to XML file
            doc_type: Type of document (SOURCE or TARGET)
            
        Returns:
            Document object
        """
        try:
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Create document
            doc_id = root.get('id', f"{doc_type}_DOC")
            document = self.create_document(doc_id, f"{doc_type} Document", doc_type)
            
            # Process sections
            for section in root.findall('.//section'):
                self._process_section(section, document, doc_type)
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error creating document hierarchy: {str(e)}")
            raise
    
    @handle_exception
    def debug_requirements_hierarchy(self) -> None:
        """Debug the requirements hierarchy in Neo4j"""
        try:
            with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH path = (p)-[:CONTAINS*]->(r:Requirement)
                    RETURN path
                """, {'project_name': self.project_name})
                
                for record in result:
                    path = record['path']
                    self.logger.info(f"Path: {path}")
                    
        except Exception as e:
            self.logger.error(f"Error debugging requirements hierarchy: {str(e)}")
            raise