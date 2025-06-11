"""
Neo4j graph database management for requirements traceability
"""

import os
import time
import logging
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm.notebook import tqdm
from typing import List, Dict, Tuple, Optional, Any
from neo4j import GraphDatabase
from datetime import datetime

from ..logger import setup_logging, handle_exception, DebugTimer
from ..analyzers.sentence_transformer import SentenceTransformerAnalyzer
from .models import RequirementNode
from ..loaders.requirements_loader import RequirementsLoader
from ..utils.cuda import initialize_cuda, cleanup_cuda, cleanup_resources
from .neo4j_client import Neo4jClient

# Set up logging
logger = setup_logging("neo4j-operations", logging.DEBUG)

class RequirementsTraceabilityGraph:
    """
    Manages requirement traceability relationships in Neo4j graph database
    Integrates with sentence transformers and TF-IDF for similarity analysis
    """
    
    def __init__(self, analyzer: SentenceTransformerAnalyzer, alpha: float = 0.2, 
                 answer_set: Optional[List[Tuple[str, str]]] = None,
                 project_name: Optional[str] = None):
        """Initialize with environment-based configuration"""
        # Database connection parameters
        self.uri = os.getenv('NEO4J_URI', 'bolt://172.22.156.237:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Project name from parameter or environment
        self.project_name = project_name or os.getenv('PROJECT_NAME')
        if not self.project_name:
            raise ValueError("Project name must be provided either through parameter or PROJECT_NAME environment variable")
        
        # Neo4j optimization parameters from environment
        self.batch_size = int(os.getenv('NEO4J_BATCH_SIZE', '1000'))
        self.max_connections = int(os.getenv('NEO4J_MAX_CONNECTIONS', '5'))
        self.transaction_size = int(os.getenv('NEO4J_TRANSACTION_SIZE', '100'))
        
        # Analysis components
        self.analyzer = analyzer
        self.alpha = alpha
        self.answer_set = answer_set
        self.driver = None
        
        # Initialize Neo4j client
        self.client = Neo4jClient(
            uri=self.uri,
            username=self.user,
            password=self.password,
            database=self.database
        )
        
        # Add loader initialization
        self.loader = RequirementsLoader()
        
        # Initialize connection and schema
        self._connect()
        self._setup_schema()
        
        # Create project if it doesn't exist
        self._create_project()

    def _create_project(self):
        """Create project node if it doesn't exist"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    MERGE (p:Project {name: $project_name})
                    SET p.description = $description,
                        p.created_at = datetime()
                """, {
                    'project_name': self.project_name,
                    'description': f"Project {self.project_name}"
                })
                logger.info(f"Created/verified project: {self.project_name}")
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            raise

    def _connect(self):
        """Establish connection to Neo4j database with environment-based settings"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=self.max_connections,
                connection_acquisition_timeout=60,
                max_transaction_retry_time=30,
                connection_timeout=30,
                resolver=None,
                encrypted=False,
                trust="TRUST_ALL_CERTIFICATES"
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
    
    def reconnect(self):
        """
        Reconnect to the database by closing existing connection and establishing a new one
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        try:
            # Close existing connection if any
            if self.driver:
                try:
                    self.driver.close()
                except Exception as e:
                    logger.warning(f"Error closing existing connection: {str(e)}")
                self.driver = None
                
            # Create new connection
            return self._connect()
            
        except Exception as e:
            logger.error(f"Error reconnecting to database: {str(e)}")
            return False
    
    def close(self):
        """Close all database connections"""
        if self.driver:
            self.driver.close()
        if self.client:
            self.client.close()
            
    def _setup_schema(self):
        """Set up Neo4j database schema and constraints"""
        try:
            with self.driver.session(database=self.database) as session:
                # Drop existing constraints
                session.run("DROP CONSTRAINT requirement_id_unique IF EXISTS")
                session.run("DROP CONSTRAINT section_id_unique IF EXISTS")
                session.run("DROP CONSTRAINT subsection_id_unique IF EXISTS")
                session.run("DROP CONSTRAINT document_id_unique IF EXISTS")
                session.run("DROP CONSTRAINT requirement_project_unique IF EXISTS")

                # Create new constraints using path for uniqueness
                constraints = [
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT requirement_project_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE (r.id, r.project) IS UNIQUE",
                    "CREATE CONSTRAINT section_project_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.id, s.project) IS UNIQUE",
                    "CREATE CONSTRAINT subsection_project_unique IF NOT EXISTS FOR (s:Subsection) REQUIRE (s.id, s.project) IS UNIQUE"
                ]
                
                for constraint in constraints:
                    session.run(constraint)

        except Exception as e:
            logger.error(f"Error setting up database schema: {str(e)}")
            raise

    @handle_exception
    def parse_requirement_id(self, req_id: str) -> Dict[str, Any]:
        """
        Parse a requirement ID into its hierarchical or flat components.
        
        Args:
            req_id: Requirement ID (e.g., 'SRS5.19.1.2' or 'CC140')
            
        Returns:
            Dict containing:
                - doc_type: Document type (e.g., 'SRS' or 'CC')
                - is_flat: Boolean indicating if this is a flat structure
                - section_id: Section ID (if hierarchical)
                - subsection_ids: List of subsection IDs (if hierarchical)
                - req_num: Requirement number
                - full_id: Complete ID
        """
        try:
            # Check if this is a flat structure (no dots)
            is_flat = '.' not in req_id
            
            if is_flat:
                # Extract document type (letters) and requirement number (digits)
                doc_type = ''.join(c for c in req_id if c.isalpha())
                req_num = ''.join(c for c in req_id if c.isdigit())
                
                return {
                    'doc_type': doc_type,
                    'is_flat': True,
                    'section_id': None,
                    'subsection_ids': [],
                    'req_num': req_num,
                    'full_id': req_id
                }
            else:
                # Handle hierarchical structure as before
                doc_type = ''.join(c for c in req_id if c.isalpha())
                numeric_parts = req_id[len(doc_type):].split('.')
                
                section = numeric_parts[0]
                subsections = numeric_parts[1:-1] if len(numeric_parts) > 2 else []
                req_num = numeric_parts[-1]
                
                section_id = f"{doc_type}{section}"
                subsection_ids = []
                current_id = section_id
                
                for sub in subsections:
                    current_id = f"{current_id}.{sub}"
                    subsection_ids.append(current_id)
                    
                full_id = f"{current_id}.{req_num}" if subsections else f"{section_id}.{req_num}"
                
                return {
                    'doc_type': doc_type,
                    'is_flat': False,
                    'section_id': section_id,
                    'subsection_ids': subsection_ids,
                    'req_num': req_num,
                    'full_id': full_id
                }
                
        except Exception as e:
            logger.error(f"Error parsing requirement ID {req_id}: {str(e)}")
            raise

    def get_requirement_path(self, req_id: str) -> str:
        """
        Get the full path for a requirement ID.
        
        Args:
            req_id: Requirement ID to parse
            
        Returns:
            Full requirement path
        """
        parsed = self.parse_requirement_id(req_id)
        return parsed['full_id']

    @handle_exception
    def create_requirement_node(self, tx, req_data: dict, doc_type: str):
        """
        Create a requirement node in Neo4j with minimal properties.
        
        Args:
            tx: Neo4j transaction
            req_data: Dictionary containing requirement data
            doc_type: Type of document ('SOURCE' or 'TARGET')
        """
        # Create the requirement node with minimal properties
        query = """
        MATCH (p:Project {name: $project_name})
        MERGE (r:Requirement {id: $id, project: $project_name})
        SET r.content = $content,
            r.path = $id,
            r.type = $type
        """
        
        tx.run(query, {
            'id': req_data['id'],
            'content': req_data['content'],
            'type': doc_type,
            'project_name': self.project_name
        })

    @handle_exception
    def create_document_hierarchy(self, tx, requirements: list, doc_type: str):
        """Create document hierarchy from requirements list"""
        try:
            # Get document type from first requirement
            first_req = requirements[0]
            parsed = self.parse_requirement_id(first_req['id'])
            doc_id = parsed['doc_type']
            
            # Create document node and link to project
            tx.run("""
                MATCH (p:Project {name: $project_name})
                MERGE (d:Document {
                    id: $id,
                    type: $type,
                    name: $id,
                    project: $project_name
                })
                MERGE (p)-[:CONTAINS]->(d)
            """, {
                'id': doc_id,
                'type': doc_type,
                'project_name': self.project_name
            })
            logger.debug(f"Created document node: {doc_id} for project {self.project_name}")

            # Process each requirement
            for req in requirements:
                parsed = self.parse_requirement_id(req['id'])
                
                if parsed['is_flat']:
                    # Create requirement directly under document for flat structure
                    tx.run("""
                        MATCH (p:Project {name: $project_name})
                        MATCH (d:Document {id: $doc_id, project: $project_name})
                        MERGE (r:Requirement {id: $req_id, project: $project_name})
                        SET r.content = $content,
                            r.type = $doc_type,
                            r.name = $req_id,
                            r.level = 1
                        MERGE (d)-[:CONTAINS]->(r)
                    """, {
                        'doc_id': parsed['doc_type'],
                        'req_id': req['id'],
                        'content': req['content'],
                        'doc_type': doc_type,
                        'project_name': self.project_name
                    })
                    logger.debug(f"Created flat requirement: {req['id']} for project {self.project_name}")
                else:
                    # Handle hierarchical structure
                    # Create section if it doesn't exist
                    tx.run("""
                        MATCH (p:Project {name: $project_name})
                        MATCH (d:Document {id: $doc_id, project: $project_name})
                        MERGE (s:Section {id: $section_id, project: $project_name})
                        SET s.name = $section_id
                        MERGE (d)-[:CONTAINS]->(s)
                    """, {
                        'doc_id': parsed['doc_type'],
                        'section_id': parsed['section_id'],
                        'project_name': self.project_name
                    })
                    
                    # Create subsections
                    current_id = parsed['section_id']
                    for subsection_id in parsed['subsection_ids']:
                        tx.run("""
                            MATCH (p:Project {name: $project_name})
                            MATCH (parent {id: $parent_id, project: $project_name})
                            MERGE (s:Subsection {id: $subsection_id, project: $project_name})
                            SET s.name = $subsection_id
                            MERGE (parent)-[:CONTAINS]->(s)
                        """, {
                            'parent_id': current_id,
                            'subsection_id': subsection_id,
                            'project_name': self.project_name
                        })
                        current_id = subsection_id
                    
                    # Create requirement node
                    tx.run("""
                        MATCH (p:Project {name: $project_name})
                        MATCH (parent {id: $parent_id, project: $project_name})
                        MERGE (r:Requirement {id: $req_id, project: $project_name})
                        SET r.content = $content,
                            r.type = $doc_type,
                            r.name = $req_id
                        MERGE (parent)-[:CONTAINS]->(r)
                    """, {
                        'parent_id': current_id,
                        'req_id': req['id'],
                        'content': req['content'],
                        'doc_type': doc_type,
                        'project_name': self.project_name
                    })
                    logger.debug(f"Created requirement: {req['id']} for project {self.project_name}")

            logger.info(f"Successfully created hierarchy for {doc_type} document in project {self.project_name}")
            
        except Exception as e:
            logger.error(f"Error creating document hierarchy: {str(e)}")
            logger.debug(f"Sample requirement: {requirements[0] if requirements else 'No requirements'}")
            raise

    @handle_exception
    def load_requirements(self, source_file: str, target_file: str, answer_file: str) -> bool:
        """Load requirements into Neo4j with full document hierarchy"""
        try:
            # Parse requirements
            source_reqs = self.loader.parse_requirements(source_file)
            target_reqs = self.loader.parse_requirements(target_file)
            
            # Create Neo4j session
            with self.driver.session(database=self.database) as session:
                # Create document hierarchies in transactions
                tx = session.begin_transaction()
                try:
                    # Create source document hierarchy
                    self.create_document_hierarchy(tx, source_reqs, 'SOURCE')
                    
                    # Create target document hierarchy
                    self.create_document_hierarchy(tx, target_reqs, 'TARGET')
                    
                    # Commit transaction
                    tx.commit()
                    logger.info(f"Successfully created document hierarchies")
                    logger.info(f"Loaded {len(source_reqs)} source requirements")
                    logger.info(f"Loaded {len(target_reqs)} target requirements")
                    
                    # Verify the hierarchy was created correctly
                    if not self.verify_database_state():
                        raise ValueError("Database verification failed after loading requirements")
                    
                    return True
                    
                except Exception as e:
                    tx.rollback()
                    logger.error(f"Error creating document hierarchies: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error loading requirements: {str(e)}")
            logger.exception("Detailed error trace:")
            return False

    @handle_exception
    def get_requirements(self, req_type: str) -> List[Dict]:
        """
        Get requirements of specified type from database
        
        Parameters:
            req_type (str): 'SOURCE' or 'TARGET'
            
        Returns:
            List[Dict]: List of requirement dictionaries with id and description
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (r:Requirement {artifact_type: $type})
                    RETURN r.id as id, r.content as description
                    ORDER BY r.id
                """, type=req_type)
                
                requirements = [dict(record) for record in result]
                logger.debug(f"Retrieved {len(requirements)} {req_type} requirements")
                return requirements
                
        except Exception as e:
            logger.error(f"Error getting {req_type} requirements: {str(e)}")
            return []

    @handle_exception
    def _store_similarity_results(self, relationships: List[Dict[str, Any]]):
        """
        Store all similarity relationships in Neo4j using full paths
        
        Args:
            relationships: List of dictionaries containing source_id, target_id, and similarity
        """
        try:
            with self.driver.session(database=self.database) as session:
                total_batches = len(relationships) // self.batch_size + (1 if len(relationships) % self.batch_size else 0)
                logger.info(f"Processing {len(relationships)} relationships in {total_batches} batches")
                
                total_created = 0
                total_skipped = 0
                
                # Store in batches
                batch_size = self.batch_size
                for i in range(0, len(relationships), batch_size):
                    batch = relationships[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    
                    # Log batch details
                    logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} relationships)")
                    
                    # Create relationships using paths and count results
                    result = session.run("""
                        UNWIND $batch as rel
                        MATCH (p:Project {name: $project_name})
                        MATCH (s:Requirement {path: rel.source_id})
                        MATCH (t:Requirement {path: rel.target_id})
                        WHERE s <> t  // Prevent self-referential relationships
                        AND EXISTS((p)-[:CONTAINS*]->(s))
                        AND EXISTS((p)-[:CONTAINS*]->(t))
                        MERGE (s)-[r:SIMILAR_TO]->(t)
                        SET r.similarity = rel.similarity,
                            r.timestamp = datetime(),
                            r.project = $project_name
                        RETURN count(*) as created
                    """, batch=batch, project_name=self.project_name)
                    
                    created = result.single()["created"]
                    skipped = len(batch) - created
                    
                    total_created += created
                    total_skipped += skipped
                    
                    logger.debug(f"Batch {batch_num}: Created {created}, Skipped {skipped} relationships")
                    
                logger.info(f"Completed storing relationships:")
                logger.info(f"- Total created: {total_created}")
                logger.info(f"- Total skipped: {total_skipped}")
                logger.info(f"- Total processed: {total_created + total_skipped}")
                
                if total_skipped > 0:
                    logger.warning(f"Skipped {total_skipped} relationships due to:")
                    logger.warning("- Self-referential relationships")
                    logger.warning("- Requirements not in the same project")
                    logger.warning("- Missing source/target requirements")
                    
        except Exception as e:
            logger.error(f"Error storing similarity results: {str(e)}")
            raise

    def verify_database_state(self) -> bool:
        """Verify the database state and hierarchy"""
        try:
            with self.driver.session(database=self.database) as session:
                # Check node counts
                counts = session.run("""
                    MATCH (n)
                    WITH labels(n) as type, count(n) as count
                    RETURN type, count
                """).data()
                
                # Log counts
                logger.info("Database node counts:")
                for record in counts:
                    logger.info(f"- {record['type']}: {record['count']}")
                
                # Verify hierarchy
                hierarchy_check = session.run("""
                    MATCH p=(d:Document)-[:CONTAINS*]->(r:Requirement)
                    RETURN count(p) as paths
                """).single()
                
                path_count = hierarchy_check['paths']
                logger.info(f"Found {path_count} complete paths from Document to Requirement")
                
                # Verify relationships
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                """).data()
                
                logger.info("Relationship counts:")
                for record in rel_counts:
                    logger.info(f"- {record['type']}: {record['count']}")
                
                return path_count > 0
                
        except Exception as e:
            logger.error(f"Error verifying database state: {str(e)}")
            return False
        
    def load_answer_set_links(self, answer_file: str):
        """
        Load ground truth links from answer set into Neo4j
        
        Parameters:
            answer_file (str): Path to answer set XML file
        """
        try:
            logger.info(f"Loading answer set links from {answer_file}")
            answer_links = self.analyzer.loader.parse_answer_set(answer_file)
            
            if not answer_links:
                logger.warning("No answer set links found")
                return
                
            logger.info(f"Found {len(answer_links)} ground truth links")
            
            # Create links in batches
            with self.driver.session(database=self.database) as session:
                batch_size = self.batch_size
                total_created = 0
                
                for i in range(0, len(answer_links), batch_size):
                    batch = answer_links[i:i + batch_size]
                    
                    # Create links in batch
                    query = """
                    UNWIND $links as link
                    MATCH (s:Requirement {id: link.source})
                    MATCH (t:Requirement {id: link.target})
                    MERGE (s)-[r:GROUND_TRUTH]->(t)
                    SET r.confidence = 1.0,
                        r.timestamp = datetime()
                    RETURN count(r) as created
                    """
                    
                    result = session.run(query, links=[
                        {"source": src, "target": tgt} 
                        for src, tgt in batch
                    ])
                    
                    created = result.single()["created"]
                    total_created += created
                    
                    if (i + batch_size) % 500 == 0:
                        logger.debug(f"Created {total_created} ground truth links so far...")
                
                # Verify links were created
                verification = session.run("""
                    MATCH ()-[r:GROUND_TRUTH]->()
                    RETURN count(r) as link_count
                """).single()
                
                actual_count = verification["link_count"]
                logger.info(f"Created {actual_count} ground truth links in database")
                
                if actual_count != len(answer_links):
                    logger.warning(
                        f"Mismatch between expected ({len(answer_links)}) and "
                        f"actual ({actual_count}) ground truth links"
                    )
                    
        except Exception as e:
            logger.error(f"Error loading answer set links: {str(e)}")
            logger.exception("Detailed error trace:")
            raise

    @handle_exception
    def validate_requirement_id(self, req_id: str) -> bool:
        """
        Validate requirement ID format
        
        Args:
            req_id: Requirement ID to validate
            
        Returns:
            bool: True if valid format
        """
        try:
            # Check basic format
            if not req_id or not isinstance(req_id, str):
                logger.warning(f"Invalid requirement ID: {req_id}")
                return False
            
            # Parse and validate components
            parsed = self.parse_requirement_id(req_id)
            if not parsed:
                return False
            
            # Validate document type
            if parsed['doc_type'] not in ['SRS', 'DPUSDS']:
                logger.warning(f"Invalid document type in ID {req_id}: {parsed['doc_type']}")
                return False
            
            # Validate section number
            if parsed['section'] != '5':
                logger.warning(f"Invalid section number in ID {req_id}: {parsed['section']}")
                return False
            
            # Validate subsection format
            for subsection in parsed['subsections']:
                if not str(subsection).isdigit():
                    logger.warning(f"Invalid subsection in ID {req_id}: {subsection}")
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating requirement ID {req_id}: {str(e)}")
            return False

    @handle_exception
    def load_requirement_metadata(self, tx, req_data: dict, doc_type: str):
        """
        Load additional requirement metadata from XML
        
        Args:
            tx: Neo4j transaction
            req_data: Requirement data dictionary
            doc_type: Document type (SOURCE/TARGET)
        """
        try:
            # Extract metadata fields
            metadata = {
                'parent_id': req_data.get('parent_id', ''),
                'collection_info': req_data.get('collection_info', {}),
                'artifact_info': req_data.get('artifact_info', {})
            }
            
            # Update requirement node with metadata
            query = """
            MATCH (r:Requirement {id: $req_id})
            SET r += {
                parent_id: $parent_id,
                collection_name: $collection_name,
                collection_version: $collection_version,
                collection_description: $collection_description,
                artifact_type: $artifact_type,
                metadata: $metadata
            }
            """
            
            tx.run(query, {
                'req_id': req_data['id'],
                'parent_id': metadata['parent_id'],
                'collection_name': metadata['collection_info'].get('name', ''),
                'collection_version': metadata['collection_info'].get('version', ''),
                'collection_description': metadata['collection_info'].get('description', ''),
                'artifact_type': doc_type,
                'metadata': metadata
            })
            
            logger.debug(f"Added metadata to requirement {req_data['id']}")
            
        except Exception as e:
            logger.error(f"Error loading metadata for requirement {req_data['id']}: {str(e)}")
            raise

    @handle_exception
    def log_database_metrics(self):
        """Log detailed database metrics and statistics"""
        try:
            with self.driver.session(database=self.database) as session:
                metrics = {}
                
                # Count requirements by type for current project
                result = session.run("""
                    MATCH (p:Project {name: $project_name})-[:CONTAINS*]->(r:Requirement)
                    RETURN r.type as type, count(r) as count
                """, project_name=self.project_name)
                metrics['requirements'] = {r['type']: r['count'] for r in result}
                
                # Count relationships by type for current project
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH (s)-[r]->(t)
                    WHERE EXISTS((p)-[:CONTAINS*]->(s))
                    AND EXISTS((p)-[:CONTAINS*]->(t))
                    RETURN type(r) as type, count(r) as count
                """, project_name=self.project_name)
                metrics['relationships'] = {r['type']: r['count'] for r in result}
                
                # Get hierarchy depth for current project
                result = session.run("""
                    MATCH (p:Project {name: $project_name})
                    MATCH path = (d:Document)-[:CONTAINS*]->(r:Requirement)
                    WHERE EXISTS((p)-[:CONTAINS]->(d))
                    RETURN max(length(path)) as max_depth
                """, project_name=self.project_name)
                metrics['max_hierarchy_depth'] = result.single()['max_depth']
                
                # Log metrics
                logger.info(f"\nDatabase Metrics for project {self.project_name}:")
                logger.info("-" * 40)
                logger.info("Requirements:")
                for type_, count in metrics['requirements'].items():
                    logger.info(f"  {type_}: {count}")
                
                logger.info("\nRelationships:")
                for type_, count in metrics['relationships'].items():
                    logger.info(f"  {type_}: {count}")
                
                logger.info(f"\nMax Hierarchy Depth: {metrics['max_hierarchy_depth']}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error logging database metrics: {str(e)}")
            return None

    def get_source_requirements(self, project_name: str) -> List[Dict[str, Any]]:
        """Get source requirements for a specific project"""
        return self.client.get_requirements_by_type("SOURCE", project_name)

    def get_target_requirements(self, project_name: str) -> List[Dict[str, Any]]:
        """Get target requirements for a specific project"""
        return self.client.get_requirements_by_type("TARGET", project_name)

    @handle_exception
    def compute_tfidf_similarities(self) -> bool:
        """
        Compute TF-IDF similarities for all requirement pairs and store in Neo4j.
        Returns:
            bool: True if successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Get requirements for current project
                source_reqs = self.get_source_requirements(self.project_name)
                target_reqs = self.get_target_requirements(self.project_name)
                
                if not source_reqs or not target_reqs:
                    logger.error(f"No requirements found in database for project {self.project_name}")
                    return False

                # Compute TF-IDF similarities
                logger.info(f"Computing TF-IDF similarities for project {self.project_name}...")
                tfidf_similarities = self.analyzer.compute_batch_tfidf_similarities(
                    [req['content'] for req in source_reqs],
                    [req['content'] for req in target_reqs]
                )

                # Store relationships in batches
                store_query = """
                UNWIND $batch as rel
                MATCH (p:Project {name: $project_name})
                MATCH (s:Requirement {id: rel.source_id})
                MATCH (t:Requirement {id: rel.target_id})
                WHERE EXISTS((p)-[:CONTAINS*]->(s))
                AND EXISTS((p)-[:CONTAINS*]->(t))
                MERGE (s)-[r:SIMILAR_TO {model: 'TF-IDF'}]->(t)
                SET r.similarity = rel.similarity_score,
                    r.timestamp = datetime(),
                    r.project = $project_name
                """

                batch_size = 1000
                relationships = []
                
                # Create relationships for all pairs
                for i, source_req in enumerate(source_reqs):
                    for j, target_req in enumerate(target_reqs):
                        score = float(tfidf_similarities[i, j])
                        relationships.append({
                            'source_id': source_req['id'],
                            'target_id': target_req['id'],
                            'similarity_score': score
                        })

                # Store in batches
                total_stored = 0
                for i in range(0, len(relationships), batch_size):
                    batch = relationships[i:i + batch_size]
                    session.run(store_query, {
                        'batch': batch,
                        'project_name': self.project_name
                    })
                    total_stored += len(batch)
                    logger.debug(f"Stored {total_stored} TF-IDF relationships")

                logger.info(f"Successfully stored {total_stored} TF-IDF relationships for project {self.project_name}")
                return True

        except Exception as e:
            logger.error(f"Error computing TF-IDF similarities: {str(e)}")
            raise

    @handle_exception
    def cleanup_database(self):
        """Cleanup database and verify state"""
        try:
            with self.driver.session(database=self.database) as session:
                # Log initial state
                logger.info("Database state before cleanup:")
                self.log_database_metrics()
                
                # Remove invalid relationships
                session.run("""
                    MATCH ()-[r:SIMILAR_TO]->()
                    WHERE r.similarity IS NULL
                    DELETE r
                """)
                
                # Remove orphaned nodes
                session.run("""
                    MATCH (r:Requirement)
                    WHERE NOT exists((r)<-[:CONTAINS]-())
                    AND NOT exists((r)-[:CONTAINS]->())
                    DELETE r
                """)
                
                # Log final state
                logger.info("\nDatabase state after cleanup:")
                self.log_database_metrics()
                
        except Exception as e:
            logger.error(f"Error during database cleanup: {str(e)}")
            raise

    @handle_exception
    def get_parent_id(self, parsed_id: dict) -> str:
        """
        Generate parent ID from parsed requirement ID components.
        
        Args:
            parsed_id (dict): Dictionary containing parsed ID components:
                {
                    'doc_type': Document type (e.g. 'SRS' or 'DPUSDS'),
                    'section': Main section number,
                    'subsections': List of subsection numbers
                }
        
        Returns:
            str: Parent requirement ID, or empty string if no parent exists
        """
        try:
            # If no subsections, this is a top-level requirement
            if not parsed_id['subsections']:
                return ''
            
            # Remove last subsection to get parent ID
            parent_subsections = parsed_id['subsections'][:-1]
            
            # Construct parent ID
            if parent_subsections:
                return f"{parsed_id['doc_type']}{parsed_id['section']}.{'.'.join(parent_subsections)}"
            else:
                return f"{parsed_id['doc_type']}{parsed_id['section']}"
            
        except Exception as e:
            logger.error(f"Error generating parent ID from {parsed_id}: {str(e)}")
            return ''

    @handle_exception
    def _create_similarity_relationships_batch(self, relationships: List[Dict]):
        """Create similarity relationships in batch"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    UNWIND $rels as rel
                    MATCH (s:Requirement {id: rel.source_id})
                    MATCH (t:Requirement {id: rel.target_id})
                    MERGE (s)-[r:SIMILAR_TO]->(t)
                    SET r.similarity = rel.similarity_score,
                        r.model_name = rel.model_name,
                        r.alpha = rel.alpha,
                        r.threshold = rel.threshold,
                        r.timestamp = datetime()
                """, rels=relationships)
                
        except Exception as e:
            logger.error(f"Error creating similarity relationships: {str(e)}")
            raise

    @handle_exception
    def evaluate_results(self, predicted_links: List[Tuple[str, str]], 
                        ground_truth: List[Tuple[str, str]], 
                        threshold: float) -> Dict:
        """
        Evaluate predicted links against ground truth
        
        Parameters:
            predicted_links: List of (source_id, target_id) tuples for predicted links
            ground_truth: List of (source_id, target_id) tuples for ground truth
            threshold: Threshold value used
            
        Returns:
            Dict containing evaluation metrics
        """
        try:
            # Convert to sets for efficient comparison
            predicted_set = set(predicted_links)
            ground_truth_set = set(ground_truth)
            
            # Calculate metrics
            tp = len(predicted_set.intersection(ground_truth_set))
            fp = len(predicted_set - ground_truth_set)
            fn = len(ground_truth_set - predicted_set)
            
            # Get total possible combinations for true negatives
            total_source = len({src for src, _ in ground_truth})
            total_target = len({tgt for _, tgt in ground_truth})
            total_possible = total_source * total_target
            tn = total_possible - (tp + fp + fn)
            
            # Calculate rates
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_possible if total_possible > 0 else 0
            
            return {
                "metrics": {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "accuracy": accuracy
                },
                "counts": {
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                    "total_possible": total_possible
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating results: {str(e)}")
            return {}

    def ensure_indexes(self):
        """Ensure all necessary indexes exist for optimal performance"""
        with self.driver.session(database=self.database) as session:
            # Create indexes for faster lookups
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (r:Requirement) ON (r.id);
                CREATE INDEX IF NOT EXISTS FOR (r:Requirement) ON (r.type);
                CREATE INDEX IF NOT EXISTS FOR ()-[r:SIMILARITY_SCORE]-() ON (r.algorithm);
            """)
            logger.info("Database indexes created/verified")

    @handle_exception
    def compute_sentence_transformer_similarities(self) -> bool:
        """
        Compute sentence transformer similarities for all requirement pairs and store in Neo4j.
        Returns:
            bool: True if successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Get requirements for current project
                source_reqs = self.get_source_requirements(self.project_name)
                target_reqs = self.get_target_requirements(self.project_name)
                
                if not source_reqs or not target_reqs:
                    logger.error(f"No requirements found in database for project {self.project_name}")
                    return False

                # Compute transformer similarities
                logger.info(f"Computing transformer similarities for project {self.project_name}...")
                transformer_similarities = self.analyzer.compute_sentence_transformer_similarities(
                    [req['content'] for req in source_reqs],
                    [req['content'] for req in target_reqs]
                )

                # Store relationships in batches
                store_query = """
                UNWIND $batch as rel
                MATCH (p:Project {name: $project_name})
                MATCH (s:Requirement {id: rel.source_id})
                MATCH (t:Requirement {id: rel.target_id})
                WHERE EXISTS((p)-[:CONTAINS*]->(s))
                AND EXISTS((p)-[:CONTAINS*]->(t))
                MERGE (s)-[r:SIMILAR_TO {model: $model_name}]->(t)
                SET r.similarity = rel.similarity_score,
                    r.timestamp = datetime(),
                    r.project = $project_name
                """

                batch_size = 1000
                relationships = []
                
                # Create relationships for all pairs
                for i, source_req in enumerate(source_reqs):
                    for j, target_req in enumerate(target_reqs):
                        score = float(transformer_similarities[i, j])
                        relationships.append({
                            'source_id': source_req['id'],
                            'target_id': target_req['id'],
                            'similarity_score': score
                        })

                # Store in batches
                total_stored = 0
                for i in range(0, len(relationships), batch_size):
                    batch = relationships[i:i + batch_size]
                    session.run(store_query, {
                        'batch': batch,
                        'project_name': self.project_name,
                        'model_name': self.analyzer.model_name
                    })
                    total_stored += len(batch)
                    logger.debug(f"Stored {total_stored} transformer relationships")

                logger.info(f"Successfully stored {total_stored} transformer relationships for project {self.project_name}")
                return True

        except Exception as e:
            logger.error(f"Error computing transformer similarities: {str(e)}")
            raise

    @handle_exception
    def cleanup_old_relationships(self):
        """Clean up any old relationship types"""
        try:
            with self.driver.session(database=self.database) as session:
                # Remove old TFIDF relationships
                session.run("""
                    MATCH ()-[r:TFIDF_SIMILAR]->()
                    DELETE r
                """)
                
                # Convert any remaining old relationships
                session.run("""
                    MATCH (s)-[r:TFIDF_SIMILAR]->(t)
                    WITH s, t, r.similarity as sim
                    MERGE (s)-[new:SIMILAR_TO {model: 'TF-IDF'}]->(t)
                    SET new.similarity = sim,
                        new.timestamp = datetime()
                    DELETE r
                """)
                
                logger.info("Cleaned up old relationship types")
                
        except Exception as e:
            logger.error(f"Error cleaning up old relationships: {str(e)}")
            raise