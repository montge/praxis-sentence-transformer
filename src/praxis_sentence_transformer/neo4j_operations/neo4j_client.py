"""
Neo4j client for database operations
"""

from typing import List, Dict, Any, Tuple
import logging
from neo4j import GraphDatabase, Driver
from ..logger import setup_logging, handle_exception
import pandas as pd

logger = setup_logging("Neo4jClient")

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j client with connection details.
        
        Args:
            uri (str): Neo4j database URI
            username (str): Database username
            password (str): Database password
            database (str): Database name, defaults to "neo4j"
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.project_name = None
        self.current_llm_model = None  # Track current LLM model
        logger.debug(f"Initialized Neo4j client with URI: {uri}, User: {username}, Database: {database}")
        self.connect()

    @handle_exception
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.debug(f"Attempting to connect to Neo4j at {self.uri}")
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Verify connection
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition UNWIND versions as version RETURN name, version, edition")
                record = result.single()
                if record:
                    logger.info(f"Connected to Neo4j {record['name']} {record['edition']} version {record['version']}")
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    @handle_exception
    def close(self) -> bool:
        """
        Close the Neo4j connection.

        This method ensures that the Neo4j driver connection is properly closed
        and logs the action. If the connection is already closed, it will log
        that the connection is not open.

        Returns:
            bool: True if the connection was closed successfully, False if there was no active connection.
        """
        logger.debug("Closing Neo4j driver connection")
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
            return True
        else:
            logger.warning("No active Neo4j connection to close")
            return False

    @handle_exception
    def setup_constraints(self) -> None:
        """Set up Neo4j constraints and indexes."""
        # First drop the old constraint if it exists
        drop_constraints = [
            "DROP CONSTRAINT requirement_id_unique IF EXISTS",
            "DROP CONSTRAINT document_id_unique IF EXISTS",
            "DROP CONSTRAINT project_name_unique IF EXISTS"
        ]
        
        # Create new constraints with composite keys
        constraints = [
            "CREATE CONSTRAINT requirement_project_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE (r.id, r.project) IS UNIQUE",
            "CREATE CONSTRAINT document_project_unique IF NOT EXISTS FOR (d:Document) REQUIRE (d.id, d.project) IS UNIQUE",
            "CREATE CONSTRAINT project_name_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX requirement_type_idx IF NOT EXISTS FOR (r:Requirement) ON (r.type)"
        ]
        
        with self.driver.session(database=self.database) as session:
            # Drop old constraints first
            for drop_constraint in drop_constraints:
                try:
                    session.run(drop_constraint)
                    logger.debug(f"Dropped constraint: {drop_constraint}")
                except Exception as e:
                    logger.warning(f"Error dropping constraint: {str(e)}")
            
            # Create new constraints
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.error(f"Error creating constraint: {str(e)}")
                    raise
            
            # Create indexes
            for index in indexes:
                try:
                    session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.error(f"Error creating index: {str(e)}")
                    raise

    @handle_exception
    def get_requirements_by_type(self, req_type: str, project_name: str) -> List[Dict[str, Any]]:
        """
        Get requirements of specified type for a specific project.
        
        Args:
            req_type: Either 'SOURCE' or 'TARGET'
            project_name: Name of the project to filter requirements
            
        Returns:
            List of requirement dictionaries
        """
        query = """
        MATCH (p:Project {name: $project_name})-[:CONTAINS*]->(r:Requirement)
        WHERE r.type = $req_type
        RETURN r.id as id, r.content as content, r.type as type
        ORDER BY r.id
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, project_name=project_name, req_type=req_type)
            requirements = [dict(record) for record in result]
            logger.debug(f"Retrieved {len(requirements)} {req_type.lower()} requirements for project {project_name}")
            if requirements:
                logger.debug(f"Sample {req_type.lower()} req: {requirements[0]}")
            return requirements

    @handle_exception
    def create_project(self, project_name: str) -> None:
        """Create a project node."""
        query = """
        MERGE (p:Project {name: $project_name})
        RETURN p
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, project_name=project_name)
            logger.info(f"Created project: {project_name}")

    @handle_exception
    def create_document(self, doc_id: str, doc_type: str, project_name: str) -> None:
        """Create a document node and link it to project."""
        query = """
        MATCH (p:Project {name: $project_name})
        MERGE (d:Document {id: $doc_id, type: $doc_type})
        MERGE (p)-[:CONTAINS]->(d)
        RETURN d
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, doc_id=doc_id, doc_type=doc_type, project_name=project_name)
            logger.info(f"Created document: {doc_id} ({doc_type})")

    @handle_exception
    def create_section(self, section_id: str, level: int, parent_id: str) -> None:
        """Create a section node and link it to parent."""
        query = """
        MATCH (parent) 
        WHERE parent.id = $parent_id
        MERGE (s:Section {id: $section_id, level: $level})
        MERGE (parent)-[:CONTAINS]->(s)
        RETURN s
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, section_id=section_id, level=level, parent_id=parent_id)
            logger.info(f"Created section: {section_id} (level {level})")

    @handle_exception
    def create_requirement(self, req_id: str, content: str, req_type: str, parent_id: str) -> None:
        """Create a requirement node and link it to parent."""
        query = """
        MATCH (parent) 
        WHERE parent.id = $parent_id
        MERGE (r:Requirement {id: $req_id, content: $content, type: $req_type})
        MERGE (parent)-[:CONTAINS]->(r)
        RETURN r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, req_id=req_id, content=content, req_type=req_type, parent_id=parent_id)
            logger.debug(f"Created requirement: {req_id}")

    @handle_exception
    def create_similarity_relationship(self, source_id: str, target_id: str, 
                                    similarity: float, model_name: str) -> None:
        """Create a similarity relationship between requirements."""
        query = """
        MATCH (p:Project {name: $project_name})
        MATCH (s:Requirement {id: $source_id})
        MATCH (t:Requirement {id: $target_id})
        WHERE EXISTS((p)-[:CONTAINS*]->(s))
        AND EXISTS((p)-[:CONTAINS*]->(t))
        MERGE (s)-[r:SIMILAR_TO {model: $model_name}]->(t)
        SET r.similarity = $similarity,
            r.project = $project_name,
            r.timestamp = datetime()
        RETURN r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, source_id=source_id, target_id=target_id, 
                       similarity=similarity, model_name=model_name,
                       project_name=self.project_name)

    @handle_exception
    def create_tfidf_relationship(self, source_id: str, target_id: str, 
                                similarity: float) -> None:
        """Create a TF-IDF similarity relationship between requirements."""
        query = """
        MATCH (p:Project {name: $project_name})
        MATCH (s:Requirement {id: $source_id})
        MATCH (t:Requirement {id: $target_id})
        WHERE EXISTS((p)-[:CONTAINS*]->(s))
        AND EXISTS((p)-[:CONTAINS*]->(t))
        MERGE (s)-[r:SIMILAR_TO {model: 'TF-IDF'}]->(t)
        SET r.similarity = $similarity,
            r.project = $project_name,
            r.timestamp = datetime()
        RETURN r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, source_id=source_id, target_id=target_id, 
                       similarity=similarity,
                       project_name=self.project_name)

    @handle_exception
    def create_ground_truth_link(self, source_id: str, target_id: str, project_name: str) -> None:
        """Create a ground truth link between requirements within the same project."""
        query = """
        MATCH (p:Project {name: $project_name})
        MATCH (s:Requirement {id: $source_id})
        MATCH (t:Requirement {id: $target_id})
        WHERE EXISTS((p)-[:CONTAINS*]->(s))
        AND EXISTS((p)-[:CONTAINS*]->(t))
        MERGE (s)-[r:GROUND_TRUTH]->(t)
        SET r.project = $project_name,
            r.timestamp = datetime()
        RETURN r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, 
                       source_id=source_id, 
                       target_id=target_id,
                       project_name=project_name)

    @handle_exception
    def cleanup_project(self, project_name: str) -> None:
        """
        Clean up all nodes and relationships for a specific project by recursively traversing
        the hierarchy from the project node and deleting relationships and nodes in the correct order.
        
        Args:
            project_name: Name of the project to clean up
        """
        try:
            # First collect all nodes connected to the project
            collect_nodes_query = """
            MATCH (p:Project {name: $project_name})
            OPTIONAL MATCH path = (p)-[:CONTAINS*]->(n)
            WITH COLLECT(n) + COLLECT(p) as nodes
            RETURN nodes
            """
            
            # Delete all relationships between collected nodes
            delete_rels_query = """
            MATCH (p:Project {name: $project_name})
            OPTIONAL MATCH path = (p)-[:CONTAINS*]->(n)
            WITH COLLECT(n) + COLLECT(p) as nodes
            UNWIND nodes as node
            MATCH (node)-[r]-()
            DELETE r
            """
            
            # Finally delete all nodes
            delete_nodes_query = """
            MATCH (p:Project {name: $project_name})
            OPTIONAL MATCH path = (p)-[:CONTAINS*]->(n)
            WITH COLLECT(n) + COLLECT(p) as nodes
            UNWIND nodes as node
            DELETE node
            """
            
            with self.driver.session(database=self.database) as session:
                # First verify we have nodes to delete
                result = session.run(collect_nodes_query, project_name=project_name)
                nodes = result.single()['nodes']
                if not nodes:
                    logger.warning(f"No nodes found for project {project_name}")
                    return
                
                logger.info(f"Found {len(nodes)} nodes to delete for project {project_name}")
                
                # Delete relationships first
                session.run(delete_rels_query, project_name=project_name)
                logger.info(f"Deleted all relationships for project {project_name}")
                
                # Then delete nodes
                session.run(delete_nodes_query, project_name=project_name)
                logger.info(f"Deleted all nodes for project {project_name}")
                
                logger.info(f"Successfully cleaned up project {project_name}")
                
        except Exception as e:
            logger.error(f"Error during project cleanup: {str(e)}")
            raise

    @handle_exception
    def get_similarity_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """Get similarity metrics from the database."""
        query = """
        MATCH (s:Requirement)-[r:SIMILAR_TO]->(t:Requirement)
        WHERE r.model = $model_name
        RETURN avg(r.similarity) as avg_sim, 
               min(r.similarity) as min_sim,
               max(r.similarity) as max_sim,
               count(r) as total_relationships
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, model_name=model_name)
            metrics = dict(result.single())
            logger.info(f"Retrieved metrics for model {model_name}")
            return metrics

    @handle_exception
    def verify_project_exists(self, project_name: str) -> bool:
        """
        Verify that a project exists in Neo4j database.
        
        Args:
            project_name: Name of the project to verify
            
        Returns:
            bool: True if exactly one project exists, False otherwise
        """
        verify_project_query = """
        MATCH (p:Project {name: $project_name})
        RETURN count(p) as project_count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(verify_project_query, project_name=project_name)
            project_count = result.single()['project_count']
            
            if project_count == 0:
                logger.error(f"Project '{project_name}' not found in Neo4j database")
                return False
            elif project_count > 1:
                logger.warning(f"Multiple projects found with name '{project_name}'")
            
            logger.info(f"Successfully verified project: {project_name}")
            return True

    @handle_exception
    def check_schema(self, project_name: str) -> Dict[str, Any]:
        """
        Check Neo4j schema and relationship types based on project hierarchy
        
        Args:
            project_name: Name of the project to check
            
        Returns:
            Dict containing schema information from Neo4j
        """
        logger.debug("Initializing Neo4j schema check")
        try:
            # Count query with proper collection and relationship counting
            count_query = """
            MATCH (p:Project {name: $project_name})
            WITH p
            MATCH (p)-[:CONTAINS*1..]->(r1:Requirement)
            WHERE r1.type = 'SOURCE'
            WITH p, collect(DISTINCT r1) as source_reqs
            MATCH (p)-[:CONTAINS*1..]->(r2:Requirement)
            WHERE r2.type = 'TARGET'
            WITH source_reqs, collect(DISTINCT r2) as target_reqs
            OPTIONAL MATCH (r1)-[rel:SIMILAR_TO]->(r2)
            WHERE r1 IN source_reqs 
            AND r2 IN target_reqs
            AND rel.project = $project_name
            RETURN 
                size(source_reqs) as source_req_count,
                size(target_reqs) as target_req_count,
                count(rel) as similar_count,
                size(source_reqs) * size(target_reqs) as potential_total_pairs
            """
            
            # Schema query with similar pattern
            schema_query = """
            MATCH (p:Project {name: $project_name})
            WITH p
            MATCH (p)-[:CONTAINS*1..]->(r1:Requirement)
            WHERE r1.type = 'SOURCE'
            WITH p, collect(DISTINCT r1) as source_reqs
            MATCH (p)-[:CONTAINS*1..]->(r2:Requirement)
            WHERE r2.type = 'TARGET'
            WITH source_reqs, collect(DISTINCT r2) as target_reqs
            MATCH (r1)-[rel:SIMILAR_TO]->(r2)
            WHERE r1 IN source_reqs 
            AND r2 IN target_reqs
            AND rel.project = $project_name
            WITH DISTINCT labels(r1) as req_labels,
                 collect(DISTINCT type(rel)) as rel_types,
                 collect(DISTINCT keys(rel)) as rel_properties
            RETURN {
                requirement_labels: req_labels,
                relationship_types: rel_types,
                relationship_properties: rel_properties[0]
            } as value
            """
            
            with self.driver.session(database=self.database) as session:
                # Get counts
                count_result = session.run(count_query, project_name=project_name)
                counts = count_result.single()
                
                if counts:
                    logger.info("Database Schema Statistics:")
                    logger.info(f"Project: {project_name}")
                    logger.info(f"Number of source requirements: {counts['source_req_count']}")
                    logger.info(f"Number of target requirements: {counts['target_req_count']}")
                    logger.info(f"Number of SIMILAR_TO relationships: {counts['similar_count']}")
                    logger.info(f"Potential total pairs: {counts['potential_total_pairs']}")
                else:
                    logger.warning("No relationships found in the database")
                
                # Get schema
                schema_result = session.run(schema_query, project_name=project_name)
                schema_record = schema_result.single()
                
                if not schema_record:
                    logger.warning(f"No schema information found for project: {project_name}")
                    schema = {}
                else:
                    schema = schema_record['value']
                    logger.debug("Schema details:")
                    logger.debug(f"Requirement labels: {schema.get('requirement_labels', [])}")
                    logger.debug(f"Relationship types: {schema.get('relationship_types', [])}")
                    logger.debug(f"Relationship properties: {schema.get('relationship_properties', [])}")
                
                return {
                    'counts': dict(counts) if counts else {},
                    'schema': schema
                }
                
        except Exception as e:
            logger.error("Error checking Neo4j schema", exc_info=True)
            handle_exception(e)
            raise

    @handle_exception
    def get_similarity_data(self, project_name: str) -> pd.DataFrame:
        """
        Retrieve similarity data for the current project including model scores
        
        Args:
            project_name: Name of the project to analyze
            
        Returns:
            pandas.DataFrame: DataFrame containing similarity scores and ground truth
        """
        logger.debug("Initializing similarity data retrieval")
        try:
            # Query to get all valid requirement pairs and their scores
            query = """
            MATCH (p:Project {name: $project_name})
            WITH p
            MATCH (p)-[:CONTAINS*1..]->(r1:Requirement)
            WHERE r1.type = 'SOURCE'
            WITH p, collect(DISTINCT r1) as source_reqs
            MATCH (p)-[:CONTAINS*1..]->(r2:Requirement)
            WHERE r2.type = 'TARGET'
            WITH source_reqs, collect(DISTINCT r2) as target_reqs
            UNWIND source_reqs as r1
            UNWIND target_reqs as r2
            OPTIONAL MATCH (r1)-[s:SIMILAR_TO]->(r2)
            WHERE s.project = $project_name
            OPTIONAL MATCH (r1)-[g:GROUND_TRUTH]->(r2)
            WHERE g.project = $project_name
            WITH 
                r1.id as source_id,
                r2.id as target_id,
                s.similarity as similarity_score,
                s.model as model_name,
                CASE WHEN g IS NOT NULL THEN 1 ELSE 0 END as is_related
            WHERE similarity_score IS NOT NULL OR is_related = 1
            RETURN *
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, project_name=project_name)
                records = [dict(record) for record in result]
                
                if not records:
                    logger.warning(f"No data found for project: {project_name}")
                    return pd.DataFrame()
                
                # Create initial DataFrame
                data = pd.DataFrame(records)
                
                # Pivot the data to create separate columns for each model
                model_scores = data.pivot(
                    index=['source_id', 'target_id', 'is_related'],
                    columns='model_name',
                    values='similarity_score'
                ).reset_index()
                
                # Log dataset statistics
                logger.info("\nDataset Statistics:")
                logger.info(f"Total pairs: {len(model_scores)}")
                logger.info(f"Related pairs: {model_scores['is_related'].sum()}")
                logger.info(f"Unrelated pairs: {len(model_scores) - model_scores['is_related'].sum()}")
                
                # Check for missing values
                if model_scores.isnull().values.any():
                    logger.warning("Missing values found in the dataset")
                    logger.debug(f"Missing value counts:\n{model_scores.isnull().sum()}")
                else:
                    logger.debug("No missing values found in the dataset")
                
                # Log sample data
                logger.info("\nFirst 5 rows of dataset:")
                logger.info(f"\n{model_scores.head()}")
                logger.info("\nLast 5 rows of dataset:")
                logger.info(f"\n{model_scores.tail()}")
                logger.info("\nDataset Info:")
                logger.info(f"\n{model_scores.info()}")
                
                return model_scores
                
        except Exception as e:
            logger.error("Error retrieving similarity data", exc_info=True)
            handle_exception(e)
            raise

    @handle_exception
    def get_requirement_texts(self, source_ids: List[str], target_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get requirement texts from Neo4j for given IDs
        
        Args:
            source_ids: List of source requirement IDs
            target_ids: List of target requirement IDs
            
        Returns:
            tuple: (source_texts dict, target_texts dict) mapping IDs to content
            
        Raises:
            Exception: If database query fails
        """
        logger.debug("Fetching requirement texts from Neo4j")
        try:
            # Query to get source requirement texts
            query = """
            MATCH (s:Requirement {type: 'SOURCE'})
            WHERE s.id IN $source_ids
            RETURN s.id as source_id, s.content as source_text
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(query, source_ids=source_ids)
                source_texts = {record['source_id']: record['source_text'] for record in result}
            
            # Query to get target requirement texts
            query = """
            MATCH (t:Requirement {type: 'TARGET'})
            WHERE t.id IN $target_ids
            RETURN t.id as target_id, t.content as target_text
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(query, target_ids=target_ids)
                target_texts = {record['target_id']: record['target_text'] for record in result}
            
            logger.debug(f"Successfully retrieved texts for {len(source_texts)} source and {len(target_texts)} target requirements")
            return source_texts, target_texts
            
        except Exception as e:
            logger.error("Failed to retrieve requirement texts", exc_info=True)
            raise

    @handle_exception
    def select_project(self, project_name: str) -> None:
        """Set the project name for filtering queries."""
        self.project_name = project_name
        logger.debug(f"Project name set to: {self.project_name}")

    @handle_exception
    def get_all_requirement_mappings(self, project_name: str) -> Dict[str, Any]:
        """
        Get all requirement mappings for a specific project, including source requirements
        and their similar target requirements.
        
        Args:
            project_name: Name of the project to get mappings for
            
        Returns:
            Dict containing:
                - metadata: Dict with total relationships and other stats
                - mappings: List of source requirements with their target matches
        """
        query = """
        MATCH (p:Project {name: $project_name})-[:CONTAINS*]->(s:Requirement)
        WHERE s.type = 'SOURCE'
        WITH s
        OPTIONAL MATCH (s)-[r:SIMILAR_TO]->(t:Requirement)
        WHERE t.type = 'TARGET'
        WITH s, 
             collect({
                id: t.id,
                content: t.content,
                similarity: r.similarity
             }) as targets
        RETURN {
            source: {
                id: s.id,
                content: s.content
            },
            targets: targets
        } as mapping
        """
        
        # Query to get total relationship count
        count_query = """
        MATCH (p:Project {name: $project_name})-[:CONTAINS*]->(s:Requirement)
        WHERE s.type = 'SOURCE'
        WITH s
        MATCH (s)-[r:SIMILAR_TO]->(:Requirement {type: 'TARGET'})
        RETURN count(r) as total
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                # Get total relationships
                count_result = session.run(count_query, project_name=project_name)
                total_relationships = count_result.single()["total"]
                logger.info(f"Found {total_relationships} total relationships of type SIMILAR_TO")
                
                # Get mappings
                result = session.run(query, project_name=project_name)
                mappings = [record["mapping"] for record in result]
                logger.info(f"Processed {len(mappings)} source requirements with their targets")
                
                return {
                    "metadata": {
                        "project_name": project_name,
                        "total_relationships": total_relationships,
                        "source_requirements": len(mappings)
                    },
                    "mappings": mappings
                }
                
        except Exception as e:
            logger.error(f"Error getting requirement mappings: {str(e)}", exc_info=True)
            raise

    @handle_exception
    def get_requirement_similarity_data(self, project_name: str) -> Dict[str, Any]:
        """
        Retrieve all requirement similarity data including model scores and ground truth
        for a specific project.
        
        Args:
            project_name: Name of the project to analyze
            
        Returns:
            Dict containing:
                - metadata: Dict with statistics about relationships and requirements
                - data: pandas DataFrame with columns:
                    - source_id: ID of source requirement
                    - target_id: ID of target requirement 
                    - source_content: Content of source requirement
                    - target_content: Content of target requirement
                    - is_ground_truth: Boolean indicating if pair has ground truth relationship
                    - model specific columns: One column per model with similarity scores
        """
        logger.debug(f"Retrieving similarity data for project: {project_name}")
        try:
            query = """
            MATCH (p:Project {name: $project_name})
            WITH p
            
            // Get all source requirements
            MATCH (p)-[:CONTAINS*1..]->(r1:Requirement)
            WHERE r1.type = 'SOURCE'
            WITH p, collect(DISTINCT r1) as source_reqs
            
            // Get all target requirements
            MATCH (p)-[:CONTAINS*1..]->(r2:Requirement)
            WHERE r2.type = 'TARGET'
            WITH source_reqs, collect(DISTINCT r2) as target_reqs
            
            // Create all possible source-target pairs
            UNWIND source_reqs as r1
            UNWIND target_reqs as r2
            
            // Get similarity relationships and ground truth
            OPTIONAL MATCH (r1)-[s:SIMILAR_TO]->(r2)
            WHERE s.project = $project_name
            OPTIONAL MATCH (r1)-[g:GROUND_TRUTH]->(r2)
            WHERE g.project = $project_name
            
            WITH 
                r1.id as source_id,
                r1.content as source_content,
                r2.id as target_id,
                r2.content as target_content,
                s.similarity as similarity_score,
                s.model as similar_to_id,
                CASE WHEN g IS NOT NULL THEN true ELSE false END as is_ground_truth
            WHERE similarity_score IS NOT NULL OR is_ground_truth = true
            
            RETURN 
                source_id,
                source_content,
                target_id, 
                target_content,
                similarity_score,
                similar_to_id,
                is_ground_truth
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, project_name=project_name)
                records = [dict(record) for record in result]
                
                if not records:
                    logger.warning(f"No similarity data found for project: {project_name}")
                    return {
                        "metadata": {
                            "project_name": project_name,
                            "total_pairs": 0,
                            "ground_truth_pairs": 0,
                            "models": []
                        },
                        "data": pd.DataFrame()
                    }
                
                # Create DataFrame and pivot model scores into columns
                df = pd.DataFrame(records)
                model_scores = df.pivot(
                    index=['source_id', 'target_id', 'source_content', 
                           'target_content', 'is_ground_truth'],
                    columns='similar_to_id',
                    values='similarity_score'
                ).reset_index()
                
                # Calculate metadata
                metadata = {
                    "project_name": project_name,
                    "total_pairs": len(model_scores),
                    "ground_truth_pairs": int(model_scores['is_ground_truth'].sum()),
                    "models": [col for col in model_scores.columns 
                              if col not in ['source_id', 'target_id', 'source_content', 
                                           'target_content', 'is_ground_truth']]
                }
                
                # Log statistics
                logger.info("\nDataset Statistics:")
                logger.info(f"Total requirement pairs: {metadata['total_pairs']}")
                logger.info(f"Ground truth pairs: {metadata['ground_truth_pairs']}")
                logger.info(f"Models found: {metadata['models']}")
                
                # Check for missing values
                if model_scores.isnull().values.any():
                    logger.warning("Missing similarity scores found in the dataset")
                    logger.debug(f"Missing value counts:\n{model_scores.isnull().sum()}")
                
                # Log sample data
                logger.debug("\nFirst few rows of dataset:")
                logger.debug(f"\n{model_scores.head()}")
                
                return {
                    "metadata": metadata,
                    "data": model_scores
                }
                
        except Exception as e:
            logger.error("Error retrieving similarity data", exc_info=True)
            handle_exception(e)
            raise

    @handle_exception
    def create_llm_traces_from_results(self, results_set, llm_model_name: str, remove_existing: bool = False, transformers_utilized: List[str] = None) -> Tuple[int, int]:
        """
        Create or update LLM analysis traces between requirements in Neo4j.
        
        Args:
            results_set: Container with processed matches containing RequirementMatch objects
            llm_model_name (str): Name of the LLM model used for analysis
            remove_existing (bool): If True, removes existing relationships before creating new ones
            transformers_utilized (List[str], optional): List of transformer names used in the analysis
            
        Returns:
            Tuple[int, int]: (success_count, failure_count) number of successful and failed relationship creations
        """
        logger.info(f"Creating LLM traces for model: {llm_model_name}")
        
        success_count = 0
        failure_count = 0
        
        # Determine relationship type based on whether transformers are used
        relationship_type = "LLM_RESULT_WITH_TRANSFORMERS" if transformers_utilized else "LLM_RESULT"
        
        # If specified, remove existing relationships for this model and type
        if remove_existing:
            cleanup_query = """
            MATCH ()-[r:%s {model: $model_name}]->()
            DELETE r
            """ % relationship_type
            
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cleanup_query, model_name=llm_model_name)
                    logger.info(f"Removed existing {relationship_type} relationships for model {llm_model_name}")
            except Exception as e:
                logger.error(f"Error removing existing results: {str(e)}")
                raise

        # Create or update relationships
        create_query = f"""
        MATCH (source:Requirement {{id: $source_id, type: 'SOURCE'}})
        MATCH (target:Requirement {{id: $target_id, type: 'TARGET'}})
        MERGE (source)-[r:{relationship_type} {{model: $model_name}}]->(target)
        SET r.association_probability = $probability,
            r.is_associated = $is_associated,
            r.explanation = $explanation,
            r.timestamp = datetime()
        """
        
        # Add transformers to query if provided
        if transformers_utilized:
            create_query += ",\n        r.transformers_utilized = $transformers"
        
        create_query += "\nRETURN r"
        
        try:
            with self.driver.session(database=self.database) as session:
                # Process each match in the results
                for match in results_set.processed_matches:
                    try:
                        # Prepare query parameters
                        params = {
                            "source_id": match.source_id,
                            "target_id": match.target_id,
                            "model_name": llm_model_name,
                            "probability": float(match.association_probability),
                            "is_associated": bool(match.is_associated),
                            "explanation": str(match.explanation)
                        }
                        
                        # Add transformers if provided
                        if transformers_utilized:
                            params["transformers"] = transformers_utilized
                        
                        result = session.run(create_query, **params)
                        
                        if result.single():
                            success_count += 1
                            logger.debug(f"Created/Updated {relationship_type}: {match.source_id} -> {match.target_id}")
                        else:
                            failure_count += 1
                            logger.warning(f"Failed to create {relationship_type}: {match.source_id} -> {match.target_id}")
                            
                    except Exception as e:
                        failure_count += 1
                        logger.error(f"Error creating {relationship_type} for {match.source_id} -> {match.target_id}: {str(e)}")
                        continue
                
                logger.info(f"Completed {relationship_type} creation: {success_count} successful, {failure_count} failed")
                return success_count, failure_count
                
        except Exception as e:
            logger.error(f"Error in create_llm_traces_from_results: {str(e)}")
            raise