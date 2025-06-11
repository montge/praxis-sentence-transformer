"""
Data models for Neo4j operations
"""

from typing import List, NamedTuple
from dataclasses import dataclass, field

class RequirementNode(NamedTuple):
    """Structure to hold parsed requirement ID components"""
    prefix: str  # SRS or DPUSDS
    section: int  # Always 5 in this case
    subsections: List[int]  # The remaining numbers
    full_id: str
    level: int 

@dataclass
class Requirement:
    """Requirement node"""
    id: str
    content: str
    type: str  # SOURCE or TARGET
    level: int = 1

@dataclass
class Project:
    """Project containing documents"""
    name: str
    description: str = ""
    documents: List['Document'] = field(default_factory=list)

@dataclass
class Document:
    """Represents a document in the requirements hierarchy"""
    
    def __init__(self, id: str, title: str, content: str, type: str, project_name: str):
        self.id = id
        self.title = title
        self.content = content
        self.type = type
        self.project_name = project_name
        self.requirements = []  # Add this to store requirements
        
    def get_all_requirements(self) -> List[Requirement]:
        """Get all requirements associated with this document"""
        return self.requirements
        
    def add_requirement(self, requirement: Requirement):
        """Add a requirement to this document"""
        self.requirements.append(requirement)

@dataclass
class Section:
    """Section within a document"""
    id: str
    number: str
    level: int
    content: str = ""
    subsections: List['Section'] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)