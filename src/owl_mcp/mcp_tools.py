"""
MCP tools for working with OWL ontologies.

This module provides a Model-Context-Protocol wrapper around
the OWL Server functionality, allowing integration with other MCP systems.
"""

from functools import wraps
import os
from pathlib import Path
import re
from typing import Callable, Generic, Optional, TypeVar, TypedDict

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from owl_mcp.config import OWLMCPConfig, get_config_manager
from owl_mcp.owl_api import SimpleOwlAPI

import logging
from mcp.server.fastmcp.utilities.logging import get_logger
to_client_logger = get_logger(name="fastmcp.server.context.to_client")
to_client_logger.setLevel(level=logging.ERROR)


class PagedResult(TypedDict):
    items: list
    next_cursor: str | None
    page_size: int
    current_page: int
    total_items: int


def paginated(all_items: list, cursor: str = "0", page_size: int = 50) -> PagedResult:
    """
    Decorator to add cursor-based pagination to a FastMCP tool.

    Wrapped tool must be an async generator yielding items in order.

    Args:
        page_size: Number of items per page.

    Usage:
        @paginate(page_size=20)
        async def fn(...):
            yield item1
            yield item2
            ...
    """
    skip = int(cursor)
    items = []
    
    items = all_items[skip:skip + page_size]

    next_cursor = str(skip + len(items)) if len(items) == page_size else None

    return {
        "items": items,
        "next_cursor": next_cursor,
        "page_size": page_size,
        "current_page": skip // page_size,
        "total_items": len(all_items),
    }

    
# Initialize FastMCP server
mcp = FastMCP(
    "owl-server",
    log_level="ERROR",
    host="127.0.0.1",
    port=8080,
    instructions="""
OWL Server provides tools for managing Web Ontology Language (OWL) ontologies.
Use these tools to add, remove, and find axioms in OWL files, and to manage prefix mappings.

The tools operate on OWL files specified either by:
1. Absolute file paths (e.g., /path/to/ontology.owl)
2. Named ontologies from the configuration (e.g., "my-ontology")

## Configuration System

OWL-Server includes a configuration system that stores ontology metadata and settings
in ~/.owl-mcp/config.yaml. This allows you to:

- Define named ontologies with paths and metadata
- Set ontologies as readonly to prevent modifications
- Add default metadata axioms to ontologies
- Specify preferred serialization formats

To work with configured ontologies, use the configuration tools or the "by_name" variants
of the standard tools.

## Resources

The server provides the following resources:
- `resource://config/ontologies` - List of all configured ontologies
- `resource://config/ontology/{name}` - Details about a specific ontology configuration
- `resource://active` - List of all active OWL file paths
""",
)


@mcp.prompt()
def ask_for_axioms_about_prompt(topic: str) -> str:
    """Generates a user message asking for axioms matching a string."""
    return f"What axioms include the string '{topic}'?"


@mcp.prompt()
def add_subclass_of_prompt(child: str, parent: str) -> str:
    """Generates a user message asking to add a subclass of axiom."""
    return (
        f"Add a subClassOf axiom where the subclass is '{child}' and the superclass is '{parent}'"
    )


# Dictionary to cache SimpleOwlAPI instances
_api_instances = dict()


def _get_api_instance(owl_file_path: str, auto_register: bool = False) -> SimpleOwlAPI:
    """
    Get or create a SimpleOwlAPI instance for the given file path.

    Args:
        owl_file_path: Absolute path to the OWL file
        auto_register: If True and the ontology is not in the configuration yet,
                      automatically register it

    Returns:
        SimpleOwlAPI: Instance for the given file path
    """
    owl_file_path = os.path.abspath(owl_file_path)

    if owl_file_path not in _api_instances:
        # Check if we have configuration for this file
        config_manager = get_config_manager()
        ontology_config = config_manager.get_ontology_by_path(owl_file_path)

        if ontology_config:
            # Use configured settings
            _api_instances[owl_file_path] = SimpleOwlAPI(
                owl_file_path,
                serialization=ontology_config.preferred_serialization,
                readonly=ontology_config.readonly,
                annotation_property=ontology_config.annotation_property,
            )
        else:
            # Use default settings
            _api_instances[owl_file_path] = SimpleOwlAPI(owl_file_path)

            # Auto-register if requested
            if auto_register:
                _api_instances[owl_file_path].register_in_config()

    return _api_instances[owl_file_path]



@mcp.tool()
async def add_axiom(owl_file_path: str, axiom_str: str) -> str:
    """
    Add an axiom to the ontology using OWL functional syntax.

    Args:
        owl_file_path: Absolute path to the OWL file
        axiom_str: String representation of the axiom in OWL functional syntax
                 e.g., "SubClassOf(:Dog :Animal)"

    Returns:
        str: Success message or error
    """
    api = _get_api_instance(owl_file_path)
    success = api.add_axiom(axiom_str)

    if success:
        return f"Successfully added axiom: {axiom_str}"
    return f"Failed to add axiom: {axiom_str}"


@mcp.tool()
async def add_axioms(owl_file_path: str, axiom_strs: list[str]) -> str:
    """
    Adds a list of axioms to the ontology, using OWL functional syntax.

    Args:
        owl_file_path: Absolute path to the OWL file
        axiom_strs: List of string representation of the axiom in OWL functional syntax
                 e.g., ["SubClassOf(:Dog :Animal)", ...]

    Returns:
        str: Success message or error
    """
    api = _get_api_instance(owl_file_path)
    for axiom_str in axiom_strs:
        success = api.add_axiom(axiom_str)
        if not success:
            return f"Failed to add axiom: {axiom_str}"

    return f"Successfully added axioms: {axiom_strs}"


@mcp.tool()
async def remove_axiom(owl_file_path: str, axiom_str: str) -> str:
    """
    Remove an axiom from the ontology using OWL functional syntax.

    Args:
        owl_file_path: Absolute path to the OWL file
        axiom_str: String representation of the axiom in OWL functional syntax

    Returns:
        str: Success message or error
    """
    api = _get_api_instance(owl_file_path)
    success = api.remove_axiom(axiom_str)

    if success:
        return f"Successfully removed axiom: {axiom_str}"
    return f"Failed to remove axiom: {axiom_str}"


@mcp.tool()
async def get_all_classes(owl_file_path: str, label_pattern: Optional[str]=None, cursor: str = "0") -> PagedResult:
    """
    Get all classes defined in the ontology as their IRI and label.
    
    Args:
        owl_file_path: Absolute path to the OWL file
        label_pattern: Optional regex pattern that class labels must match to be included
        cursor: Cursor for pagination (default: "0")
        
    Returns:
        PagedResult: PagedResult containing list of matching classes. Each class is represented as a dictionary with fields "iri" and "label".
    """
    api = _get_api_instance(owl_file_path)

    # results = [f"{e['label']} [{e['iri']}]" for e in api.get_all_entities("class", True) if (not label_pattern or (e['label'] and re.search(label_pattern, e['label'])))]
    results = [e for e in api.get_all_entities("class", True) if (not label_pattern or (e['label'] and re.search(label_pattern, e['label'])))]
    
    return paginated(results, cursor=cursor, page_size=50)

@mcp.tool()
async def search_class_by_definition(owl_file_path: str, search_pattern: str, cursor: str = "0") -> PagedResult:
    """
    Search for classes by matching their natural language definition to a pattern.

    Args:
        owl_file_path: Absolute path to the OWL file
        search_pattern: A substring or regex pattern to match against definitions
                        (supports full Python regex syntax, e.g., r"vertebrate")
        cursor: Cursor for pagination (default: "0")

    Returns:
        PagedResult: PagedResult containing list of matching classes. The dictionary contains the fields "iri", "label", and "definition"
    """
    api = _get_api_instance(owl_file_path)

    return paginated(api.search_definitions(search_pattern), cursor=cursor, page_size=50)

@mcp.tool()
async def search_class_by_name(owl_file_path: str, search_pattern: str, cursor: str = "0") -> PagedResult:
    """
    Search for classes by matching their name/label to a pattern.

    Args:
        owl_file_path: Absolute path to the OWL file
        search_pattern: A substring or regex pattern to match against class labels
                        (supports full Python regex syntax, e.g., r"^A.*e$")
        cursor: Cursor for pagination (default: "0")
    Returns:
        PagedResult: PagedResult containing list of matching classes. The dictionary contains the fields "iri", "label", and "definition"
    """
    
    api = _get_api_instance(owl_file_path)

    # results = [f"{e['label']} [{e['iri']}]" for e in api.get_all_entities("class", True) if (not label_pattern or (e['label'] and re.search(label_pattern, e['label'])))]
    results = [e for e in api.get_all_entities("class", True) if (not search_pattern or (e['label'] and re.search(search_pattern, e['label'])))]
    
    return paginated(results, cursor=cursor, page_size=50)

@mcp.tool()
async def get_superclasses(owl_file_path: str, iri: str) -> list[str]:
    """
    Get the superclasses for a given IRI in the ontology up to the root ordered by distance (direct superclasses first).

    Args:
        owl_file_path: Absolute path to the OWL file
        iri: The IRI to get the superclasses for (as a string)

    Returns:
        list[str]: List of superclass IRIs
    """
    api = _get_api_instance(owl_file_path)
    assert api.ontology is not None, "No ontology loaded!"
    return sorted(api.ontology.get_superclasses(iri))

@mcp.tool()
async def get_subclasses(owl_file_path: str, iri: str, cursor: str = "0") -> PagedResult:
    """
    Get the direct subclasses for a given IRI in the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file
        iri: The IRI to get the subclasses for (as a string)
        cursor: Cursor for pagination (default: "0")

    Returns:
        PagedResult: PagedResult containing list of direct subclass IRIs
    """
    api = _get_api_instance(owl_file_path)
    assert api.ontology is not None, "No ontology loaded!"
    return paginated(sorted(api.ontology.get_subclasses(iri)), cursor=cursor, page_size=50)
@mcp.tool()
async def get_root_classes(owl_file_path: str) -> list[str]:
    """
    Get the root classes (direct subclasses of owl:Thing) in the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file

    Returns:
        list[str]: List of root class IRIs
    """
    api = _get_api_instance(owl_file_path)
    assert api.ontology is not None, "No ontology loaded!"
    return sorted(api.ontology.get_subclasses("http://www.w3.org/2002/07/owl#Thing"))

@mcp.tool()
async def get_hierarchy(owl_file_path: str, root: Optional[str]=None, depth: int = 3) -> dict:
    """
    Get the class hierarchy for a given IRI in the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file
        root: The IRI to get the hierarchy for (as a string). If None, use owl:Thing (root of every ontology)
        depth: The depth of the hierarchy to retrieve (default: 3)

    Returns:
        dict: A dictionary representation of the class hierarchy
    """
    api = _get_api_instance(owl_file_path)
    assert api.ontology is not None, "No ontology loaded!"

    if not root:
        root = "http://www.w3.org/2002/07/owl#Thing"
        
    root_node = dict()
    hierarchy = {root: root_node}

    queue = [(root, root_node, 1)]
    while queue:
        current_root, current_node, current_depth = queue.pop(0)
        
        if current_depth > depth:
            continue
        
        children = api.ontology.get_subclasses(current_root)
        for child in children:
            current_node[child] = dict()
            queue.append((child, current_node[child], current_depth + 1))

    return hierarchy


@mcp.tool()
async def get_definition(owl_file_path: str, iri: str, annotation_property: Optional[str] = None) -> Optional[str]:
    """
    Get the definition for a given IRI in the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file
        iri: The IRI to get the definition for (as a string)
        annotation_property: Optional annotation property IRI to use for definition
                            (defaults to `IAO:0000115 (definition)` if None)

    Returns:
        Optional[str]: The definition if found, otherwise None
    """
    api = _get_api_instance(owl_file_path)
    return next(iter(api.get_definition_for_iri(iri, annotation_property)), None)

# @mcp.tool()
# async def find_axioms(
#     owl_file_path: str,
#     pattern: str,
#     limit=100,
#     include_labels: bool = False,
#     annotation_property: Optional[str] = None,
# ) -> list[str]:
#     """
#     Find axioms matching a pattern in the ontology.

#     Args:
#         owl_file_path: Absolute path to the OWL file
#         pattern: A substring or regex pattern to match against axiom strings
#                  (supports full Python regex syntax, e.g., r"SubClassOf.*:Animal")
#         limit: (int) Maximum number of axioms to return (default: 100)
#         include_labels: If True, include human-readable labels after ## in the output
#         annotation_property: Optional annotation property IRI to use for labels
#                             (defaults to rdfs:label)

#     Returns:
#         list[str]: List of matching axiom strings
#     """
#     api = _get_api_instance(owl_file_path)
#     if isinstance(limit, str):
#         # dumb AI may keep trying this with strings
#         limit = int(limit) if limit else 100
#     return api.find_axioms(
#         pattern, include_labels=include_labels, annotation_property=annotation_property
#     )[0:limit]


@mcp.tool()
async def get_all_axioms(
    owl_file_path: str,
    limit: int =100,
    include_labels: bool = False,
    annotation_property: Optional[str] = None,
) -> list[str]:
    """
    Get all axioms in the ontology as strings.

    Args:
        owl_file_path: Absolute path to the OWL file
        limit: Maximum number of axioms to return (default: 100)
        include_labels: If True, include human-readable labels after ## in the output
        annotation_property: Optional annotation property IRI to use for labels
                            (defaults to rdfs:label)

    Returns:
        list[str]: List of all axiom strings
    """
    try:
        limit = int(limit)
    except:
        limit = 100
    
    api = _get_api_instance(owl_file_path)
    return api.get_all_axiom_strings(
        include_labels=include_labels, annotation_property=annotation_property
    )[0:limit]


@mcp.tool()
async def add_prefix(owl_file_path: str, prefix: str, uri: str) -> str:
    """
    Add a prefix mapping to the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file
        prefix: The prefix string (e.g., "ex")
        uri: The URI the prefix maps to (e.g., "http://example.org/")

    Note that usually an ontology will contain standard prefixes for rdf, rdfs, owl, xsd

    Returns:
        str: Success message
    """
    api = _get_api_instance(owl_file_path)
    success = api.add_prefix(prefix, uri)

    if success:
        return f"Successfully added prefix mapping: {prefix} -> {uri}"
    return f"Failed to add prefix mapping: {prefix} -> {uri}"


@mcp.tool()
async def ontology_metadata(owl_file_path: str) -> list[str]:
    """
    Get metadata about the ontology.

    Args:
        owl_file_path: Absolute path to the OWL file

    Returns:
        list[str]: List of metadata items
    """
    api = _get_api_instance(owl_file_path)
    return api.ontology_annotations()


# Configuration resources


@mcp.resource("resource://config/ontologies")
async def get_config_ontologies() -> OWLMCPConfig:
    """
    Resource that provides a list of all configured ontologies.

    Returns:
        List[OntologyConfigInfo]: List of configured ontologies with their details
    """
    config_manager = get_config_manager()
    return config_manager.config


@mcp.resource("resource://active")
async def list_active_owl_files() -> list[str]:
    """
    List all OWL files currently being managed.

    Returns:
        list[str]: List of file paths for active OWL files
    """
    return list(_api_instances.keys())


async def stop_owl_service(owl_file_path: str) -> str:
    owl_file_path = os.path.abspath(owl_file_path)
    api = _api_instances.get(owl_file_path)
    if api:
        api.stop()
        del _api_instances[owl_file_path]
        return f"Successfully stopped OWL service for {owl_file_path}"
    return "No OWL service running for this file"


# Configuration tools


class OntologyConfigInfo(BaseModel):
    """Simplified model for OntologyConfig that can be passed through MCP."""

    name: str
    path: str
    metadata_axioms: list[str]
    readonly: bool
    description: Optional[str] = None
    preferred_serialization: Optional[str] = None
    annotation_property: Optional[str] = None


@mcp.tool()
async def list_configured_ontologies() -> list[OntologyConfigInfo]:
    """
    List all ontologies defined in the configuration.

    Returns:
        List[OntologyConfigInfo]: List of configured ontologies
    """
    config_manager = get_config_manager()
    ontologies = config_manager.list_ontologies()

    result = []
    for name, config in ontologies.items():
        result.append(
            OntologyConfigInfo(
                name=name,
                path=config.path,
                metadata_axioms=config.metadata_axioms,
                readonly=config.readonly,
                description=config.description,
                preferred_serialization=config.preferred_serialization,
                annotation_property=config.annotation_property,
            )
        )

    return result


@mcp.tool()
async def configure_ontology(
    name: str,
    path: str,
    metadata_axioms: Optional[list[str]] = None,
    readonly: bool = False,
    description: Optional[str] = None,
    preferred_serialization: Optional[str] = None,
    annotation_property: Optional[str] = None,
) -> str:
    """
    Add or update an ontology in the configuration.

    Args:
        name: A unique name for the ontology
        path: Absolute path to the ontology file
        metadata_axioms: List of metadata axioms as strings
        readonly: Whether the ontology is read-only (default: False)
        description: Optional description
        preferred_serialization: Optional preferred serialization format
        annotation_property: Optional annotation property IRI for labels (default: rdfs:label)

    Returns:
        str: Success or error message
    """
    config_manager = get_config_manager()

    # Check if the ontology is already loaded
    path = os.path.abspath(path)
    if path in _api_instances:
        # Reload with new settings
        api = _api_instances[path]
        api.stop()
        del _api_instances[path]

    # Add to configuration
    config_manager.add_ontology(
        name=name,
        path=path,
        metadata_axioms=metadata_axioms or [],
        readonly=readonly,
        description=description,
        preferred_serialization=preferred_serialization,
        annotation_property=annotation_property,
    )

    # Try to verify the file exists
    if not os.path.exists(path):
        return f"Configured ontology '{name}' at {path}, but file does not exist yet."

    return f"Successfully configured ontology '{name}' at {path}"


@mcp.tool()
async def remove_ontology_config(name: str) -> str:
    """
    Remove an ontology from the configuration.

    Args:
        name: Name of the ontology to remove

    Returns:
        str: Success or error message
    """
    config_manager = get_config_manager()
    ontology_config = config_manager.get_ontology(name)

    if not ontology_config:
        return f"No ontology with name '{name}' found in configuration."

    # Check if the ontology is currently loaded
    path = os.path.abspath(ontology_config.path)
    if path in _api_instances:
        # Stop the service
        await stop_owl_service(path)

    # Remove from configuration
    config_manager.remove_ontology(name)
    return f"Successfully removed ontology '{name}' from configuration."


@mcp.tool()
async def get_ontology_config(name: str) -> Optional[OntologyConfigInfo]:
    """
    Get configuration for a specific ontology.

    Args:
        name: Name of the ontology

    Returns:
        Optional[OntologyConfigInfo]: The ontology configuration or None if not found
    """
    config_manager = get_config_manager()
    config = config_manager.get_ontology(name)

    if not config:
        return None

    return OntologyConfigInfo(
        name=name,
        path=config.path,
        metadata_axioms=config.metadata_axioms,
        readonly=config.readonly,
        description=config.description,
        preferred_serialization=config.preferred_serialization,
        annotation_property=config.annotation_property,
    )


@mcp.tool()
async def register_ontology_in_config(
    owl_file_path: str,
    name: Optional[str] = None,
    readonly: Optional[bool] = None,
    description: Optional[str] = None,
    preferred_serialization: Optional[str] = None,
    annotation_property: Optional[str] = None,
) -> str:
    """
    Register an existing ontology in the configuration system.

    This allows you to save preferences and metadata for frequently used ontologies,
    making them accessible by name in future sessions.

    Args:
        owl_file_path: Absolute path to the ontology file
        name: Optional custom name for the ontology (defaults to filename without extension)
        readonly: Whether the ontology should be read-only (defaults to current setting if loaded)
        description: Optional description for the ontology
        preferred_serialization: Optional preferred serialization format
        annotation_property: Optional annotation property IRI for labels (defaults to current setting if loaded)

    Returns:
        str: Name of the registered ontology
    """
    # Get or create the API instance
    api = _get_api_instance(owl_file_path)

    # Register in configuration
    registered_name = api.register_in_config(
        name=name,
        readonly=readonly,
        description=description,
        preferred_serialization=preferred_serialization,
        annotation_property=annotation_property,
    )

    return f"Successfully registered ontology '{registered_name}' in configuration"


@mcp.tool()
async def load_and_register_ontology(
    owl_file_path: str,
    name: Optional[str] = None,
    readonly: bool = False,
    create_if_not_exists: bool = True,
    description: Optional[str] = None,
    preferred_serialization: Optional[str] = None,
    metadata_axioms: Optional[list[str]] = None,
    annotation_property: Optional[str] = None,
) -> str:
    """
    Load an ontology and register it in the configuration system in one step.

    Args:
        owl_file_path: Absolute path to the ontology file
        name: Optional name for the ontology (defaults to filename stem)
        readonly: Whether the ontology should be read-only (default: False)
        create_if_not_exists: If True, create the file if it doesn't exist (default: True)
        description: Optional description of the ontology
        preferred_serialization: Optional preferred serialization format
        metadata_axioms: Optional list of metadata axioms to add to the ontology
        annotation_property: Optional annotation property IRI for labels (default: rdfs:label)

    Returns:
        str: Success message
    """
    # Convert to absolute path
    owl_file_path = os.path.abspath(owl_file_path)

    # Check if the ontology is already registered
    get_config_manager()

    # If name is not provided, derive it from the filename
    if name is None:
        name = Path(owl_file_path).stem

    # Check if we need to create the file
    file_exists = os.path.exists(owl_file_path)
    if not file_exists and not create_if_not_exists:
        return f"File does not exist: {owl_file_path}"

    # Get or create an API instance (don't auto-register yet)
    api = _get_api_instance(owl_file_path)

    # Add metadata axioms if provided
    if metadata_axioms:
        for axiom in metadata_axioms:
            api.add_axiom(axiom, bypass_readonly=True)

    # Register in configuration
    api.register_in_config(
        name=name,
        readonly=readonly,
        description=description,
        preferred_serialization=preferred_serialization,
        annotation_property=annotation_property,
    )

    # Determine if the file was created or opened
    action = "Created and registered" if not file_exists else "Loaded and registered"

    return f"{action} ontology '{name}' at {owl_file_path}"


# Tools to work with configured ontologies by name


def _get_ontology_path_by_name(name: str) -> Optional[str]:
    """
    Helper function to get the path for a configured ontology by name.

    Args:
        name: Name of the ontology in the configuration

    Returns:
        Optional[str]: Path to the ontology file or None if not found
    """
    config_manager = get_config_manager()
    config = config_manager.get_ontology(name)

    if not config:
        return None

    return config.path


@mcp.tool()
async def add_axiom_by_name(ontology_name: str, axiom_str: str) -> str:
    """
    Add an axiom to a configured ontology using its name.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        axiom_str: String representation of the axiom in OWL functional syntax

    Returns:
        str: Success message or error
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return f"No ontology with name '{ontology_name}' found in configuration."

    return await add_axiom(owl_file_path, axiom_str)


@mcp.tool()
async def remove_axiom_by_name(ontology_name: str, axiom_str: str) -> str:
    """
    Remove an axiom from a configured ontology using its name.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        axiom_str: String representation of the axiom in OWL functional syntax

    Returns:
        str: Success message or error
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return f"No ontology with name '{ontology_name}' found in configuration."

    return await remove_axiom(owl_file_path, axiom_str)


@mcp.tool()
async def find_axioms_by_name(
    ontology_name: str,
    pattern: str,
    limit=100,
    include_labels: bool = False,
    annotation_property: Optional[str] = None,
) -> list[str]:
    """
    Find axioms matching a pattern in a configured ontology using its name.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        pattern: A string pattern to match against axiom strings
        limit: Maximum number of axioms to return (default: 100)
        include_labels: If True, include human-readable labels after ## in the output
        annotation_property: Optional annotation property IRI to use for labels
                            (defaults to rdfs:label)

    Returns:
        list[str]: List of matching axiom strings or empty list if ontology not found
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return []

    return await find_axioms(
        owl_file_path,
        pattern,
        limit,
        include_labels=include_labels,
        annotation_property=annotation_property,
    )


@mcp.tool()
async def add_prefix_by_name(ontology_name: str, prefix: str, uri: str) -> str:
    """
    Add a prefix mapping to a configured ontology using its name.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        prefix: The prefix string (e.g., "ex:")
        uri: The URI the prefix maps to (e.g., "http://example.org/")

    Returns:
        str: Success message or error
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return f"No ontology with name '{ontology_name}' found in configuration."

    return await add_prefix(owl_file_path, prefix, uri)

@mcp.tool()
async def get_iri_for_label(ontology_name: str, label: str) -> Optional[str]:
    """
    Get the IRI for a given label in a configured ontology.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        label: The label to find the IRI for

    Returns:
        Optional[str]: The IRI if found, otherwise None
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return None

    api = _get_api_instance(owl_file_path)
    return api.get_iri_for_label(ontology_name, label)

@mcp.tool()
async def get_labels_for_iri(
    owl_file_path: str, iri: str, annotation_property: Optional[str] = None
) -> list[str]:
    """
    Get all labels for a given IRI.

    Args:
        owl_file_path: Absolute path to the OWL file
        iri: The IRI to get labels for (as a string)
        annotation_property: Optional annotation property IRI to use for labels
                            (defaults to rdfs:label if None)

    Returns:
        List[str]: List of label strings
    """
    api = _get_api_instance(owl_file_path)
    return api.get_labels_for_iri(iri, annotation_property)


@mcp.tool()
async def get_labels_for_iri_by_name(
    ontology_name: str, iri: str, annotation_property: Optional[str] = None
) -> list[str]:
    """
    Get all labels for a given IRI in a configured ontology.

    Args:
        ontology_name: Name of the ontology as defined in configuration
        iri: The IRI to get labels for (as a string)
        annotation_property: Optional annotation property IRI to use for labels
                            (defaults to rdfs:label if None)

    Returns:
        List[str]: List of label strings or empty list if ontology not found
    """
    owl_file_path = _get_ontology_path_by_name(ontology_name)
    if not owl_file_path:
        return []

    return await get_labels_for_iri(owl_file_path, iri, annotation_property)

def main():
    """
    Run the MCP server.
    """
    mcp.run(transport="stdio", )


if __name__ == "__main__":
    # Initialize and run the server
    main()
