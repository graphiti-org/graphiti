# API Reference

## Overview

Graphiti is a Python library for building and querying temporal knowledge graphs. It provides a high-level API for extracting entities and relationships from unstructured data (text, JSON), storing them in a graph database, and performing intelligent searches across the knowledge graph.

Key features:
- Automatic entity and relationship extraction from text and JSON
- Temporal knowledge tracking with validity dates
- Hybrid search combining semantic similarity and BM25
- Support for multiple graph databases (Neo4j, FalkorDB, Kuzu, Neptune)
- Customizable LLM and embedding providers
- Community detection and hierarchical knowledge organization

## Quick Start

```python
import asyncio
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

async def main():
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    try:
        # Add an episode (document) to the graph
        await graphiti.add_episode(
            name="Meeting Notes",
            episode_body="Alice met Bob at the coffee shop. They discussed the new project.",
            source=EpisodeType.text,
            source_description="meeting transcript",
            reference_time=datetime.now(timezone.utc)
        )

        # Search the knowledge graph
        results = await graphiti.search("Who did Alice meet?")

        for edge in results:
            print(f"Fact: {edge.fact}")
            print(f"Valid from: {edge.valid_at}")
            print("---")
    finally:
        await graphiti.close()

asyncio.run(main())
```

## Core Classes

### Graphiti (Main Entry Point)

**File:** `graphiti_core/graphiti.py` (lines 128-1264)

The main class for interacting with Graphiti. Manages connections to the graph database and provides methods for adding data and searching.

#### Constructor

```python
def __init__(
    self,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    llm_client: LLMClient | None = None,
    embedder: EmbedderClient | None = None,
    cross_encoder: CrossEncoderClient | None = None,
    store_raw_episode_content: bool = True,
    graph_driver: GraphDriver | None = None,
    max_coroutines: int | None = None,
    tracer: Tracer | None = None,
    trace_span_prefix: str = 'graphiti',
)
```

**Parameters:**

- `uri` (str | None): The URI of the graph database (e.g., "bolt://localhost:7687" for Neo4j). Required if `graph_driver` is None.
- `user` (str | None): Username for database authentication. Required if `graph_driver` is None.
- `password` (str | None): Password for database authentication. Required if `graph_driver` is None.
- `llm_client` (LLMClient | None): Custom LLM client for entity/relationship extraction. Defaults to OpenAIClient.
- `embedder` (EmbedderClient | None): Custom embedder for generating vector embeddings. Defaults to OpenAIEmbedder.
- `cross_encoder` (CrossEncoderClient | None): Custom cross-encoder for reranking search results. Defaults to OpenAIRerankerClient.
- `store_raw_episode_content` (bool): Whether to store the raw content of episodes in the database. Defaults to True.
- `graph_driver` (GraphDriver | None): Custom graph driver instance. If provided, uri/user/password are ignored.
- `max_coroutines` (int | None): Maximum number of concurrent operations. Overrides SEMAPHORE_LIMIT environment variable.
- `tracer` (Tracer | None): OpenTelemetry tracer for distributed tracing. If None, tracing is disabled.
- `trace_span_prefix` (str): Prefix for OpenTelemetry span names. Defaults to 'graphiti'.

**Example:**

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import AnthropicClient
from graphiti_core.embedder import OpenAIEmbedder

# Basic initialization with defaults (uses OpenAI)
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Advanced initialization with custom providers
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=AnthropicClient(),
    embedder=OpenAIEmbedder(),
    max_coroutines=10
)

# Using a custom graph driver
from graphiti_core.driver import FalkorDriver

driver = FalkorDriver(host="localhost", port=6379)
graphiti = Graphiti(graph_driver=driver)
```

#### Methods

##### add_episode()

**File:** `graphiti_core/graphiti.py` (lines 615-825)

Process an episode and update the graph. Extracts entities and relationships from the content and stores them in the knowledge graph.

```python
async def add_episode(
    self,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    source: EpisodeType = EpisodeType.message,
    group_id: str | None = None,
    uuid: str | None = None,
    update_communities: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    previous_episode_uuids: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
) -> AddEpisodeResults
```

**Parameters:**

- `name` (str): Name or title of the episode
- `episode_body` (str): The content to process (text or JSON string)
- `source_description` (str): Description of where this data came from (e.g., "podcast transcript", "meeting notes")
- `reference_time` (datetime): Timestamp when this episode occurred or was created
- `source` (EpisodeType): Type of episode. Options: `EpisodeType.text`, `EpisodeType.json`, `EpisodeType.message`. Defaults to `EpisodeType.message`.
- `group_id` (str | None): Graph partition identifier for multi-tenant scenarios. If None, uses default group.
- `uuid` (str | None): Optional UUID for the episode. If None, generates automatically.
- `update_communities` (bool): Whether to update community nodes after adding this episode. Defaults to False.
- `entity_types` (dict[str, type[BaseModel]] | None): Custom entity type definitions using Pydantic models.
- `excluded_entity_types` (list[str] | None): Entity types to exclude from extraction.
- `previous_episode_uuids` (list[str] | None): Specific episodes to use as context. If None, uses most recent episodes.
- `edge_types` (dict[str, type[BaseModel]] | None): Custom edge type definitions using Pydantic models.
- `edge_type_map` (dict[tuple[str, str], list[str]] | None): Mapping of (source_type, target_type) to allowed edge types.

**Returns:**

- `AddEpisodeResults`: Object containing:
  - `episode` (EpisodicNode): The created episode node
  - `episodic_edges` (list[EpisodicEdge]): Edges connecting episode to entities
  - `nodes` (list[EntityNode]): Extracted entity nodes
  - `edges` (list[EntityEdge]): Extracted relationship edges
  - `communities` (list[CommunityNode]): Updated community nodes (if `update_communities=True`)
  - `community_edges` (list[CommunityEdge]): Edges to communities (if `update_communities=True`)

**Example:**

```python
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType

# Add a simple text episode
result = await graphiti.add_episode(
    name="Customer Call",
    episode_body="Sarah called about her order #12345. She wants to change the delivery address.",
    source=EpisodeType.text,
    source_description="customer service call",
    reference_time=datetime.now(timezone.utc)
)

print(f"Extracted {len(result.nodes)} entities")
print(f"Extracted {len(result.edges)} relationships")

# Add a JSON episode
import json
json_data = {
    "customer": "Sarah Johnson",
    "order_id": "12345",
    "issue": "address change",
    "priority": "high"
}

result = await graphiti.add_episode(
    name="Support Ticket",
    episode_body=json.dumps(json_data),
    source=EpisodeType.json,
    source_description="support system",
    reference_time=datetime.now(timezone.utc)
)

# Add episode with custom entity types
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Full name")
    role: str = Field(description="Job role or position")

entity_types = {"Person": Person}

result = await graphiti.add_episode(
    name="Team Meeting",
    episode_body="Alex, the lead engineer, presented the roadmap to Maria, the product manager.",
    source=EpisodeType.text,
    source_description="meeting notes",
    reference_time=datetime.now(timezone.utc),
    entity_types=entity_types
)
```

##### add_episode_bulk()

**File:** `graphiti_core/graphiti.py` (lines 826-1011)

Process multiple episodes in a single batch operation for better performance.

```python
async def add_episode_bulk(
    self,
    bulk_episodes: list[RawEpisode],
    group_id: str | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
) -> AddBulkEpisodeResults
```

**Parameters:**

- `bulk_episodes` (list[RawEpisode]): List of episodes to process
- `group_id` (str | None): Graph partition identifier
- `entity_types` (dict[str, type[BaseModel]] | None): Custom entity type definitions
- `excluded_entity_types` (list[str] | None): Entity types to exclude
- `edge_types` (dict[str, type[BaseModel]] | None): Custom edge type definitions
- `edge_type_map` (dict[tuple[str, str], list[str]] | None): Mapping of entity pairs to edge types

**Returns:**

- `AddBulkEpisodeResults`: Contains lists of episodes, edges, nodes, communities

**Example:**

```python
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Prepare multiple episodes
episodes = [
    RawEpisode(
        name=f"Episode {i}",
        content=f"Content for episode {i}",
        source=EpisodeType.text,
        source_description="bulk import",
        reference_time=datetime.now(timezone.utc)
    )
    for i in range(10)
]

# Process all episodes in one batch
result = await graphiti.add_episode_bulk(bulk_episodes=episodes)

print(f"Processed {len(result.episodes)} episodes")
print(f"Extracted {len(result.nodes)} total entities")
```

##### search()

**File:** `graphiti_core/graphiti.py` (lines 1051-1110)

Perform a hybrid search on the knowledge graph using both semantic similarity and BM25.

```python
async def search(
    self,
    query: str,
    center_node_uuid: str | None = None,
    group_ids: list[str] | None = None,
    num_results: int = 10,
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
) -> list[EntityEdge]
```

**Parameters:**

- `query` (str): Search query string
- `center_node_uuid` (str | None): UUID of a node to use as center for distance-based reranking
- `group_ids` (list[str] | None): Graph partitions to search. If None, searches all partitions.
- `num_results` (int): Maximum number of results to return. Defaults to 10.
- `search_filter` (SearchFilters | None): Filters to apply to search results
- `driver` (GraphDriver | None): Custom driver instance. If None, uses the default driver.

**Returns:**

- `list[EntityEdge]`: List of relationship edges relevant to the query

**Example:**

```python
# Basic search
results = await graphiti.search("What is Alice's role?")

for edge in results:
    print(f"Fact: {edge.fact}")
    print(f"Source: {edge.source_node_uuid}")
    print(f"Target: {edge.target_node_uuid}")
    print(f"Valid: {edge.valid_at} to {edge.invalid_at or 'present'}")
    print("---")

# Search with center node reranking
# This reranks results based on graph distance from a specific node
results = await graphiti.search(
    "Tell me about the project",
    center_node_uuid="some-node-uuid",
    num_results=20
)

# Search with filters
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
from datetime import datetime, timedelta

# Only get facts valid after a certain date
filters = SearchFilters(
    valid_at=[[DateFilter(
        date=datetime.now() - timedelta(days=30),
        comparison_operator=ComparisonOperator.greater_than
    )]]
)

results = await graphiti.search(
    "Recent updates",
    search_filter=filters
)
```

##### search_()

**File:** `graphiti_core/graphiti.py` (lines 1127-1153)

Advanced search method with configurable search strategies and rerankers. Returns nodes, edges, episodes, and communities.

```python
async def search_(
    self,
    query: str,
    config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    group_ids: list[str] | None = None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults
```

**Parameters:**

- `query` (str): Search query
- `config` (SearchConfig): Search configuration defining methods and rerankers. Defaults to `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`.
- `group_ids` (list[str] | None): Graph partitions to search
- `center_node_uuid` (str | None): Center node for distance-based reranking
- `bfs_origin_node_uuids` (list[str] | None): Starting nodes for breadth-first search
- `search_filter` (SearchFilters | None): Additional filters
- `driver` (GraphDriver | None): Custom driver instance

**Returns:**

- `SearchResults`: Object containing:
  - `edges` (list[EntityEdge]): Matching relationship edges
  - `edge_reranker_scores` (list[float]): Relevance scores for edges
  - `nodes` (list[EntityNode]): Matching entity nodes
  - `node_reranker_scores` (list[float]): Relevance scores for nodes
  - `episodes` (list[EpisodicNode]): Matching episodes
  - `episode_reranker_scores` (list[float]): Relevance scores for episodes
  - `communities` (list[CommunityNode]): Matching communities
  - `community_reranker_scores` (list[float]): Relevance scores for communities

**Example:**

```python
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

# Search for edges only using RRF reranking
edge_config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
edge_config.limit = 15

results = await graphiti.search_(
    "Alice's projects",
    config=edge_config
)

print(f"Found {len(results.edges)} edges")
for edge, score in zip(results.edges, results.edge_reranker_scores):
    print(f"Score: {score:.3f} - {edge.fact}")

# Search for nodes using MMR for diversity
node_config = NODE_HYBRID_SEARCH_MMR.model_copy(deep=True)
node_config.limit = 10

results = await graphiti.search_(
    "engineers",
    config=node_config
)

print(f"Found {len(results.nodes)} nodes")
for node, score in zip(results.nodes, results.node_reranker_scores):
    print(f"Score: {score:.3f} - {node.name}: {node.summary}")

# Combined search across all graph layers
results = await graphiti.search_(
    "company reorganization",
    config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

print(f"Edges: {len(results.edges)}")
print(f"Nodes: {len(results.nodes)}")
print(f"Episodes: {len(results.episodes)}")
print(f"Communities: {len(results.communities)}")
```

##### build_communities()

**File:** `graphiti_core/graphiti.py` (lines 1014-1048)

Use community detection algorithms to find clusters of related entities and create community summary nodes.

```python
async def build_communities(
    self,
    group_ids: list[str] | None = None,
    driver: GraphDriver | None = None
) -> tuple[list[CommunityNode], list[CommunityEdge]]
```

**Parameters:**

- `group_ids` (list[str] | None): Only build communities for specific graph partitions
- `driver` (GraphDriver | None): Custom driver instance

**Returns:**

- `tuple[list[CommunityNode], list[CommunityEdge]]`: Created community nodes and their edges to member entities

**Example:**

```python
# Build communities across the entire graph
communities, community_edges = await graphiti.build_communities()

print(f"Created {len(communities)} communities")

for community in communities:
    print(f"Community: {community.name}")
    print(f"Summary: {community.summary}")
    print("---")

# Build communities for specific groups
communities, edges = await graphiti.build_communities(
    group_ids=["project-alpha", "project-beta"]
)
```

##### retrieve_episodes()

**File:** `graphiti_core/graphiti.py` (lines 577-613)

Retrieve the most recent episodic nodes from the graph.

```python
async def retrieve_episodes(
    self,
    reference_time: datetime,
    last_n: int = EPISODE_WINDOW_LEN,
    group_ids: list[str] | None = None,
    source: EpisodeType | None = None,
    driver: GraphDriver | None = None,
) -> list[EpisodicNode]
```

**Parameters:**

- `reference_time` (datetime): Get episodes before this time
- `last_n` (int): Number of episodes to retrieve
- `group_ids` (list[str] | None): Filter by graph partitions
- `source` (EpisodeType | None): Filter by episode type
- `driver` (GraphDriver | None): Custom driver instance

**Returns:**

- `list[EpisodicNode]`: List of episode nodes

**Example:**

```python
from datetime import datetime, timezone

# Get the 10 most recent episodes
episodes = await graphiti.retrieve_episodes(
    reference_time=datetime.now(timezone.utc),
    last_n=10
)

for episode in episodes:
    print(f"Episode: {episode.name}")
    print(f"Time: {episode.valid_at}")
    print(f"Content: {episode.content[:100]}...")
```

##### add_triplet()

**File:** `graphiti_core/graphiti.py` (lines 1169-1233)

Directly add a source node, edge, and target node to the graph without going through episode processing.

```python
async def add_triplet(
    self,
    source_node: EntityNode,
    edge: EntityEdge,
    target_node: EntityNode
) -> AddTripletResults
```

**Parameters:**

- `source_node` (EntityNode): Source entity node
- `edge` (EntityEdge): Relationship edge
- `target_node` (EntityNode): Target entity node

**Returns:**

- `AddTripletResults`: Contains created/updated nodes and edges

**Example:**

```python
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from datetime import datetime, timezone

# Create nodes
source = EntityNode(
    name="Alice",
    group_id="default",
    labels=["Person"],
    summary="Software engineer"
)

target = EntityNode(
    name="Project X",
    group_id="default",
    labels=["Project"],
    summary="New product initiative"
)

# Create edge
edge = EntityEdge(
    source_node_uuid=source.uuid,
    target_node_uuid=target.uuid,
    name="WORKS_ON",
    fact="Alice works on Project X",
    group_id="default",
    created_at=datetime.now(timezone.utc),
    valid_at=datetime.now(timezone.utc)
)

# Add to graph
result = await graphiti.add_triplet(source, edge, target)
```

##### build_indices_and_constraints()

**File:** `graphiti_core/graphiti.py` (lines 321-353)

Build database indices and constraints for optimal query performance.

```python
async def build_indices_and_constraints(
    self,
    delete_existing: bool = False
)
```

**Parameters:**

- `delete_existing` (bool): Whether to delete existing indices first. Defaults to False.

**Example:**

```python
# Build indices (called automatically on initialization)
await graphiti.build_indices_and_constraints()

# Rebuild indices from scratch
await graphiti.build_indices_and_constraints(delete_existing=True)
```

##### close()

**File:** `graphiti_core/graphiti.py` (lines 289-319)

Close the database connection and release resources.

```python
async def close()
```

**Example:**

```python
# Always close when done
try:
    # Use graphiti...
    pass
finally:
    await graphiti.close()

# Or use as async context manager (if supported)
async with graphiti:
    # Use graphiti...
    pass
```

---

## Node Types

### Node (Base Class)

**File:** `graphiti_core/nodes.py` (lines 87-293)

Abstract base class for all node types.

**Attributes:**

- `uuid` (str): Unique identifier
- `name` (str): Display name
- `group_id` (str): Graph partition identifier
- `labels` (list[str]): Category labels
- `created_at` (datetime): Creation timestamp

**Methods:**

- `save(driver)`: Save node to database
- `delete(driver)`: Delete node from database
- `delete_by_group_id(driver, group_id)`: Delete all nodes in a partition
- `delete_by_uuids(driver, uuids)`: Delete multiple nodes by UUID
- `get_by_uuid(driver, uuid)`: Retrieve node by UUID
- `get_by_uuids(driver, uuids)`: Retrieve multiple nodes

### EpisodicNode

**File:** `graphiti_core/nodes.py` (lines 295-433)

Represents a document or episode added to the knowledge graph.

**Attributes:**

- Inherits all Node attributes
- `source` (EpisodeType): Type of episode (text, json, message)
- `source_description` (str): Description of data source
- `content` (str): Raw episode content
- `valid_at` (datetime): When the episode occurred
- `entity_edges` (list[str]): UUIDs of edges extracted from this episode

**Example:**

```python
from graphiti_core.nodes import EpisodicNode, EpisodeType
from datetime import datetime, timezone

episode = EpisodicNode(
    name="Meeting Summary",
    group_id="team-alpha",
    source=EpisodeType.text,
    source_description="Zoom transcript",
    content="Alice and Bob discussed the Q4 roadmap...",
    valid_at=datetime.now(timezone.utc),
    entity_edges=[]
)

await episode.save(driver)

# Retrieve episode
retrieved = await EpisodicNode.get_by_uuid(driver, episode.uuid)
print(retrieved.content)
```

### EntityNode

**File:** `graphiti_core/nodes.py` (lines 435-589)

Represents an entity (person, place, thing, concept) extracted from episodes.

**Attributes:**

- Inherits all Node attributes
- `name_embedding` (list[float] | None): Vector embedding of entity name
- `summary` (str): Summary of entity and its relationships
- `attributes` (dict[str, Any]): Additional custom attributes

**Methods:**

- `generate_name_embedding(embedder)`: Generate vector embedding for the name
- `load_name_embedding(driver)`: Load embedding from database

**Example:**

```python
from graphiti_core.nodes import EntityNode

# Create entity
entity = EntityNode(
    name="Alice Smith",
    group_id="default",
    labels=["Person", "Employee"],
    summary="Senior software engineer specializing in ML",
    attributes={
        "department": "Engineering",
        "hire_date": "2020-01-15",
        "skills": ["Python", "ML", "NLP"]
    }
)

# Generate embedding
await entity.generate_name_embedding(embedder)

# Save to database
await entity.save(driver)

# Search by attributes
nodes = await EntityNode.get_by_group_ids(
    driver,
    group_ids=["default"],
    limit=10
)

for node in nodes:
    print(f"{node.name}: {node.attributes}")
```

### CommunityNode

**File:** `graphiti_core/nodes.py` (lines 591-729)

Represents a cluster of related entities identified by community detection algorithms.

**Attributes:**

- Inherits all Node attributes
- `name_embedding` (list[float] | None): Vector embedding of community name
- `summary` (str): Summary of the community and its members

**Methods:**

- `generate_name_embedding(embedder)`: Generate vector embedding
- `load_name_embedding(driver)`: Load embedding from database

**Example:**

```python
from graphiti_core.nodes import CommunityNode

# Communities are typically created by build_communities()
communities, _ = await graphiti.build_communities()

for community in communities:
    print(f"Community: {community.name}")
    print(f"Summary: {community.summary}")

# Manually retrieve communities
communities = await CommunityNode.get_by_group_ids(
    driver,
    group_ids=["default"]
)
```

---

## Edge Types

### Edge (Base Class)

**File:** `graphiti_core/edges.py` (lines 45-129)

Abstract base class for all edge types.

**Attributes:**

- `uuid` (str): Unique identifier
- `group_id` (str): Graph partition identifier
- `source_node_uuid` (str): UUID of source node
- `target_node_uuid` (str): UUID of target node
- `created_at` (datetime): Creation timestamp

**Methods:**

- `save(driver)`: Save edge to database
- `delete(driver)`: Delete edge from database
- `delete_by_uuids(driver, uuids)`: Delete multiple edges

### EpisodicEdge

**File:** `graphiti_core/edges.py` (lines 131-219)

Connects an episodic node to an entity node, indicating that the entity was mentioned in that episode.

**Relationship Type:** MENTIONS

**Example:**

```python
from graphiti_core.edges import EpisodicEdge
from datetime import datetime, timezone

# Usually created automatically by add_episode
edge = EpisodicEdge(
    source_node_uuid=episode.uuid,  # EpisodicNode UUID
    target_node_uuid=entity.uuid,   # EntityNode UUID
    group_id="default",
    created_at=datetime.now(timezone.utc)
)

await edge.save(driver)

# Retrieve episodic edges
edges = await EpisodicEdge.get_by_group_ids(
    driver,
    group_ids=["default"],
    limit=100
)
```

### EntityEdge

**File:** `graphiti_core/edges.py` (lines 221-478)

Represents a relationship between two entities, containing a fact statement and temporal validity information.

**Relationship Type:** RELATES_TO

**Attributes:**

- Inherits all Edge attributes
- `name` (str): Relationship type name (e.g., "WORKS_FOR", "KNOWS")
- `fact` (str): Natural language statement of the relationship
- `fact_embedding` (list[float] | None): Vector embedding of the fact
- `episodes` (list[str]): Episode UUIDs that mention this relationship
- `expired_at` (datetime | None): When the edge was invalidated
- `valid_at` (datetime | None): When the fact became true
- `invalid_at` (datetime | None): When the fact stopped being true
- `attributes` (dict[str, Any]): Additional custom attributes

**Methods:**

- `generate_embedding(embedder)`: Generate vector embedding for the fact
- `load_fact_embedding(driver)`: Load embedding from database
- `get_between_nodes(driver, source_uuid, target_uuid)`: Get edges between two nodes
- `get_by_node_uuid(driver, node_uuid)`: Get all edges connected to a node

**Example:**

```python
from graphiti_core.edges import EntityEdge
from datetime import datetime, timezone

# Create a relationship
edge = EntityEdge(
    source_node_uuid=alice.uuid,
    target_node_uuid=acme_corp.uuid,
    name="WORKS_FOR",
    fact="Alice works for Acme Corp as a senior engineer",
    group_id="default",
    created_at=datetime.now(timezone.utc),
    valid_at=datetime(2020, 1, 15, tzinfo=timezone.utc),
    episodes=["episode-uuid-1", "episode-uuid-2"],
    attributes={
        "job_title": "Senior Software Engineer",
        "department": "R&D"
    }
)

await edge.generate_embedding(embedder)
await edge.save(driver)

# Find all relationships for an entity
edges = await EntityEdge.get_by_node_uuid(driver, alice.uuid)

for edge in edges:
    print(f"{edge.name}: {edge.fact}")
    print(f"Valid: {edge.valid_at} to {edge.invalid_at or 'present'}")

# Get edges between two specific nodes
edges = await EntityEdge.get_between_nodes(
    driver,
    source_node_uuid=alice.uuid,
    target_node_uuid=bob.uuid
)
```

### CommunityEdge

**File:** `graphiti_core/edges.py` (lines 480-562)

Connects a community node to its member entity nodes.

**Relationship Type:** HAS_MEMBER

**Example:**

```python
from graphiti_core.edges import CommunityEdge

# Usually created automatically by build_communities
edge = CommunityEdge(
    source_node_uuid=community.uuid,
    target_node_uuid=entity.uuid,
    group_id="default",
    created_at=datetime.now(timezone.utc)
)

await edge.save(driver)
```

---

## Database Drivers

### GraphDriver (Base Class)

**File:** `graphiti_core/driver/driver.py` (lines 73-125)

Abstract base class for database drivers.

**Attributes:**

- `provider` (GraphProvider): Database type enum
- `fulltext_syntax` (str): Database-specific fulltext query syntax

### Neo4jDriver

**File:** `graphiti_core/driver/neo4j_driver.py` (lines 31-118)

Driver for Neo4j graph database.

**Constructor:**

```python
def __init__(
    self,
    uri: str,
    user: str | None,
    password: str | None,
    database: str = 'neo4j',
)
```

**Parameters:**

- `uri` (str): Neo4j connection URI (e.g., "bolt://localhost:7687")
- `user` (str | None): Database username
- `password` (str | None): Database password
- `database` (str): Database name. Defaults to 'neo4j'.

**Example:**

```python
from graphiti_core.driver import Neo4jDriver

driver = Neo4jDriver(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    database="graphiti"
)

# Use with Graphiti
graphiti = Graphiti(graph_driver=driver)
```

### FalkorDriver

**File:** `graphiti_core/driver/falkordb_driver.py`

Driver for FalkorDB (Redis-based graph database).

**Example:**

```python
from graphiti_core.driver import FalkorDriver

driver = FalkorDriver(
    host="localhost",
    port=6379,
    password=None  # optional
)

graphiti = Graphiti(graph_driver=driver)
```

### KuzuDriver

**File:** `graphiti_core/driver/kuzu_driver.py`

Driver for Kuzu embedded graph database.

**Example:**

```python
from graphiti_core.driver import KuzuDriver

driver = KuzuDriver(
    database_path="./kuzu_db"
)

graphiti = Graphiti(graph_driver=driver)
```

### NeptuneDriver

**File:** `graphiti_core/driver/neptune_driver.py`

Driver for AWS Neptune graph database.

**Example:**

```python
from graphiti_core.driver import NeptuneDriver

driver = NeptuneDriver(
    host="your-neptune-endpoint.amazonaws.com",
    port=8182,
    # Additional Neptune-specific configuration
)

graphiti = Graphiti(graph_driver=driver)
```

---

## LLM Clients

### LLMClient (Base Class)

**File:** `graphiti_core/llm_client/client.py` (lines 66-243)

Abstract base class for LLM clients.

**Constructor:**

```python
def __init__(
    self,
    config: LLMConfig | None,
    cache: bool = False
)
```

**Parameters:**

- `config` (LLMConfig | None): Configuration object
- `cache` (bool): Enable response caching. Defaults to False.

### OpenAIClient

**File:** `graphiti_core/llm_client/openai_client.py` (lines 27-100)

Client for OpenAI language models.

**Constructor:**

```python
def __init__(
    self,
    config: LLMConfig | None = None,
    cache: bool = False,
    client: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning: str = DEFAULT_REASONING,
    verbosity: str = DEFAULT_VERBOSITY,
)
```

**Example:**

```python
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig

# Basic usage with environment variable OPENAI_API_KEY
client = OpenAIClient()

# Custom configuration
config = LLMConfig(
    api_key="your-api-key",
    model="gpt-4-turbo-preview",
    temperature=0.7,
    max_tokens=4096
)

client = OpenAIClient(config=config, cache=True)

# Use with Graphiti
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=client
)
```

### AnthropicClient

**File:** `graphiti_core/llm_client/anthropic_client.py` (lines 27-100)

Client for Anthropic Claude models.

**Example:**

```python
from graphiti_core.llm_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

# Basic usage with environment variable ANTHROPIC_API_KEY
client = AnthropicClient()

# Custom configuration
config = LLMConfig(
    api_key="your-api-key",
    model="claude-sonnet-4-5-latest",
    temperature=1.0,
    max_tokens=8192
)

client = AnthropicClient(config=config)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=client
)
```

### GeminiClient

**File:** `graphiti_core/llm_client/gemini_client.py`

Client for Google Gemini models.

**Example:**

```python
from graphiti_core.llm_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(
    api_key="your-google-api-key",
    model="gemini-2.0-flash"
)

client = GeminiClient(config=config)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=client
)
```

### AzureOpenAIClient

**File:** `graphiti_core/llm_client/azure_openai_client.py`

Client for Azure OpenAI Service.

**Example:**

```python
from graphiti_core.llm_client import AzureOpenAIClient
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
    model="gpt-4"  # deployment name
)

client = AzureOpenAIClient(config=config)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=client
)
```

### GroqClient

**File:** `graphiti_core/llm_client/groq_client.py`

Client for Groq's fast inference API.

**Example:**

```python
from graphiti_core.llm_client import GroqClient
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(
    api_key="your-groq-api-key",
    model="llama-3.3-70b-versatile"
)

client = GroqClient(config=config)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=client
)
```

---

## Embedders

### EmbedderClient (Base Class)

**File:** `graphiti_core/embedder/client.py` (lines 30-39)

Abstract base class for embedding clients.

### OpenAIEmbedder

**File:** `graphiti_core/embedder/openai.py` (lines 33-67)

Embedder using OpenAI's embedding models.

**Constructor:**

```python
def __init__(
    self,
    config: OpenAIEmbedderConfig | None = None,
    client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
)
```

**Example:**

```python
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig

# Basic usage
embedder = OpenAIEmbedder()

# Custom configuration
config = OpenAIEmbedderConfig(
    api_key="your-api-key",
    embedding_model="text-embedding-3-large",
    embedding_dim=1024
)

embedder = OpenAIEmbedder(config=config)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    embedder=embedder
)
```

### AzureOpenAIEmbedder

**File:** `graphiti_core/embedder/azure_openai.py`

Embedder using Azure OpenAI embedding models.

**Example:**

```python
from graphiti_core.embedder import AzureOpenAIEmbedder

embedder = AzureOpenAIEmbedder(
    endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-key",
    deployment_name="text-embedding-ada-002"
)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    embedder=embedder
)
```

### VoyageEmbedder

**File:** `graphiti_core/embedder/voyage.py`

Embedder using Voyage AI's embedding models.

**Example:**

```python
from graphiti_core.embedder import VoyageEmbedder

embedder = VoyageEmbedder(
    api_key="your-voyage-api-key",
    model="voyage-2"
)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    embedder=embedder
)
```

### GeminiEmbedder

**File:** `graphiti_core/embedder/gemini.py`

Embedder using Google Gemini embedding models.

**Example:**

```python
from graphiti_core.embedder import GeminiEmbedder

embedder = GeminiEmbedder(
    api_key="your-google-api-key",
    model="models/embedding-001"
)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    embedder=embedder
)
```

---

## Search API

### SearchConfig

**File:** `graphiti_core/search/search_config.py` (lines 112-119)

Configuration object defining search methods and rerankers for each graph layer.

**Attributes:**

- `edge_config` (EdgeSearchConfig | None): Configuration for edge search
- `node_config` (NodeSearchConfig | None): Configuration for node search
- `episode_config` (EpisodeSearchConfig | None): Configuration for episode search
- `community_config` (CommunitySearchConfig | None): Configuration for community search
- `limit` (int): Maximum number of results. Defaults to 10.
- `reranker_min_score` (float): Minimum reranker score threshold. Defaults to 0.

### SearchFilters

**File:** `graphiti_core/search/search_filters.py` (lines 44-56)

Filters to apply to search results.

**Attributes:**

- `node_labels` (list[str] | None): Filter by node labels
- `edge_types` (list[str] | None): Filter by edge relationship names
- `valid_at` (list[list[DateFilter]] | None): Filter by valid_at dates
- `invalid_at` (list[list[DateFilter]] | None): Filter by invalid_at dates
- `created_at` (list[list[DateFilter]] | None): Filter by created_at dates
- `expired_at` (list[list[DateFilter]] | None): Filter by expired_at dates
- `edge_uuids` (list[str] | None): Filter by specific edge UUIDs

**Example:**

```python
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    ComparisonOperator
)
from datetime import datetime, timedelta

# Filter for recent facts
filters = SearchFilters(
    valid_at=[[DateFilter(
        date=datetime.now() - timedelta(days=30),
        comparison_operator=ComparisonOperator.greater_than
    )]],
    node_labels=["Person", "Organization"]
)

results = await graphiti.search(
    "who works where?",
    search_filter=filters
)

# Complex date filtering with OR logic
# Get facts that are either:
# - Created in the last week, OR
# - Valid after 2024-01-01
filters = SearchFilters(
    created_at=[
        [DateFilter(
            date=datetime.now() - timedelta(days=7),
            comparison_operator=ComparisonOperator.greater_than
        )],
        [DateFilter(
            date=datetime(2024, 1, 1),
            comparison_operator=ComparisonOperator.greater_than
        )]
    ]
)
```

### SearchResults

**File:** `graphiti_core/search/search_config.py` (lines 121-161)

Container for search results across all graph layers.

**Attributes:**

- `edges` (list[EntityEdge]): Matching relationship edges
- `edge_reranker_scores` (list[float]): Relevance scores for edges
- `nodes` (list[EntityNode]): Matching entity nodes
- `node_reranker_scores` (list[float]): Relevance scores for nodes
- `episodes` (list[EpisodicNode]): Matching episodes
- `episode_reranker_scores` (list[float]): Relevance scores for episodes
- `communities` (list[CommunityNode]): Matching communities
- `community_reranker_scores` (list[float]): Relevance scores for communities

**Methods:**

- `merge(results_list)`: Merge multiple SearchResults objects

### Predefined Search Configurations

**File:** `graphiti_core/search/search_config_recipes.py`

Graphiti provides several predefined search configurations:

**Edge Search:**
- `EDGE_HYBRID_SEARCH_RRF`: Hybrid BM25 + cosine similarity with RRF reranking
- `EDGE_HYBRID_SEARCH_MMR`: Hybrid search with MMR (diversity) reranking
- `EDGE_HYBRID_SEARCH_NODE_DISTANCE`: Rerank by distance from center node
- `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS`: Rerank by episode mention frequency
- `EDGE_HYBRID_SEARCH_CROSS_ENCODER`: Hybrid + BFS with cross-encoder reranking

**Node Search:**
- `NODE_HYBRID_SEARCH_RRF`: Hybrid search with RRF reranking
- `NODE_HYBRID_SEARCH_MMR`: Hybrid search with MMR reranking
- `NODE_HYBRID_SEARCH_NODE_DISTANCE`: Rerank by distance from center
- `NODE_HYBRID_SEARCH_EPISODE_MENTIONS`: Rerank by mention frequency
- `NODE_HYBRID_SEARCH_CROSS_ENCODER`: Hybrid + BFS with cross-encoder

**Community Search:**
- `COMMUNITY_HYBRID_SEARCH_RRF`: Hybrid search with RRF
- `COMMUNITY_HYBRID_SEARCH_MMR`: Hybrid search with MMR
- `COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER`: Hybrid with cross-encoder

**Combined Search:**
- `COMBINED_HYBRID_SEARCH_RRF`: Search all layers with RRF
- `COMBINED_HYBRID_SEARCH_MMR`: Search all layers with MMR
- `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`: Search all layers with cross-encoder

**Example:**

```python
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

# Customize a recipe
config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
config.limit = 20
config.edge_config.sim_min_score = 0.7

results = await graphiti.search_("query", config=config)
```

### Search Helper Functions

**File:** `graphiti_core/search/search_helpers.py`

**search_results_to_context_string(search_results)**

Convert SearchResults to a formatted string for use as LLM context.

```python
from graphiti_core.search.search_helpers import search_results_to_context_string

results = await graphiti.search_("recent events")
context = search_results_to_context_string(results)

# Use in LLM prompt
prompt = f"""
Based on the following context, answer the question.

{context}

Question: What happened recently?
"""
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI clients |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Anthropic clients |
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_USER` | Neo4j username | neo4j |
| `NEO4J_PASSWORD` | Neo4j password | password |
| `EMBEDDING_DIM` | Embedding vector dimensions | 1024 |
| `SEMAPHORE_LIMIT` | Max concurrent operations | System default |
| `ENTITY_INDEX_NAME` | Entity search index name | entities |
| `EPISODE_INDEX_NAME` | Episode search index name | episodes |
| `COMMUNITY_INDEX_NAME` | Community search index name | communities |
| `ENTITY_EDGE_INDEX_NAME` | Edge search index name | entity_edges |

### Constructor Options

**Graphiti Constructor:**

```python
graphiti = Graphiti(
    # Database connection (required if graph_driver is None)
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",

    # Custom clients (optional)
    llm_client=None,         # Defaults to OpenAIClient
    embedder=None,           # Defaults to OpenAIEmbedder
    cross_encoder=None,      # Defaults to OpenAIRerankerClient

    # Custom driver (optional)
    graph_driver=None,       # If provided, uri/user/password ignored

    # Configuration options
    store_raw_episode_content=True,  # Store full episode content
    max_coroutines=None,     # Override SEMAPHORE_LIMIT

    # Observability (optional)
    tracer=None,             # OpenTelemetry tracer
    trace_span_prefix='graphiti'  # Span name prefix
)
```

**LLMConfig:**

```python
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(
    api_key="your-api-key",
    model="gpt-4-turbo-preview",
    small_model="gpt-3.5-turbo",  # For simpler tasks
    base_url="https://api.openai.com",  # Custom endpoint
    temperature=0.7,
    max_tokens=4096
)
```

**EmbedderConfig:**

```python
from graphiti_core.embedder.client import EmbedderConfig

config = EmbedderConfig(
    embedding_dim=1024  # Vector dimensions
)
```

---

## Usage Patterns

### Adding Data

**Single Episode:**

```python
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType

result = await graphiti.add_episode(
    name="Document Title",
    episode_body="Content...",
    source=EpisodeType.text,
    source_description="data source",
    reference_time=datetime.now(timezone.utc)
)

print(f"Extracted {len(result.nodes)} entities")
print(f"Extracted {len(result.edges)} relationships")
```

**Bulk Episodes:**

```python
from graphiti_core.utils.bulk_utils import RawEpisode

episodes = [
    RawEpisode(
        name=f"Doc {i}",
        content="Content...",
        source=EpisodeType.text,
        source_description="bulk import",
        reference_time=datetime.now(timezone.utc)
    )
    for i in range(100)
]

result = await graphiti.add_episode_bulk(bulk_episodes=episodes)
```

**Custom Entity Types:**

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Full name")
    role: str = Field(description="Job role")
    email: str | None = Field(default=None, description="Email address")

class Project(BaseModel):
    name: str = Field(description="Project name")
    status: str = Field(description="Project status")

entity_types = {
    "Person": Person,
    "Project": Project
}

result = await graphiti.add_episode(
    name="Sprint Planning",
    episode_body="Alice will lead Project X...",
    source=EpisodeType.text,
    source_description="meeting notes",
    reference_time=datetime.now(timezone.utc),
    entity_types=entity_types
)

# Extracted nodes will have typed attributes
for node in result.nodes:
    if "Person" in node.labels:
        print(f"Person: {node.attributes.get('role')}")
```

### Searching

**Basic Search:**

```python
# Simple fact retrieval
results = await graphiti.search("What is Alice's role?")

for edge in results:
    print(edge.fact)
```

**Advanced Search:**

```python
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

# Get entities, facts, episodes, and communities
results = await graphiti.search_(
    "company restructuring",
    config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

print(f"Edges: {len(results.edges)}")
print(f"Nodes: {len(results.nodes)}")
print(f"Episodes: {len(results.episodes)}")
print(f"Communities: {len(results.communities)}")

# Use as LLM context
from graphiti_core.search.search_helpers import search_results_to_context_string

context = search_results_to_context_string(results)
# Pass context to your LLM
```

**Filtered Search:**

```python
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    ComparisonOperator
)
from datetime import datetime, timedelta

# Only recent facts about specific entity types
filters = SearchFilters(
    node_labels=["Person", "Project"],
    valid_at=[[DateFilter(
        date=datetime.now() - timedelta(days=7),
        comparison_operator=ComparisonOperator.greater_than
    )]],
    edge_types=["WORKS_ON", "MANAGES"]
)

results = await graphiti.search(
    "project updates",
    search_filter=filters,
    num_results=20
)
```

### Error Handling

```python
from graphiti_core.errors import (
    NodeNotFoundError,
    EdgeNotFoundError,
    GraphitiException
)

try:
    result = await graphiti.add_episode(
        name="Document",
        episode_body="Content...",
        source=EpisodeType.text,
        source_description="source",
        reference_time=datetime.now(timezone.utc)
    )
except GraphitiException as e:
    print(f"Graphiti error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Node retrieval
try:
    node = await EntityNode.get_by_uuid(driver, "unknown-uuid")
except NodeNotFoundError as e:
    print(f"Node not found: {e}")

# Edge retrieval
try:
    edge = await EntityEdge.get_by_uuid(driver, "unknown-uuid")
except EdgeNotFoundError as e:
    print(f"Edge not found: {e}")
```

---

## Best Practices

### 1. Connection Management

Always close connections when done:

```python
graphiti = Graphiti(uri, user, password)
try:
    # Use graphiti...
    pass
finally:
    await graphiti.close()
```

### 2. Episode Design

- **Meaningful names:** Use descriptive episode names for better traceability
- **Source descriptions:** Always provide clear source descriptions
- **Appropriate types:** Use EpisodeType.message for conversations, EpisodeType.text for documents, EpisodeType.json for structured data
- **Temporal accuracy:** Set reference_time to when the event actually occurred

### 3. Custom Types

Define custom entity and edge types for domain-specific extraction:

```python
from pydantic import BaseModel, Field

class Customer(BaseModel):
    name: str
    tier: str = Field(description="Customer tier: bronze, silver, gold")
    lifetime_value: float | None = None

entity_types = {"Customer": Customer}
```

### 4. Search Strategy

- Use `search()` for simple fact retrieval
- Use `search_()` with custom configs for advanced scenarios
- Apply filters to narrow results
- Use center nodes for context-aware reranking
- Choose appropriate rerankers (RRF for general use, MMR for diversity, cross-encoder for best quality)

### 5. Bulk Operations

For large datasets, use `add_episode_bulk()` instead of individual `add_episode()` calls:

```python
# Inefficient
for episode_data in large_dataset:
    await graphiti.add_episode(...)

# Efficient
episodes = [RawEpisode(...) for data in large_dataset]
await graphiti.add_episode_bulk(bulk_episodes=episodes)
```

### 6. Group IDs

Use group_ids for multi-tenant scenarios or logical data separation:

```python
# Separate graphs for different users/projects
await graphiti.add_episode(
    ...,
    group_id="user-123"
)

# Search within specific groups
results = await graphiti.search(
    "query",
    group_ids=["user-123", "user-456"]
)
```

### 7. Community Detection

Run community detection periodically to maintain hierarchical knowledge organization:

```python
# After adding substantial new data
communities, edges = await graphiti.build_communities()
```

### 8. Performance Tuning

- Set `max_coroutines` based on your system resources
- Use caching for LLM clients in development: `OpenAIClient(cache=True)`
- Set `store_raw_episode_content=False` if you don't need to retrieve original content
- Build indices with `await graphiti.build_indices_and_constraints()`

---

## Common Errors

### Connection Errors

**Error:** `neo4j.exceptions.ServiceUnavailable`

**Solution:** Ensure Neo4j is running and the URI is correct:

```bash
# Check Neo4j status
neo4j status

# Start Neo4j
neo4j start
```

### Authentication Errors

**Error:** `neo4j.exceptions.AuthError`

**Solution:** Verify credentials:

```python
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-actual-password"  # Check this!
)
```

### LLM API Errors

**Error:** `openai.error.AuthenticationError`

**Solution:** Set API key:

```bash
export OPENAI_API_KEY="your-key"
```

Or pass directly:

```python
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(api_key="your-key")
client = OpenAIClient(config=config)
```

### Rate Limiting

**Error:** `RateLimitError`

**Solution:** The client automatically retries with exponential backoff. For persistent issues:

```python
# Reduce concurrent operations
graphiti = Graphiti(
    uri, user, password,
    max_coroutines=2  # Lower concurrency
)
```

### Memory Errors

**Error:** `MemoryError` during bulk operations

**Solution:** Process in smaller batches:

```python
batch_size = 100
for i in range(0, len(all_episodes), batch_size):
    batch = all_episodes[i:i+batch_size]
    await graphiti.add_episode_bulk(bulk_episodes=batch)
```

### Missing Embeddings

**Error:** `Node/Edge has no embedding`

**Solution:** Generate embeddings:

```python
# For nodes
await node.generate_name_embedding(embedder)
await node.save(driver)

# For edges
await edge.generate_embedding(embedder)
await edge.save(driver)
```

### Index Not Found

**Error:** Index or constraint not found during search

**Solution:** Build indices:

```python
await graphiti.build_indices_and_constraints()
```

---

## Additional Resources

**Example Code:**
- Quickstart: `examples/quickstart/quickstart_neo4j.py`
- Azure OpenAI: `examples/azure-openai/azure_openai_neo4j.py`
- OpenTelemetry: `examples/opentelemetry/otel_stdout_example.py`

**Source Files:**
- Main API: `graphiti_core/graphiti.py`
- Nodes: `graphiti_core/nodes.py`
- Edges: `graphiti_core/edges.py`
- Search: `graphiti_core/search/`
- Drivers: `graphiti_core/driver/`
- LLM Clients: `graphiti_core/llm_client/`
- Embedders: `graphiti_core/embedder/`
