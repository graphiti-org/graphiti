# Component Inventory

## Overview

Graphiti is a temporal graph building library designed to create and maintain knowledge graphs from episodic data. The codebase is organized into several key layers:

1. **Core Graph Engine** (`graphiti_core/`) - Main library providing graph operations, LLM integration, and temporal knowledge management
2. **MCP Server** (`mcp_server/`) - Model Context Protocol server exposing Graphiti functionality
3. **HTTP API Server** (`server/`) - FastAPI-based REST API for graph operations
4. **Examples** (`examples/`) - Sample implementations and use cases

The architecture follows a plugin-based design supporting multiple graph databases (Neo4j, FalkorDB, Kuzu, Neptune), LLM providers (OpenAI, Anthropic, Gemini, Groq), and embedders (OpenAI, Voyage, Gemini).

## Public API

### Primary Entry Point

**Main Class**: `Graphiti`
- File: `graphiti_core/graphiti.py` (lines 128-1264)
- Description: Primary interface for all graph operations, manages connections, processing, and querying

### Core Modules

#### 1. Graph Nodes (`graphiti_core/nodes.py`)
- **Node** (line 87): Abstract base class for all node types
- **EpisodicNode** (line 295): Represents temporal episodes/events
- **EntityNode** (line 435): Represents entities extracted from episodes
- **CommunityNode** (line 591): Represents clusters of related entities
- **EpisodeType** (line 51): Enum for episode types (message, json, text)

#### 2. Graph Edges (`graphiti_core/edges.py`)
- **Edge** (line 45): Abstract base class for all edge types
- **EpisodicEdge** (line 131): Links episodes to mentioned entities
- **EntityEdge** (line 221): Represents relationships between entities
- **CommunityEdge** (line 480): Links communities to member entities

#### 3. Graph Drivers (`graphiti_core/driver/`)

**Base Interface**:
- **GraphDriver** (`driver.py`, line 73): Abstract driver interface
- **GraphDriverSession** (`driver.py`, line 49): Abstract session interface
- **GraphProvider** (`driver.py`, line 42): Enum for database types

**Concrete Implementations**:
- **Neo4jDriver** (`neo4j_driver.py`): Neo4j database driver
- **FalkorDBDriver** (`falkordb_driver.py`): FalkorDB graph database driver
- **KuzuDriver** (`kuzu_driver.py`): Kuzu embedded graph database driver
- **NeptuneDriver** (`neptune_driver.py`): AWS Neptune database driver

#### 4. LLM Clients (`graphiti_core/llm_client/`)

**Base Interface**:
- **LLMClient** (`client.py`, line 66): Abstract LLM client interface

**Implementations**:
- **OpenAIClient** (`openai_client.py`): OpenAI API client
- **AzureOpenAIClient** (`azure_openai_client.py`): Azure OpenAI client
- **AnthropicClient** (`anthropic_client.py`): Anthropic Claude client
- **GeminiClient** (`gemini_client.py`): Google Gemini client
- **GroqClient** (`groq_client.py`): Groq API client
- **OpenAIGenericClient** (`openai_generic_client.py`): Generic OpenAI-compatible client

#### 5. Embedders (`graphiti_core/embedder/`)

**Base Interface**:
- **EmbedderClient** (`client.py`, line 30): Abstract embedder interface

**Implementations**:
- **OpenAIEmbedder** (`openai.py`): OpenAI embeddings
- **AzureOpenAIEmbedder** (`azure_openai.py`): Azure OpenAI embeddings
- **GeminiEmbedder** (`gemini.py`): Google Gemini embeddings
- **VoyageEmbedder** (`voyage.py`): Voyage AI embeddings

#### 6. Cross-Encoder Rerankers (`graphiti_core/cross_encoder/`)

**Base Interface**:
- **CrossEncoderClient** (`client.py`, line 20): Abstract reranker interface

**Implementations**:
- **OpenAIRerankerClient** (`openai_reranker_client.py`): OpenAI-based reranking
- **GeminiRerankerClient** (`gemini_reranker_client.py`): Gemini-based reranking
- **BGERerankerClient** (`bge_reranker_client.py`): BGE model reranking

#### 7. Search System (`graphiti_core/search/`)

**Main Functions**:
- **search** (`search.py`, line 68): Main hybrid search function
- **edge_search** (`search.py`, line 186): Search for entity edges
- **node_search** (`search.py`, line 309): Search for entity nodes
- **episode_search** (`search.py`, line 419): Search for episodes
- **community_search** (`search.py`, line 468): Search for communities

**Configuration**:
- **SearchConfig** (`search_config.py`): Search configuration options
- **SearchFilters** (`search_filters.py`): Filtering options for search
- **SearchResults** (`search_config.py`): Search result container

**Search Recipes** (`search_config_recipes.py`):
- **EDGE_HYBRID_SEARCH_RRF**: Reciprocal Rank Fusion for edges
- **EDGE_HYBRID_SEARCH_NODE_DISTANCE**: Distance-based edge reranking
- **COMBINED_HYBRID_SEARCH_CROSS_ENCODER**: Cross-encoder based search
- **NODE_HYBRID_SEARCH_RRF**: RRF for nodes

### Public API Methods (Graphiti class)

#### Core Operations
- **add_episode** (`graphiti.py`, line 615): Add single episode to graph
- **add_episode_bulk** (`graphiti.py`, line 826): Add multiple episodes efficiently
- **add_triplet** (`graphiti.py`, line 1169): Add entity-edge-entity triplet
- **remove_episode** (`graphiti.py`, line 1235): Delete episode and cleanup

#### Retrieval Operations
- **search** (`graphiti.py`, line 1051): Basic hybrid search
- **search_** (`graphiti.py`, line 1127): Advanced search with full config
- **retrieve_episodes** (`graphiti.py`, line 577): Get recent episodes
- **get_nodes_and_edges_by_episode** (`graphiti.py`, line 1155): Get episode data

#### Graph Operations
- **build_communities** (`graphiti.py`, line 1014): Generate community clusters
- **build_indices_and_constraints** (`graphiti.py`, line 321): Setup database schema
- **close** (`graphiti.py`, line 289): Close database connection

### Result Types

**Response Models**:
- **AddEpisodeResults** (`graphiti.py`, line 105): Episode processing results
- **AddBulkEpisodeResults** (`graphiti.py`, line 114): Bulk processing results
- **AddTripletResults** (`graphiti.py`, line 123): Triplet addition results
- **SearchResults** (`search_config.py`): Search query results

### Error Types (`graphiti_core/errors.py`)

- **GraphitiError** (line 18): Base exception
- **EdgeNotFoundError** (line 22): Edge lookup failure
- **EdgesNotFoundError** (line 30): Multiple edges not found
- **GroupsEdgesNotFoundError** (line 38): No edges for group
- **GroupsNodesNotFoundError** (line 46): No nodes for group
- **NodeNotFoundError** (line 54): Node lookup failure
- **SearchRerankerError** (line 62): Reranker errors
- **EntityTypeValidationError** (line 70): Invalid entity type
- **GroupIdValidationError** (line 78): Invalid group ID

## Internal Implementation

### Core Modules

#### 1. Graph Data Types (`graphiti_core/graphiti_types.py`)
- **GraphitiClients**: Container for driver, LLM, embedder, cross-encoder clients

#### 2. Maintenance Operations (`graphiti_core/utils/maintenance/`)

**Node Operations** (`node_operations.py`):
- **extract_nodes**: Extract entities from episodes (line 88)
- **resolve_extracted_nodes**: Deduplicate and merge nodes
- **extract_attributes_from_nodes**: Enrich node attributes

**Edge Operations** (`edge_operations.py`):
- **extract_edges**: Extract relationships from episodes (line 89)
- **resolve_extracted_edges**: Deduplicate and validate edges
- **build_episodic_edges**: Create episode-entity links (line 51)
- **build_community_edges**: Create community-entity links (line 71)

**Community Operations** (`community_operations.py`):
- **build_communities**: Generate community clusters
- **remove_communities**: Clear existing communities
- **update_community**: Update single community

**Temporal Operations** (`temporal_operations.py`):
- Edge date extraction and validation
- Temporal consistency checks

**Graph Data Operations** (`graph_data_operations.py`):
- **retrieve_episodes**: Fetch episode history
- **clear_data**: Delete graph data by group

**Deduplication Helpers** (`dedup_helpers.py`):
- Similarity-based deduplication
- UUID mapping and resolution

#### 3. Prompt System (`graphiti_core/prompts/`)

**Prompt Library Structure** (`lib.py`):
- Centralized prompt management
- Version control for prompts
- Multi-language support

**Prompt Modules**:
- **extract_nodes.py**: Entity extraction prompts
- **extract_edges.py**: Relationship extraction prompts
- **dedupe_nodes.py**: Node deduplication prompts
- **dedupe_edges.py**: Edge deduplication prompts
- **extract_edge_dates.py**: Temporal extraction prompts
- **invalidate_edges.py**: Edge invalidation prompts
- **summarize_nodes.py**: Node summarization prompts
- **eval.py**: Evaluation prompts

**Models** (`models.py`):
- **Message**: LLM message format
- **PromptFunction**: Prompt template type

#### 4. Database Query Builders (`graphiti_core/models/`)

**Node Queries** (`nodes/node_db_queries.py`):
- Save queries for episodic, entity, community nodes
- Return queries optimized per database provider

**Edge Queries** (`edges/edge_db_queries.py`):
- Save queries for episodic, entity, community edges
- Provider-specific query generation

#### 5. Bulk Operations (`graphiti_core/utils/bulk_utils.py`)
- **RawEpisode**: Episode input format
- **add_nodes_and_edges_bulk**: Batch database insertion
- **extract_nodes_and_edges_bulk**: Parallel extraction
- **dedupe_nodes_bulk**: Batch deduplication
- **dedupe_edges_bulk**: Batch edge deduplication
- **resolve_edge_pointers**: UUID pointer resolution
- **retrieve_previous_episodes_bulk**: Batch episode retrieval

#### 6. Search Utilities (`graphiti_core/search/search_utils.py`)
- **node_fulltext_search**: BM25 node search
- **node_similarity_search**: Vector similarity search
- **node_bfs_search**: Breadth-first traversal
- **edge_fulltext_search**: BM25 edge search
- **edge_similarity_search**: Edge vector search
- **edge_bfs_search**: Edge traversal
- **episode_fulltext_search**: Episode text search
- **community_fulltext_search**: Community search
- **rrf**: Reciprocal Rank Fusion
- **maximal_marginal_relevance**: MMR algorithm
- **node_distance_reranker**: Distance-based reranking
- **episode_mentions_reranker**: Mention frequency reranking

#### 7. Telemetry (`graphiti_core/telemetry/`)
- **capture_event** (`telemetry.py`): Analytics event tracking
- PostHog integration for usage analytics

#### 8. Tracing (`graphiti_core/tracer.py`)
- **Tracer**: OpenTelemetry tracing interface
- **NoOpTracer**: Disabled tracing implementation
- **create_tracer**: Tracer factory function

### Helper/Utility Modules

#### Date/Time Utilities (`graphiti_core/utils/datetime_utils.py`)
- **utc_now**: Current UTC timestamp
- **ensure_utc**: Timezone normalization
- **parse_db_date**: Database date parsing

#### Text Utilities (`graphiti_core/utils/text_utils.py`)
- **truncate_at_sentence**: Sentence-aware truncation
- Character limit constants

#### Ontology Utilities (`graphiti_core/utils/ontology_utils/`)
- **validate_entity_types** (`entity_types_utils.py`): Entity type validation
- Protected attribute checking

#### General Helpers (`graphiti_core/helpers.py`)
- **semaphore_gather** (line 106): Bounded concurrent execution
- **parse_db_date** (line 41): Database date parsing
- **get_default_group_id** (line 51): Default group ID by provider
- **lucene_sanitize** (line 62): Lucene query escaping
- **normalize_l2** (line 99): L2 vector normalization
- **validate_group_id** (line 119): Group ID validation
- **validate_excluded_entity_types** (line 145): Entity exclusion validation

#### Decorators (`graphiti_core/decorators.py`)
- **handle_multiple_group_ids**: Multi-partition support

#### LLM Client Utilities (`graphiti_core/llm_client/`)
- **Config** (`config.py`): LLM configuration models
- **Errors** (`errors.py`): LLM-specific exceptions
- **Utils** (`utils.py`): Client helper functions

## Entry Points

### 1. Python Library Entry Point

**Main Entry**: `graphiti_core/__init__.py` (lines 1-3)
```python
from .graphiti import Graphiti
__all__ = ['Graphiti']
```

**Usage Pattern**:
```python
from graphiti_core import Graphiti

graphiti = Graphiti(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="password"
)
```

### 2. MCP Server Entry Point

**Main File**: `mcp_server/main.py`
- FastMCP server exposing Graphiti via Model Context Protocol
- Async queue-based episode processing
- HTTP and SSE transport support

**Key Components**:
- **QueueService** (`services/queue_service.py`): Async episode processing queue
- **Factories** (`services/factories.py`): Component factory pattern
- **Configuration** (`config/schema.py`): Server configuration models

### 3. HTTP API Server Entry Point

**Main File**: `server/graph_service/main.py` (lines 1-30)
- FastAPI application with lifespan management
- Healthcheck endpoint (line 27)

**Routers**:
- **Ingest** (`routers/ingest.py`): Episode ingestion endpoints
- **Retrieve** (`routers/retrieve.py`): Search and query endpoints

**Configuration**:
- **Settings** (`config.py`): Environment-based configuration
- **DTOs** (`dto/`): Request/response models

### 4. Example Entry Points

Located in `examples/`:

- **quickstart/quickstart_neo4j.py**: Basic Neo4j usage
- **quickstart/quickstart_falkordb.py**: FalkorDB usage
- **quickstart/quickstart_neptune.py**: AWS Neptune usage
- **podcast/podcast_runner.py**: Podcast transcript processing
- **ecommerce/runner.py**: E-commerce data processing
- **wizard_of_oz/runner.py**: Story/narrative processing
- **azure-openai/azure_openai_neo4j.py**: Azure OpenAI integration
- **opentelemetry/otel_stdout_example.py**: OpenTelemetry tracing demo

## Module Dependency Summary

### Dependency Layers

```
Layer 1 (Foundation):
- errors.py
- helpers.py
- graphiti_types.py
- utils/datetime_utils.py
- utils/text_utils.py

Layer 2 (Data Models):
- nodes.py (depends on: driver, embedder, errors, helpers)
- edges.py (depends on: driver, embedder, errors, helpers)
- models/ (node/edge database query builders)

Layer 3 (Infrastructure):
- driver/* (database abstraction)
- embedder/* (embedding providers)
- llm_client/* (LLM providers)
- cross_encoder/* (reranker providers)

Layer 4 (Prompts & Processing):
- prompts/* (LLM prompt templates)
- utils/maintenance/* (graph operations)
- search/* (search and retrieval)

Layer 5 (Core API):
- graphiti.py (orchestrates all lower layers)

Layer 6 (Servers):
- mcp_server/ (MCP protocol server)
- server/ (HTTP API server)
```

### Key Dependencies Between Modules

1. **graphiti.py** depends on:
   - driver (GraphDriver)
   - llm_client (LLMClient)
   - embedder (EmbedderClient)
   - cross_encoder (CrossEncoderClient)
   - nodes (EntityNode, EpisodicNode, CommunityNode)
   - edges (EntityEdge, EpisodicEdge, CommunityEdge)
   - search (search function, SearchConfig, SearchFilters)
   - utils/maintenance/* (all maintenance operations)
   - utils/bulk_utils (bulk processing)

2. **search/** depends on:
   - driver (for database queries)
   - cross_encoder (for reranking)
   - embedder (for vector generation)
   - nodes/edges (data models)

3. **utils/maintenance/** depends on:
   - llm_client (for extraction prompts)
   - embedder (for similarity)
   - driver (for database access)
   - prompts (prompt templates)

4. **Servers** depend on:
   - graphiti.py (core API)
   - All factory patterns for component creation

### Database Provider Abstraction

All database-specific code isolated to:
- `driver/*_driver.py` (provider implementations)
- `models/*/db_queries.py` (query builders)
- `driver/graph_operations/` (provider-specific operations)
- `driver/search_interface/` (provider-specific search)

### LLM Provider Abstraction

All LLM-specific code isolated to:
- `llm_client/*_client.py` (provider implementations)
- Common interface: `llm_client/client.py`
- Configuration: `llm_client/config.py`

### Embedder Provider Abstraction

All embedder-specific code isolated to:
- `embedder/*.py` (provider implementations)
- Common interface: `embedder/client.py`

## Configuration and Environment

### Environment Variables

Key configuration from `graphiti_core/helpers.py`:
- **SEMAPHORE_LIMIT** (default: 20): Concurrent operation limit
- **USE_PARALLEL_RUNTIME** (default: False): Parallel processing flag
- **MAX_REFLEXION_ITERATIONS** (default: 0): LLM reflexion iterations

From `graphiti_core/embedder/client.py`:
- **EMBEDDING_DIM** (default: 1024): Embedding dimensions

From `graphiti_core/driver/driver.py`:
- **ENTITY_INDEX_NAME** (default: 'entities'): Search index name
- **EPISODE_INDEX_NAME** (default: 'episodes'): Episode index name
- **COMMUNITY_INDEX_NAME** (default: 'communities'): Community index name
- **ENTITY_EDGE_INDEX_NAME** (default: 'entity_edges'): Edge index name

### Dependencies (from pyproject.toml)

**Core Dependencies**:
- pydantic >= 2.11.5 (data validation)
- neo4j >= 5.26.0 (default graph database)
- diskcache >= 5.6.3 (LLM response caching)
- openai >= 1.91.0 (default LLM/embedder)
- tenacity >= 9.0.0 (retry logic)
- numpy >= 1.0.0 (vector operations)
- python-dotenv >= 1.0.1 (environment config)
- posthog >= 3.0.0 (telemetry)

**Optional Provider Dependencies**:
- anthropic >= 0.49.0
- groq >= 0.2.0
- google-genai >= 1.8.0
- kuzu >= 0.11.3
- falkordb >= 1.1.2
- voyageai >= 0.2.3
- sentence-transformers >= 3.2.1
- boto3 + opensearch-py (for Neptune)
- opentelemetry-api/sdk >= 1.20.0 (tracing)

## Summary Statistics

- **Total Python modules**: ~100+ files
- **Main entry points**: 4 (library, MCP server, HTTP server, examples)
- **Database providers**: 4 (Neo4j, FalkorDB, Kuzu, Neptune)
- **LLM providers**: 6 (OpenAI, Azure OpenAI, Anthropic, Gemini, Groq, Generic)
- **Embedder providers**: 4 (OpenAI, Azure OpenAI, Gemini, Voyage)
- **Reranker providers**: 3 (OpenAI, Gemini, BGE)
- **Public API methods**: 15+ (Graphiti class)
- **Search configurations**: 10+ recipes
- **Node types**: 3 (Episodic, Entity, Community)
- **Edge types**: 3 (Episodic, Entity, Community)
- **Error types**: 8 custom exceptions
