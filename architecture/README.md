# Repository Architecture Documentation

## Overview

Graphiti is a Python-based temporal knowledge graph system that automatically extracts entities and relationships from unstructured data (text, JSON, messages) and stores them in a graph database with full temporal tracking. The library provides sophisticated hybrid search capabilities combining semantic similarity, BM25 text search, and graph traversal algorithms.

**Value Proposition:**

- **Automatic Knowledge Extraction**: LLM-powered entity and relationship extraction with minimal configuration
- **Temporal Intelligence**: Track when facts become valid, change, or expire over time
- **Hybrid Search**: Combine multiple search strategies (vector similarity, BM25, graph traversal) with intelligent reranking
- **Multi-Provider Support**: Pluggable architecture for databases (Neo4j, FalkorDB, Kuzu, Neptune), LLMs (OpenAI, Anthropic, Gemini, Groq), and embedders
- **Scalable Architecture**: Async operations, bulk processing, and graph partitioning for production workloads

**Target Use Cases:**

- Conversational AI memory systems
- Document intelligence and knowledge management
- Customer interaction tracking
- Multi-source data integration with temporal reasoning
- RAG (Retrieval-Augmented Generation) applications with graph context

## Quick Start

### How to Use This Documentation

This documentation is organized into several focused documents. Use this guide to find what you need:

**If you want to...**

- **Understand the system at a glance** → Continue reading this README
- **See all components and their relationships** → [Component Inventory](docs/01_component_inventory.md)
- **Understand system architecture and design patterns** → [Architecture Diagrams](diagrams/02_architecture_diagrams.md)
- **Trace data flows through the system** → [Data Flows](docs/03_data_flows.md)
- **Use the API or integrate Graphiti** → [API Reference](docs/04_api_reference.md)

**Quick Navigation by Role:**

- **New Developers**: Start with this README, then [Architecture Diagrams](diagrams/02_architecture_diagrams.md)
- **API Users**: Jump to [API Reference](docs/04_api_reference.md)
- **Contributors**: Read [Component Inventory](docs/01_component_inventory.md) and [Data Flows](docs/03_data_flows.md)
- **Architects**: Review [Architecture Diagrams](diagrams/02_architecture_diagrams.md) and this README

## Architecture Summary

### System Layers

Graphiti follows a clean layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                           │
│  - Examples (quickstart, podcast, ecommerce)                │
│  - FastAPI HTTP Server (REST endpoints)                     │
│  - MCP Server (Model Context Protocol)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Core Library Layer                                          │
│  - Graphiti (main orchestrator class)                       │
│  - Episode Management (add, retrieve)                       │
│  - Search Engine (hybrid search + reranking)                │
│  - Community Detection (clustering)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Data Models Layer                                           │
│  - Nodes: Entity, Episodic, Community                       │
│  - Edges: Entity, Episodic, Community                       │
│  - Search Results and Configuration                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Operations Layer                                            │
│  - Node Operations (extraction, dedup, attributes)          │
│  - Edge Operations (extraction, dedup, temporal)            │
│  - Deduplication Helpers (similarity matching)              │
│  - Community Operations (detection, summarization)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Integration Layer (Abstract Interfaces)                    │
│  - Graph Drivers (database abstraction)                     │
│  - LLM Clients (language model abstraction)                 │
│  - Embedders (vector embedding abstraction)                 │
│  - Cross-Encoders (reranking abstraction)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Provider Layer (Concrete Implementations)                  │
│  - Databases: Neo4j, FalkorDB, Kuzu, Neptune                │
│  - LLMs: OpenAI, Anthropic, Gemini, Groq                    │
│  - Embeddings: OpenAI, Voyage, Gemini                       │
│  - Rerankers: OpenAI, BGE, Gemini                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**

1. **Dependency Injection**: Each layer depends only on abstractions from lower layers
2. **Plugin Architecture**: Providers can be swapped at runtime without code changes
3. **Async-First Design**: All I/O operations use async/await for concurrency
4. **Separation of Concerns**: Clear boundaries between data models, business logic, and infrastructure

### Key Design Patterns

Graphiti employs several well-established design patterns:

| Pattern | Purpose | Location |
|---------|---------|----------|
| **Abstract Factory** | Runtime provider selection for databases, LLMs, embedders | `/graphiti_core/driver/`, `/graphiti_core/llm_client/` |
| **Strategy** | Swappable search algorithms and rerankers | `/graphiti_core/search/search_config.py` |
| **Repository** | Abstract persistence operations for nodes and edges | Static methods in `/graphiti_core/nodes.py`, `/graphiti_core/edges.py` |
| **Dependency Injection** | Inject clients into Graphiti orchestrator | `/graphiti_core/graphiti.py` constructor |
| **Template Method** | Base class defines retry/caching, subclasses implement specifics | `/graphiti_core/llm_client/client.py` |
| **Decorator** | Handle multi-partition queries transparently | `/graphiti_core/decorators.py` |

**Why These Patterns:**

- **Testability**: Easy to mock dependencies and test components in isolation
- **Extensibility**: Add new providers without modifying core code
- **Maintainability**: Clear contracts between components reduce coupling
- **Flexibility**: Users can customize behavior by swapping implementations

### Technology Stack

**Core Technologies:**

- **Language**: Python 3.10+ (async/await, type hints)
- **Data Validation**: Pydantic 2.11+ (models, serialization)
- **Async Runtime**: asyncio (concurrency, coroutines)

**Graph Databases:**

- **Neo4j** (primary, production-ready): Bolt protocol, Cypher queries
- **FalkorDB** (Redis-based): Custom fulltext syntax
- **Kuzu** (embedded): Unique edge modeling
- **Neptune** (AWS managed): OpenSearch integration

**LLM Providers:**

- **OpenAI**: GPT-4, GPT-3.5 (default)
- **Anthropic**: Claude Sonnet 4.5, Claude Opus
- **Google Gemini**: Gemini 2.0 Flash
- **Groq**: Fast inference endpoints
- **Azure OpenAI**: Enterprise OpenAI hosting

**Embedding Providers:**

- **OpenAI**: text-embedding-3-large (default, 1024-dim)
- **Voyage AI**: Specialized embeddings
- **Gemini**: Google embeddings
- **Azure OpenAI**: Enterprise embeddings

**Rerankers:**

- **OpenAI** (LLM-based)
- **Gemini** (LLM-based)
- **BGE** (BAAI/bge-reranker cross-encoder)

**Supporting Technologies:**

- **diskcache**: LLM response caching
- **tenacity**: Retry logic with exponential backoff
- **numpy**: Vector operations
- **OpenTelemetry**: Distributed tracing
- **PostHog**: Usage analytics
- **FastAPI**: HTTP server framework
- **FastMCP**: Model Context Protocol server

## Component Overview

### Core Components

**Primary Entry Point:**

```python
from graphiti_core import Graphiti

# The Graphiti class is the main interface
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

**Main Modules:**

| Component | Purpose | File Reference |
|-----------|---------|----------------|
| **Graphiti** | Main orchestrator, exposes all operations | `/graphiti_core/graphiti.py:128-1264` |
| **Nodes** | Entity, Episodic, Community data models | `/graphiti_core/nodes.py` |
| **Edges** | Relationships between nodes | `/graphiti_core/edges.py` |
| **Search** | Hybrid search engine with reranking | `/graphiti_core/search/search.py` |
| **Drivers** | Database abstraction layer | `/graphiti_core/driver/` |
| **LLM Clients** | Language model abstraction | `/graphiti_core/llm_client/` |
| **Embedders** | Vector embedding generation | `/graphiti_core/embedder/` |

**Key Operations:**

- `add_episode()`: Ingest data and extract knowledge (lines 615-825)
- `add_episode_bulk()`: Batch processing for efficiency (lines 826-1011)
- `search()`: Basic hybrid search (lines 1051-1110)
- `search_()`: Advanced search with full configuration (lines 1127-1153)
- `build_communities()`: Community detection (lines 1014-1048)
- `retrieve_episodes()`: Get historical episodes (lines 577-613)

### Public API Surface

**What's Exposed to Users:**

1. **Main Class**: `Graphiti` - All operations go through this interface
2. **Node Types**: `EpisodicNode`, `EntityNode`, `CommunityNode`
3. **Edge Types**: `EpisodicEdge`, `EntityEdge`, `CommunityEdge`
4. **Search Configuration**: `SearchConfig`, `SearchFilters`, `SearchResults`
5. **Result Types**: `AddEpisodeResults`, `AddBulkEpisodeResults`, `AddTripletResults`
6. **Error Types**: 8 custom exception classes in `/graphiti_core/errors.py`

**Entry Points:**

- **Python Library**: Import from `graphiti_core` package
- **HTTP Server**: FastAPI REST endpoints at `/messages`, `/search`
- **MCP Server**: Model Context Protocol tools for AI agents
- **Examples**: Ready-to-run scripts in `/examples/`

### Provider Ecosystem

**Pluggable Providers:**

```
Graph Databases          LLM Providers           Embedders            Rerankers
┌─────────────┐         ┌──────────────┐        ┌─────────────┐      ┌──────────┐
│ Neo4j       │         │ OpenAI       │        │ OpenAI      │      │ OpenAI   │
│ FalkorDB    │         │ Anthropic    │        │ Voyage AI   │      │ Gemini   │
│ Kuzu        │         │ Gemini       │        │ Gemini      │      │ BGE      │
│ Neptune     │         │ Groq         │        │ Azure OAI   │      └──────────┘
└─────────────┘         │ Azure OAI    │        └─────────────┘
                        │ Generic OAI  │
                        └──────────────┘
```

**How to Switch Providers:**

```python
# Switch database
from graphiti_core.driver import FalkorDriver
driver = FalkorDriver(host="localhost", port=6379)

# Switch LLM
from graphiti_core.llm_client import AnthropicClient
llm = AnthropicClient()

# Switch embedder
from graphiti_core.embedder import VoyageEmbedder
embedder = VoyageEmbedder()

# Use custom providers
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=llm,
    embedder=embedder
)
```

## Data Flows

### Primary Workflows

#### 1. Episode Ingestion Flow

The most common operation - transforming raw content into structured knowledge:

```
User Input (text/JSON/message)
    ↓
Create EpisodicNode with metadata
    ↓
Retrieve context (previous N episodes)
    ↓
Extract Entities via LLM
    ├─ Use episode type-specific prompts
    ├─ Reflexion loop to catch missed entities
    └─ Map entity types to labels
    ↓
Deduplicate Nodes
    ├─ Similarity search for candidates
    ├─ Exact name + embedding distance matching
    └─ LLM-based deduplication for edge cases
    ↓
Extract Relationships via LLM
    ├─ Use resolved entity IDs
    ├─ Reflexion loop to catch missed facts
    └─ Extract temporal metadata
    ↓
Deduplicate Edges
    ├─ Generate embeddings for facts
    ├─ Search for similar edges between same nodes
    ├─ LLM deduplication
    └─ LLM invalidation (superseded facts)
    ↓
Enrich Entities
    ├─ Extract type-specific attributes
    ├─ Generate summaries
    └─ Create name embeddings
    ↓
Build Episodic Edges (MENTIONS relationships)
    ↓
Bulk Save to Graph Database
    └─ Results: nodes, edges, episode
```

**Key Characteristics:**

- LLM calls are parallelized with semaphore limiting
- Deduplication prevents entity/fact proliferation
- Temporal metadata enables time-aware querying
- Episode context enables coreference resolution

#### 2. Search/Query Flow

Sophisticated multi-method search with intelligent reranking:

```
User Query String
    ↓
Generate Query Embedding
    ↓
Parallel Search Methods
    ├─ BM25 Fulltext Search
    │   └─ Database native fulltext index
    ├─ Cosine Similarity Search
    │   └─ Vector database queries
    └─ BFS Graph Traversal
        └─ Find nearby nodes/edges
    ↓
Reciprocal Rank Fusion (RRF)
    └─ Combine rankings: 1/(k + rank)
    ↓
Reranking Strategy
    ├─ Cross-Encoder: Semantic scoring
    ├─ MMR: Diversity optimization
    ├─ Node Distance: Context-aware ranking
    └─ Episode Mentions: Frequency-based
    ↓
Filter and Limit
    ├─ Apply min_score threshold
    ├─ Apply date filters
    └─ Limit to top-k results
    ↓
Return SearchResults
    └─ Nodes, edges, episodes, communities + scores
```

**Search Flexibility:**

- 10+ predefined configurations (RRF, MMR, Cross-Encoder)
- Customizable per-layer (edge, node, episode, community)
- Filter by date ranges, labels, types
- Context-aware reranking with center nodes

#### 3. Community Building Flow

Discover clusters of related entities:

```
Trigger: graphiti.build_communities()
    ↓
Retrieve All Entities in Group
    ↓
Run Community Detection Algorithm
    └─ Graph-based clustering
    ↓
For Each Community Cluster
    ├─ Extract member entities
    ├─ Generate community summary (LLM)
    ├─ Create name embedding
    ├─ Create CommunityNode
    └─ Create CommunityEdges (HAS_MEMBER)
    ↓
Save Communities to Database
    └─ Returns: community nodes + edges
```

**Use Cases:**

- Hierarchical knowledge organization
- Topic discovery
- Entity grouping
- Improved search relevance

### Key Transformation Points

**Data Transforms Across the System:**

| Stage | Input | Output | Key Operations |
|-------|-------|--------|----------------|
| **Episode Creation** | Raw string | EpisodicNode | Parse, set metadata, generate UUID |
| **Entity Extraction** | Episode + context | EntityNode list | LLM extraction, type mapping |
| **Node Deduplication** | New entities | Resolved entities + UUID map | Similarity matching, LLM dedup |
| **Edge Extraction** | Episode + nodes | EntityEdge list | LLM relationship extraction |
| **Edge Deduplication** | New edges | Resolved edges + invalidated edges | Embedding similarity, LLM dedup |
| **Attribute Enrichment** | EntityNode | Hydrated EntityNode | LLM attributes, summary, embedding |
| **Query Embedding** | Text query | Vector | Embedder API call |
| **Search Fusion** | Multiple ranked lists | Unified ranking | RRF formula |
| **Reranking** | Query + candidates | Scored results | Cross-encoder or other strategies |

**Critical Deduplication:**

Graphiti employs multi-stage deduplication to prevent knowledge graph pollution:

1. **Similarity-based**: Exact name matches + embedding distance < threshold
2. **LLM-based**: For ambiguous cases, LLM decides if entities are the same
3. **UUID mapping**: Track original UUID → resolved UUID for consistency
4. **Edge invalidation**: Mark superseded facts as invalid (temporal tracking)

## Key Metrics

**Codebase Statistics:**

- **Total Python Modules**: 100+ files
- **Lines of Code**: ~20,000+ (estimated)
- **Main Entry Points**: 4 (library, HTTP server, MCP server, examples)
- **Public API Methods**: 15+ (Graphiti class)

**Provider Support:**

- **Graph Databases**: 4 (Neo4j, FalkorDB, Kuzu, Neptune)
- **LLM Providers**: 6 (OpenAI, Azure OpenAI, Anthropic, Gemini, Groq, Generic)
- **Embedder Providers**: 4 (OpenAI, Azure OpenAI, Gemini, Voyage)
- **Reranker Providers**: 3 (OpenAI, Gemini, BGE)

**Data Models:**

- **Node Types**: 3 (Episodic, Entity, Community)
- **Edge Types**: 3 (Episodic, Entity, Community)
- **Search Configurations**: 10+ predefined recipes
- **Error Types**: 8 custom exception classes

**Scalability Features:**

- Async operations with semaphore limiting (default: 20 concurrent)
- Bulk processing for batch operations
- Graph partitioning via group_ids
- Connection pooling for database efficiency
- LLM response caching for development

## References

### Detailed Documentation

| Document | Description | Best For |
|----------|-------------|----------|
| [Component Inventory](docs/01_component_inventory.md) | Comprehensive module-by-module breakdown with line numbers | Understanding codebase organization, finding specific components |
| [Architecture Diagrams](diagrams/02_architecture_diagrams.md) | Visual system architecture with Mermaid diagrams | Understanding layers, class hierarchies, dependencies |
| [Data Flows](docs/03_data_flows.md) | Sequence diagrams and flow analysis | Tracing operations through the system, debugging |
| [API Reference](docs/04_api_reference.md) | Complete API documentation with examples | Integrating Graphiti, using the API, code samples |

### External Resources

**Official Documentation:**

- Main Repository: [github.com/getzep/graphiti](https://github.com/getzep/graphiti)
- Issue Tracker: GitHub Issues
- Discussions: GitHub Discussions

**Provider Documentation:**

- [Neo4j Documentation](https://neo4j.com/docs/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Google Gemini Documentation](https://ai.google.dev/)
- [FalkorDB Documentation](https://www.falkordb.com/docs/)

**Related Technologies:**

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Contributing

### Extending the Documentation

This architecture documentation was auto-generated but should be kept up-to-date as the codebase evolves:

**To Update:**

1. Re-run the documentation generation tools
2. Manually update this README with new sections as needed
3. Add new diagrams to [Architecture Diagrams](diagrams/02_architecture_diagrams.md)
4. Update [Component Inventory](docs/01_component_inventory.md) with new modules
5. Extend [Data Flows](docs/03_data_flows.md) with new workflows

**Documentation Standards:**

- Use Mermaid for diagrams (they render on GitHub)
- Include file paths with line numbers for traceability
- Provide code examples for all public APIs
- Keep examples runnable and tested
- Use absolute paths (not relative) for file references

### Adding New Providers

**To Add a New Database Driver:**

1. Implement `GraphDriver` abstract class
2. Add driver to `/graphiti_core/driver/`
3. Update `GraphProvider` enum
4. Add to [Component Inventory](docs/01_component_inventory.md)
5. Update this README's provider list

**To Add a New LLM Provider:**

1. Implement `LLMClient` abstract class
2. Add client to `/graphiti_core/llm_client/`
3. Update [API Reference](docs/04_api_reference.md) with usage example
4. Add integration test

**To Add a New Embedder:**

1. Implement `EmbedderClient` abstract class
2. Add embedder to `/graphiti_core/embedder/`
3. Document in [API Reference](docs/04_api_reference.md)

## Generated

**Auto-Generated Documentation**

- **Generated**: 2025-11-29 17:56 UTC
- **Source**: Graphiti repository analysis
- **Tools**: Repository analysis scripts + manual synthesis
- **Version**: Based on commit `422558d` (main branch)

**Documentation Status:**

- Component Inventory: Complete
- Architecture Diagrams: Complete
- Data Flows: Complete
- API Reference: Complete
- README (this file): Synthesized from all sources

**Note**: This documentation reflects the codebase state at generation time. For the most current information, always refer to the source code and inline comments.

---

**Quick Links:**

- [Component Inventory →](docs/01_component_inventory.md)
- [Architecture Diagrams →](diagrams/02_architecture_diagrams.md)
- [Data Flows →](docs/03_data_flows.md)
- [API Reference →](docs/04_api_reference.md)
