# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graphiti is a Python framework for building temporally-aware knowledge graphs designed for AI agents. It enables real-time incremental updates to knowledge graphs without batch recomputation, making it suitable for dynamic environments.

Key features:

- Bi-temporal data model with explicit tracking of event occurrence times (`valid_at` for when facts are true, `created_at` for ingestion time)
- Hybrid retrieval combining semantic embeddings, keyword search (BM25), and graph traversal
- Support for custom entity definitions via Pydantic models
- Integration with Neo4j, FalkorDB, Kuzu, and Amazon Neptune as graph storage backends
- Optional OpenTelemetry distributed tracing support

## Development Commands

### Main Development Commands (run from project root)

```bash
# Install dependencies
uv sync --extra dev

# Format code (ruff import sorting + formatting)
make format

# Lint code (ruff + pyright type checking)
make lint

# Run tests (excludes integration tests by default)
make test

# Run all checks (format, lint, test)
make check
```

### Server Development (run from server/ directory)

```bash
cd server/
uv sync --extra dev
uvicorn graph_service.main:app --reload
make format && make lint && make test
```

### MCP Server Development (run from mcp_server/ directory)

```bash
cd mcp_server/
uv sync
docker-compose up
```

## Code Architecture

### Core Library (`graphiti_core/`)

- **Main Entry Point**: `graphiti.py` - The `Graphiti` class orchestrates all functionality
- **Graph Storage**: `driver/` - Database drivers for Neo4j, FalkorDB, Kuzu, and Neptune
- **LLM Integration**: `llm_client/` - Clients for OpenAI, Azure OpenAI, Anthropic, Gemini, Groq
- **Embeddings**: `embedder/` - Embedding clients (OpenAI, Azure, Voyage, Gemini)
- **Cross-Encoder**: `cross_encoder/` - Reranking clients for search result refinement
- **Graph Elements**: `nodes.py`, `edges.py` - Core graph data structures
- **Search**: `search/` - Hybrid search with configurable strategies and recipes
- **Prompts**: `prompts/` - LLM prompts for entity extraction, deduplication, summarization
- **Utilities**: `utils/` - Maintenance operations, bulk processing, datetime handling

### Data Flow: Episode Ingestion Pipeline

When `add_episode()` is called, data flows through these stages:

1. **Episode Creation** - `EpisodicNode` created with content, timestamps, and `group_id`
2. **Node Extraction** (`utils/maintenance/node_operations.py`) - LLM extracts entities from episode content
3. **Node Resolution** - Deduplicates against existing nodes, creates UUID mappings
4. **Edge Extraction** (`utils/maintenance/edge_operations.py`) - LLM extracts relationships between entities
5. **Edge Resolution** - Deduplicates edges, handles temporal invalidation of contradicting facts
6. **Attribute Extraction** - Hydrates nodes with additional attributes from custom entity types
7. **Persistence** - Saves nodes, edges, episodic edges (MENTIONS relationships), and updates embeddings

### Graph Data Model

**Node Types:**
- `EpisodicNode` - Represents ingested content (messages, JSON, text)
- `EntityNode` - Extracted entities with name, summary, and optional custom attributes
- `CommunityNode` - Clustered groups of related entities with summaries

**Edge Types:**
- `EpisodicEdge` - MENTIONS relationship connecting episodes to entities
- `EntityEdge` - RELATES_TO facts between entities with temporal validity tracking
- `CommunityEdge` - HAS_MEMBER linking communities to their entity members

**Temporal Model:**
- `valid_at` - When the fact/entity was true in the real world
- `invalid_at` - When the fact became no longer true (for contradictions)
- `created_at` - When the data was ingested into Graphiti

### Search System (`search/`)

Search is configured via `SearchConfig` objects that specify:
- **Search methods**: `bm25` (fulltext), `cosine_similarity` (semantic), `bfs` (graph traversal)
- **Rerankers**: `rrf`, `mmr`, `cross_encoder`, `node_distance`, `episode_mentions`
- **Targets**: edges, nodes, episodes, communities (each independently configurable)

Pre-built recipes in `search_config_recipes.py`:
- `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` - Full hybrid search with cross-encoder reranking (recommended)
- `EDGE_HYBRID_SEARCH_RRF` - Edge-only search with RRF fusion
- `NODE_HYBRID_SEARCH_RRF` - Node-only search with RRF fusion

### Graph Partitioning with `group_id`

The `group_id` parameter partitions data within the graph. When provided:
- For Neo4j/FalkorDB: Creates separate databases per group
- All queries filter by `group_id` to isolate data
- Use `handle_multiple_group_ids` decorator for queries spanning multiple groups

### Server (`server/`)

FastAPI REST API service with routers for ingestion and retrieval, and DTOs for API contracts.

### MCP Server (`mcp_server/`)

Model Context Protocol server for AI assistant integration, with Docker support.

## Testing

```bash
# Run unit tests only (default)
make test

# Run specific test file
pytest tests/test_specific_file.py

# Run specific test method
pytest tests/test_file.py::test_method_name

# Run integration tests (requires database and API keys)
pytest -m integration

# Run only unit tests explicitly
pytest -m "not integration"
```

Integration tests are marked with `@pytest.mark.integration` and require environment variables:
```bash
export TEST_OPENAI_API_KEY=...
export TEST_OPENAI_MODEL=...
export TEST_ANTHROPIC_API_KEY=...
export TEST_URI=neo4j://...
export TEST_USER=...
export TEST_PASSWORD=...
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for default LLM inference and embeddings
- `SEMAPHORE_LIMIT` - Controls concurrent operations (default: 10). Increase for faster ingestion if your LLM provider allows higher throughput; decrease if you hit 429 rate limit errors.
- `USE_PARALLEL_RUNTIME` - Optional boolean for Neo4j parallel runtime (enterprise only)
- Provider-specific keys: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `VOYAGE_API_KEY`

### Database Setup

Database names are configured in driver constructors. Override by passing `database` parameter:

- **Neo4j**: Version 5.26+ required, defaults to `neo4j` database
- **FalkorDB**: Version 1.1.2+ as alternative, defaults to `default_db`
- **Kuzu**: Version 0.11.2+ embedded graph database
- **Amazon Neptune**: Requires Neptune Database/Analytics + OpenSearch Serverless

## Development Guidelines

### Code Style

- Ruff for formatting and linting (line length: 100, single quotes)
- Pyright for type checking (`basic` mode for core, `standard` for server)

### LLM Provider Support

Works best with services supporting structured output (OpenAI, Gemini). Other providers may cause schema validation issues, especially with smaller models.

### Third-Party Integrations

All integrations must be optional dependencies. Use the `TYPE_CHECKING` pattern for conditional imports with clear error messages. See CONTRIBUTING.md for detailed guidelines.

## Detailed Architecture Documentation

For comprehensive architecture documentation including diagrams, data flows, and API reference, see the `architecture/` directory:

- `architecture/README.md` - System overview and design patterns
- `architecture/diagrams/02_architecture_diagrams.md` - Mermaid diagrams for class hierarchies and system layers
- `architecture/docs/01_component_inventory.md` - Module-by-module breakdown with line numbers
- `architecture/docs/03_data_flows.md` - Sequence diagrams for ingestion, search, MCP, and HTTP flows
- `architecture/docs/04_api_reference.md` - Complete API documentation with examples

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                           │
│  - Examples, FastAPI Server, MCP Server                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Core Library Layer                                          │
│  - Graphiti orchestrator, Episode/Search/Community mgmt     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Data Models Layer                                           │
│  - Nodes (Entity, Episodic, Community)                      │
│  - Edges (Entity, Episodic, Community)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Operations Layer                                            │
│  - Node/Edge extraction, deduplication, communities         │
│  - utils/maintenance/*.py                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Integration Layer (Abstract Interfaces)                     │
│  - GraphDriver, LLMClient, EmbedderClient, CrossEncoderClient│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Provider Layer (Concrete Implementations)                   │
│  - Neo4j/FalkorDB/Kuzu/Neptune, OpenAI/Anthropic/Gemini/Groq│
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Purpose | Location |
|---------|---------|----------|
| **Abstract Factory** | Runtime provider selection | `driver/`, `llm_client/`, `embedder/` |
| **Strategy** | Swappable search algorithms and rerankers | `search/search_config.py` |
| **Repository** | Abstract persistence for nodes/edges | Static methods in `nodes.py`, `edges.py` |
| **Dependency Injection** | Inject clients into Graphiti | `graphiti.py` constructor |
| **Template Method** | Base retry/caching, subclasses implement specifics | `llm_client/client.py` |
| **Decorator** | Handle multi-partition queries | `decorators.py` |

### Adding New Providers

**New Database Driver:**
1. Implement `GraphDriver` abstract class in `driver/`
2. Add provider to `GraphProvider` enum in `driver/driver.py`
3. Implement `GraphDriverSession` for session management
4. Add database-specific query builders in `models/`

**New LLM Provider:**
1. Implement `LLMClient` abstract class in `llm_client/`
2. Override `_generate_response()` method
3. Add to optional dependencies in `pyproject.toml`
4. Use `TYPE_CHECKING` pattern for conditional imports

**New Embedder:**
1. Implement `EmbedderClient` abstract class in `embedder/`
2. Implement `create()` and optionally `create_batch()` methods
3. Add to optional dependencies

### Error Handling Patterns

**Critical Errors (halt processing):**
- Database connection failures
- Invalid credentials
- Missing required parameters
- Validation failures (`validate_entity_types`, `validate_group_id`)

**Non-Critical Errors (log and continue):**
- Invalid entity IDs from LLM (skip edge, continue)
- Malformed deduplication responses (skip resolution)
- Telemetry failures (silently ignore)
- Individual embedding failures

**Retryable Errors (tenacity retry):**
- Rate limit errors (429)
- Temporary network failures
- Database connection timeouts

### Ingestion Pipeline Details

The `add_episode()` pipeline includes LLM reflexion loops to catch missed entities/facts:

1. **Entity Extraction** → Reflexion check (up to MAX_REFLEXION_ITERATIONS)
2. **Relationship Extraction** → Reflexion check for missed facts
3. **Deduplication** (nodes then edges):
   - Similarity-based: exact name + embedding distance
   - LLM-based: for ambiguous cases
   - UUID mapping: original → resolved UUIDs
4. **Edge Invalidation**: Mark superseded facts with `invalid_at` timestamp