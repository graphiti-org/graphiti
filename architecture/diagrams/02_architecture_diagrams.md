# Architecture Diagrams

## Overview

Graphiti is a temporal knowledge graph system built on Python that provides an advanced framework for managing episodic knowledge with semantic understanding. The architecture follows a layered design pattern with clear separation of concerns:

- **Core Library Layer**: The main `graphiti_core` package containing all business logic
- **Server Layer**: FastAPI-based HTTP server and MCP (Model Context Protocol) server
- **Integration Layer**: Database drivers, LLM clients, embedders, and rerankers
- **Examples Layer**: Demonstration applications showing various use cases

The system is designed around a plugin architecture where different providers (databases, LLMs, embedders) can be swapped through abstract interfaces. The core orchestrates episodic data ingestion, entity/edge extraction, deduplication, and semantic search operations.

## System Architecture

The system is organized into distinct layers that handle different aspects of the knowledge graph lifecycle:

```mermaid
graph TB
    subgraph "Application Layer"
        A1[Examples]
        A2[FastAPI Server]
        A3[MCP Server]
    end

    subgraph "Core Library Layer"
        B1[Graphiti Main Class]
        B2[Episode Management]
        B3[Search Engine]
        B4[Community Detection]
    end

    subgraph "Data Models Layer"
        C1[Nodes: Entity/Episodic/Community]
        C2[Edges: Entity/Episodic/Community]
        C3[Search Results]
    end

    subgraph "Operations Layer"
        D1[Node Operations]
        D2[Edge Operations]
        D3[Deduplication]
        D4[Temporal Operations]
    end

    subgraph "Integration Layer"
        E1[Graph Drivers]
        E2[LLM Clients]
        E3[Embedders]
        E4[Rerankers]
    end

    subgraph "Provider Layer"
        F1[Neo4j/FalkorDB/Kuzu/Neptune]
        F2[OpenAI/Anthropic/Gemini/Groq]
        F3[OpenAI/Voyage/Gemini Embeddings]
        F4[OpenAI/BGE/Gemini Reranking]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B2 --> C1
    B2 --> C2
    B3 --> C3
    B2 --> D1
    B2 --> D2
    B2 --> D3
    B2 --> D4
    D1 --> E1
    D1 --> E2
    D1 --> E3
    D2 --> E1
    D2 --> E2
    D2 --> E4
    B3 --> E1
    B3 --> E3
    B3 --> E4
    E1 --> F1
    E2 --> F2
    E3 --> F3
    E4 --> F4
```

**Layer Descriptions:**

- **Application Layer**: Entry points for using Graphiti (examples, servers, direct library usage)
- **Core Library Layer**: Main business logic orchestrating all operations
- **Data Models Layer**: Pydantic models representing graph entities and search results
- **Operations Layer**: Specialized modules for extraction, deduplication, and maintenance
- **Integration Layer**: Abstract interfaces for pluggable providers
- **Provider Layer**: Concrete implementations of external services

## Component Relationships

This diagram shows how major components interact during the typical episode ingestion workflow:

```mermaid
graph LR
    subgraph "Entry Point"
        G[Graphiti]
    end

    subgraph "Clients"
        GD[GraphDriver]
        LLM[LLMClient]
        EMB[Embedder]
        CE[CrossEncoder]
    end

    subgraph "Core Operations"
        NO[Node Operations]
        EO[Edge Operations]
        DD[Deduplication]
        CO[Community Operations]
    end

    subgraph "Data Processing"
        EXT[Extraction Prompts]
        SR[Search Engine]
        TE[Temporal Operations]
    end

    subgraph "Storage"
        DB[(Graph Database)]
    end

    G -->|contains| GD
    G -->|contains| LLM
    G -->|contains| EMB
    G -->|contains| CE

    G -->|orchestrates| NO
    G -->|orchestrates| EO
    G -->|orchestrates| DD
    G -->|orchestrates| CO

    NO -->|uses| LLM
    NO -->|uses| EMB
    NO -->|queries| GD

    EO -->|uses| LLM
    EO -->|uses| EMB
    EO -->|queries| GD

    DD -->|uses| LLM
    DD -->|queries| GD

    CO -->|uses| LLM
    CO -->|uses| EMB
    CO -->|queries| GD

    NO -->|formats| EXT
    EO -->|formats| EXT

    SR -->|uses| EMB
    SR -->|uses| CE
    SR -->|queries| GD

    TE -->|queries| GD

    GD -->|reads/writes| DB
```

**Key Interaction Patterns:**

1. **Graphiti orchestration**: The main Graphiti class coordinates all operations through injected client dependencies
2. **Operation layer**: Specialized modules handle node extraction, edge extraction, deduplication, and communities
3. **LLM integration**: Operations use LLM clients with prompt templates for entity/relation extraction
4. **Embedding generation**: Embedders create vector representations for semantic search
5. **Graph persistence**: All operations ultimately read/write through the graph driver abstraction

## Class Hierarchies

### Node Types

```mermaid
classDiagram
    class Node {
        <<abstract>>
        +str uuid
        +str name
        +str group_id
        +list~str~ labels
        +datetime created_at
        +save(driver)*
        +delete(driver)
        +delete_by_group_id(driver, group_id)$
        +delete_by_uuids(driver, uuids)$
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
    }

    class EpisodicNode {
        +EpisodeType source
        +str source_description
        +str content
        +datetime valid_at
        +list~str~ entity_edges
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
        +get_by_entity_node_uuid(driver, uuid)$
    }

    class EntityNode {
        +list~float~ name_embedding
        +str summary
        +dict attributes
        +generate_name_embedding(embedder)
        +load_name_embedding(driver)
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
    }

    class CommunityNode {
        +list~float~ name_embedding
        +str summary
        +generate_name_embedding(embedder)
        +load_name_embedding(driver)
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
    }

    Node <|-- EpisodicNode
    Node <|-- EntityNode
    Node <|-- CommunityNode
```

**Node Type Descriptions:**

- **Node**: Abstract base class providing common UUID, naming, grouping, and CRUD operations
- **EpisodicNode**: Represents a temporal episode (message, text, JSON) with raw content and temporal metadata
- **EntityNode**: Represents extracted entities with embeddings, summaries, and custom attributes
- **CommunityNode**: Represents clustered groups of related entities with summary descriptions

### Edge Types

```mermaid
classDiagram
    class Edge {
        <<abstract>>
        +str uuid
        +str group_id
        +str source_node_uuid
        +str target_node_uuid
        +datetime created_at
        +save(driver)*
        +delete(driver)
        +delete_by_uuids(driver, uuids)$
        +get_by_uuid(driver, uuid)$
    }

    class EpisodicEdge {
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
    }

    class EntityEdge {
        +str name
        +str fact
        +list~float~ fact_embedding
        +list~str~ episodes
        +datetime expired_at
        +datetime valid_at
        +datetime invalid_at
        +dict attributes
        +generate_embedding(embedder)
        +load_fact_embedding(driver)
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_between_nodes(driver, source, target)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
        +get_by_node_uuid(driver, node_uuid)$
    }

    class CommunityEdge {
        +save(driver)
        +get_by_uuid(driver, uuid)$
        +get_by_uuids(driver, uuids)$
        +get_by_group_ids(driver, group_ids)$
    }

    Edge <|-- EpisodicEdge
    Edge <|-- EntityEdge
    Edge <|-- CommunityEdge
```

**Edge Type Descriptions:**

- **Edge**: Abstract base class with source/target UUIDs and common operations
- **EpisodicEdge**: Links episodic nodes to mentioned entities (MENTIONS relationship)
- **EntityEdge**: Semantic relationships between entities with fact descriptions, embeddings, and temporal validity
- **CommunityEdge**: Links community nodes to their member entities (HAS_MEMBER relationship)

### Database Drivers

```mermaid
classDiagram
    class GraphDriver {
        <<abstract>>
        +GraphProvider provider
        +str fulltext_syntax
        +str _database
        +str default_group_id
        +SearchInterface search_interface
        +GraphOperationsInterface graph_operations_interface
        +execute_query(query, kwargs)*
        +session(database)*
        +close()*
        +delete_all_indexes()*
        +build_indices_and_constraints(delete_existing)*
        +with_database(database)
        +clone(database)
        +build_fulltext_query(query, group_ids)
    }

    class GraphDriverSession {
        <<abstract>>
        +GraphProvider provider
        +run(query, kwargs)*
        +close()*
        +execute_write(func, args, kwargs)*
    }

    class Neo4jDriver {
        +GraphProvider provider
        +Neo4jDriver _driver
        +str _database
        +str default_group_id
        +execute_query(query, kwargs)
        +session(database)
        +close()
        +delete_all_indexes()
        +build_indices_and_constraints(delete_existing)
        +clone(database)
    }

    class FalkorDriver {
        +GraphProvider provider
        +str fulltext_syntax
        +FalkorDB client
        +str graph_name
        +str default_group_id
        +execute_query(query, kwargs)
        +session(database)
        +close()
        +delete_all_indexes()
        +build_indices_and_constraints(delete_existing)
        +clone(database)
        +build_fulltext_query(query, group_ids)
    }

    class KuzuDriver {
        +GraphProvider provider
        +Database db
        +Connection conn
        +str _database
        +str default_group_id
        +execute_query(query, kwargs)
        +session(database)
        +close()
        +delete_all_indexes()
        +build_indices_and_constraints(delete_existing)
        +clone(database)
    }

    class NeptuneDriver {
        +GraphProvider provider
        +str fulltext_syntax
        +str neptune_host
        +str neptune_port
        +str aoss_endpoint
        +SearchInterface search_interface
        +str _database
        +str default_group_id
        +execute_query(query, kwargs)
        +session(database)
        +close()
        +delete_all_indexes()
        +build_indices_and_constraints(delete_existing)
        +save_to_aoss(index_name, items)
        +build_fulltext_query(query, group_ids, max_length)
    }

    GraphDriver <|-- Neo4jDriver
    GraphDriver <|-- FalkorDriver
    GraphDriver <|-- KuzuDriver
    GraphDriver <|-- NeptuneDriver
    GraphDriver o-- GraphDriverSession
```

**Database Driver Descriptions:**

- **GraphDriver**: Abstract interface defining common graph database operations
- **Neo4jDriver**: Production-ready driver for Neo4j graph database (primary support)
- **FalkorDriver**: Redis-based FalkorDB driver with custom fulltext query syntax
- **KuzuDriver**: Embedded KuzuDB driver with unique edge modeling approach
- **NeptuneDriver**: AWS Neptune driver with OpenSearch integration for fulltext search

### LLM Clients

```mermaid
classDiagram
    class LLMClient {
        <<abstract>>
        +LLMConfig config
        +str model
        +str small_model
        +float temperature
        +int max_tokens
        +bool cache_enabled
        +Cache cache_dir
        +Tracer tracer
        +set_tracer(tracer)
        +_clean_input(input)
        +_generate_response(messages, response_model, max_tokens, model_size)*
        +_generate_response_with_retry(messages, response_model, max_tokens, model_size)
        +generate_response(messages, response_model, max_tokens, model_size, group_id, prompt_name)
        +_get_provider_type()
        +_get_failed_generation_log(messages, output)
    }

    class BaseOpenAIClient {
        +OpenAI client
        +_generate_response(messages, response_model, max_tokens, model_size)
        +_parse_response(completion, response_model)
    }

    class OpenAIClient {
        +OpenAI client
    }

    class AzureOpenAILLMClient {
        +AzureOpenAI client
    }

    class OpenAIGenericClient {
        +OpenAI client
    }

    class AnthropicClient {
        +Anthropic client
        +_generate_response(messages, response_model, max_tokens, model_size)
    }

    class GeminiClient {
        +genai GenerativeModel model
        +genai GenerativeModel small_model
        +_generate_response(messages, response_model, max_tokens, model_size)
    }

    class GroqClient {
        +Groq client
        +_generate_response(messages, response_model, max_tokens, model_size)
    }

    LLMClient <|-- BaseOpenAIClient
    BaseOpenAIClient <|-- OpenAIClient
    BaseOpenAIClient <|-- AzureOpenAILLMClient
    BaseOpenAIClient <|-- OpenAIGenericClient
    LLMClient <|-- AnthropicClient
    LLMClient <|-- GeminiClient
    LLMClient <|-- GroqClient
```

**LLM Client Descriptions:**

- **LLMClient**: Abstract base with retry logic, caching, input cleaning, and tracing
- **BaseOpenAIClient**: Shared OpenAI SDK logic for response parsing
- **OpenAIClient**: Standard OpenAI API client
- **AzureOpenAILLMClient**: Azure-hosted OpenAI endpoints
- **OpenAIGenericClient**: Generic OpenAI-compatible APIs
- **AnthropicClient**: Claude models (Anthropic)
- **GeminiClient**: Google Gemini models
- **GroqClient**: Groq inference endpoints

### Embedders

```mermaid
classDiagram
    class EmbedderClient {
        <<abstract>>
        +create(input_data)*
        +create_batch(input_data_list)
    }

    class OpenAIEmbedder {
        +OpenAI client
        +str model
        +int embedding_dim
        +create(input_data)
        +create_batch(input_data_list)
    }

    class AzureOpenAIEmbedderClient {
        +AzureOpenAI client
        +str model
        +int embedding_dim
        +create(input_data)
        +create_batch(input_data_list)
    }

    class GeminiEmbedder {
        +str model
        +int embedding_dim
        +create(input_data)
        +create_batch(input_data_list)
    }

    class VoyageAIEmbedder {
        +voyageai.Client client
        +str model
        +int embedding_dim
        +create(input_data)
        +create_batch(input_data_list)
    }

    EmbedderClient <|-- OpenAIEmbedder
    EmbedderClient <|-- AzureOpenAIEmbedderClient
    EmbedderClient <|-- GeminiEmbedder
    EmbedderClient <|-- VoyageAIEmbedder
```

**Embedder Descriptions:**

- **EmbedderClient**: Abstract interface for generating vector embeddings
- **OpenAIEmbedder**: OpenAI text-embedding models (default: text-embedding-3-large)
- **AzureOpenAIEmbedderClient**: Azure-hosted OpenAI embeddings
- **GeminiEmbedder**: Google Gemini embedding models
- **VoyageAIEmbedder**: Voyage AI specialized embeddings

### Rerankers (Cross-Encoders)

```mermaid
classDiagram
    class CrossEncoderClient {
        <<abstract>>
        +rank(query, passages, top_k)*
    }

    class OpenAIRerankerClient {
        +LLMClient llm_client
        +rank(query, passages, top_k)
    }

    class GeminiRerankerClient {
        +LLMClient llm_client
        +rank(query, passages, top_k)
    }

    class BGERerankerClient {
        +CrossEncoder model
        +rank(query, passages, top_k)
    }

    CrossEncoderClient <|-- OpenAIRerankerClient
    CrossEncoderClient <|-- GeminiRerankerClient
    CrossEncoderClient <|-- BGERerankerClient
```

**Reranker Descriptions:**

- **CrossEncoderClient**: Abstract interface for reranking search results
- **OpenAIRerankerClient**: LLM-based reranking using OpenAI models
- **GeminiRerankerClient**: LLM-based reranking using Gemini models
- **BGERerankerClient**: BAAI/bge-reranker cross-encoder model

## Module Dependencies

This diagram shows the import relationships between major modules in the graphiti_core package:

```mermaid
graph TD
    subgraph "Entry Point"
        G[graphiti.py]
    end

    subgraph "Core Models"
        N[nodes.py]
        E[edges.py]
        GT[graphiti_types.py]
    end

    subgraph "Clients"
        D[driver/]
        L[llm_client/]
        EM[embedder/]
        CE[cross_encoder/]
    end

    subgraph "Search Subsystem"
        S[search/search.py]
        SC[search/search_config.py]
        SH[search/search_helpers.py]
        SU[search/search_utils.py]
        SF[search/search_filters.py]
    end

    subgraph "Prompts"
        P[prompts/]
    end

    subgraph "Utilities"
        U1[utils/bulk_utils.py]
        U2[utils/datetime_utils.py]
        U3[utils/maintenance/node_operations.py]
        U4[utils/maintenance/edge_operations.py]
        U5[utils/maintenance/dedup_helpers.py]
        U6[utils/maintenance/community_operations.py]
        U7[utils/maintenance/temporal_operations.py]
    end

    subgraph "Support"
        H[helpers.py]
        ER[errors.py]
        DEC[decorators.py]
        TEL[telemetry/]
        TR[tracer.py]
    end

    G --> N
    G --> E
    G --> GT
    G --> D
    G --> L
    G --> EM
    G --> CE
    G --> S
    G --> U1
    G --> U3
    G --> U4
    G --> U5
    G --> U6
    G --> U7
    G --> H
    G --> TEL
    G --> TR

    N --> D
    N --> EM
    N --> H
    N --> ER

    E --> D
    E --> EM
    E --> N
    E --> H
    E --> ER

    S --> SC
    S --> SH
    S --> SU
    S --> SF
    S --> D
    S --> EM
    S --> CE
    S --> N
    S --> E

    U3 --> N
    U3 --> L
    U3 --> P
    U3 --> H

    U4 --> E
    U4 --> L
    U4 --> P
    U4 --> H

    U5 --> N
    U5 --> E
    U5 --> L
    U5 --> H

    U6 --> N
    U6 --> E
    U6 --> L
    U6 --> EM
    U6 --> D

    U1 --> N
    U1 --> E
    U1 --> U3
    U1 --> U4
    U1 --> U5
    U1 --> EM

    L --> P
    L --> TR

    SC --> N
    SC --> E

    SU --> N
    SU --> E
    SU --> D
    SU --> EM
```

**Dependency Analysis:**

1. **graphiti.py**: Main orchestrator importing from all major subsystems
2. **nodes.py & edges.py**: Core data models with minimal dependencies (driver, embedder, helpers)
3. **Search subsystem**: Self-contained module for semantic and hybrid search
4. **Utils/maintenance**: Specialized operations for node/edge extraction, deduplication, and communities
5. **Clients**: Provider interfaces (driver, llm_client, embedder, cross_encoder) are independent
6. **Prompts**: Used by LLM client and maintenance operations for structured extraction
7. **Support modules**: Helpers, errors, decorators, telemetry, and tracing used throughout

## Data Flow

This diagram illustrates how data flows through the system during episode ingestion and search:

```mermaid
flowchart LR
    subgraph "Input"
        I1[Episode Content]
        I2[Search Query]
    end

    subgraph "Episode Ingestion Flow"
        E1[Create EpisodicNode]
        E2[Retrieve Context Episodes]
        E3[Extract Entities LLM]
        E4[Deduplicate Nodes LLM]
        E5[Generate Entity Embeddings]
        E6[Extract Relations LLM]
        E7[Deduplicate Edges LLM]
        E8[Generate Edge Embeddings]
        E9[Build Episodic Edges]
        E10[Save to Graph DB]
        E11[Update Communities Optional]
    end

    subgraph "Search Flow"
        S1[Generate Query Embedding]
        S2[Vector Similarity Search]
        S3[Fulltext Search]
        S4[BFS Traversal]
        S5[Merge Results RRF/MMR]
        S6[Rerank with Cross-Encoder]
        S7[Return Results]
    end

    subgraph "Storage"
        DB[(Graph Database)]
    end

    I1 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    E5 --> E6
    E6 --> E7
    E7 --> E8
    E8 --> E9
    E9 --> E10
    E10 --> E11

    E2 -.-> DB
    E4 -.-> DB
    E7 -.-> DB
    E10 --> DB
    E11 --> DB

    I2 --> S1
    S1 --> S2
    S1 --> S3
    S1 --> S4
    S2 --> S5
    S3 --> S5
    S4 --> S5
    S5 --> S6
    S6 --> S7

    S2 -.-> DB
    S3 -.-> DB
    S4 -.-> DB

    style E3 fill:#e1f5ff
    style E6 fill:#e1f5ff
    style E4 fill:#e1f5ff
    style E7 fill:#e1f5ff
    style E5 fill:#fff4e1
    style E8 fill:#fff4e1
    style S1 fill:#fff4e1
    style S6 fill:#ffe1f5
```

**Flow Legend:**
- Blue boxes: LLM operations (entity/relation extraction, deduplication)
- Yellow boxes: Embedding operations (vectorization)
- Pink boxes: Reranking operations (cross-encoder scoring)
- Solid arrows: Primary data flow
- Dashed arrows: Database queries

**Episode Ingestion Flow:**

1. **Episode Creation**: Raw content wrapped in EpisodicNode with metadata
2. **Context Retrieval**: Fetch recent episodes for context-aware extraction
3. **Entity Extraction**: LLM identifies entities with types and attributes
4. **Node Deduplication**: LLM merges similar/duplicate entities against existing graph
5. **Entity Embedding**: Generate semantic vectors for entity names
6. **Relation Extraction**: LLM identifies relationships between entities
7. **Edge Deduplication**: LLM merges similar/duplicate relations
8. **Edge Embedding**: Generate semantic vectors for relation facts
9. **Episodic Edge Creation**: Link episode to mentioned entities
10. **Graph Persistence**: Save all nodes and edges to database
11. **Community Update**: Optionally update community summaries

**Search Flow:**

1. **Query Embedding**: Vectorize search query
2. **Vector Search**: Cosine similarity against entity/edge embeddings
3. **Fulltext Search**: Text-based search using database fulltext indexes
4. **BFS Traversal**: Graph traversal from origin nodes
5. **Result Merging**: Combine results using RRF (Reciprocal Rank Fusion) or MMR (Maximal Marginal Relevance)
6. **Reranking**: Cross-encoder scores query-passage pairs for final ranking
7. **Return Results**: SearchResults with nodes and edges

## Key Design Patterns

### 1. Abstract Factory Pattern
The system uses abstract base classes (GraphDriver, LLMClient, EmbedderClient, CrossEncoderClient) with concrete implementations for each provider, allowing runtime provider selection.

### 2. Strategy Pattern
Search configurations (SearchConfig) encapsulate different search strategies (vector, fulltext, BFS) and reranking methods (RRF, MMR, cross-encoder) that can be swapped.

### 3. Repository Pattern
Node and Edge classes have static methods for database operations (get_by_uuid, get_by_group_ids, delete_by_uuids) abstracting persistence logic.

### 4. Decorator Pattern
The `@handle_multiple_group_ids` decorator in graphiti.py handles multi-partition queries by wrapping methods.

### 5. Template Method Pattern
LLMClient defines the retry and caching logic in base class, while subclasses implement `_generate_response()`.

### 6. Dependency Injection
The Graphiti class accepts driver, llm_client, embedder, and cross_encoder as constructor parameters, bundled in GraphitiClients for passing to operations.

## Scalability Considerations

1. **Batch Operations**: `add_episode_bulk()` processes multiple episodes efficiently
2. **Async/Await**: All I/O operations use async for concurrency
3. **Semaphore Limiting**: `semaphore_gather()` controls concurrent operation count
4. **Embedding Batching**: `create_batch()` methods vectorize multiple items in single API call
5. **Graph Partitioning**: `group_id` enables logical graph partitioning
6. **Caching**: LLM responses can be disk-cached to reduce API costs
7. **Provider Choice**: Embedded KuzuDB for small deployments, Neo4j/Neptune for production scale
