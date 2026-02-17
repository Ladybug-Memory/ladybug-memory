"""Dynamic schema discovery using icebug parallel leiden clustering.

This module discovers entity schemas dynamically by:
1. Extracting entities without predefined types (or with broad types)
2. Building an entity similarity graph based on embeddings/context
3. Running parallel leiden clustering to group similar entities
4. Analyzing clusters to discover high-confidence schema types
5. Creating dynamic tables in ladybug for discovered schemas
"""

from dataclasses import dataclass
import uuid
import numpy as np
import pyarrow as pa

# Note: icebug provides networkit-compatible API
import networkit as nk
from fastembed import TextEmbedding

from memory.entities import Entity

# Global registry to keep Arrow arrays alive
# Key: graph id, Value: dict with Arrow arrays
# Using weakref allows cleanup when graph is garbage collected
_arrow_registry = {}


def _generate_type_name_with_llm(
    entities: list[str], base_type: str, model: str = "gpt-4o-mini"
) -> str:
    """Use LLM to generate a descriptive type name for a cluster of entities."""
    try:
        import os
        from litellm import completion

        entity_list = ", ".join(entities[:15])
        prompt = f"""Given these entities of type "{base_type}": {entity_list}

Generate a specific, descriptive type name (2-3 words max, PascalCase) that best describes this group.
Examples:
- "Tim Cook, Sundar Pichai, Elon Musk" -> "Tech Executive"
- "Apple Inc., Microsoft, Google" -> "Tech Company"
- "Stanford, MIT, Harvard" -> "University"
- "London, Tokyo, Boston" -> "City"
- "Pfizer, Moderna, Novartis" -> "Pharmaceutical Company"

Respond with ONLY the type name, no explanation."""

        # Handle Ollama API base URL (strip trailing slash)
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0.3,
        }

        if model.startswith("ollama/"):
            ollama_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
            kwargs["api_base"] = ollama_base.rstrip("/")

        response = completion(**kwargs)

        type_name = response.choices[0].message.content.strip()
        # Clean up - remove quotes, extra whitespace
        type_name = type_name.strip("\"'").strip()
        # Ensure PascalCase (capitalize each word)
        type_name = "".join(
            word.capitalize() for word in type_name.replace("_", " ").split()
        )
        return type_name

    except Exception:
        # Fallback to base type if LLM fails
        return base_type.capitalize()


@dataclass
class DiscoveredSchema:
    """A discovered schema type from clustering."""

    type_name: str
    confidence: float
    sample_entities: list[str]
    cluster_id: int
    size: int


@dataclass
class EntitySimilarityEdge:
    """Edge in entity similarity graph."""

    source_idx: int
    target_idx: int
    similarity: float


class DynamicSchemaDiscovery:
    """Discovers entity schemas dynamically using clustering."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
        min_confidence: float = 0.7,
        llm_model: str | None = "gpt-4o-mini",
    ):
        """Initialize dynamic schema discovery.

        Args:
            similarity_threshold: Minimum cosine similarity to create edge
            min_cluster_size: Minimum entities in a cluster to be a schema type
            min_confidence: Minimum confidence for discovered schema
            llm_model: LiteLLM model name for generating descriptive type names.
                       Set to None to disable LLM and use base entity types.
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.min_confidence = min_confidence
        self.llm_model = llm_model
        self._embedding_model: TextEmbedding | None = None

    def _init_embedding_model(self) -> None:
        """Initialize embedding model for entity similarity."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def _get_embeddings(self, entities: list[Entity]) -> np.ndarray:
        """Get embeddings for entities."""
        self._init_embedding_model()
        texts = [e.text for e in entities]
        embeddings = list(self._embedding_model.embed(texts))
        return np.array(embeddings)

    def _build_similarity_graph(
        self, entities: list[Entity]
    ) -> tuple[nk.Graph, list[Entity], np.ndarray]:
        """Build entity similarity graph using embeddings.

        Returns:
            Tuple of (networkit graph, filtered entity list, normalized embeddings)
        """
        if len(entities) < 2:
            # Create empty graph with single node if only one entity
            graph = nk.Graph(1)
            return graph, entities, np.array([])

        # Get embeddings
        embeddings = self._get_embeddings(entities)

        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

        # Build edge list based on similarity threshold
        sources = []
        targets = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                similarity = similarity_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    sources.append(i)
                    targets.append(j)

        # Create DataFrame and convert to Arrow
        if len(sources) == 0:
            # No edges above threshold, create graph with just nodes
            graph = nk.Graph(len(entities))
            return graph, entities, embeddings_norm

        # Remap node IDs to consecutive integers (CSR requires 0 to n-1)
        # IMPORTANT: Map ALL entity indices (0 to len(entities)-1) to ensure partition aligns
        unique_nodes = list(range(len(entities)))
        node_map = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}

        sources_list = [node_map[s] for s in sources]
        targets_list = [node_map[t] for t in targets]

        # Undirected graph - add reverse edges
        all_sources = sources_list + targets_list
        all_targets = targets_list + sources_list

        n_nodes = len(unique_nodes)

        # Build CSR indptr
        indptr = [0] * (n_nodes + 1)
        edge_counts = [0] * n_nodes
        for src in all_sources:
            edge_counts[src] += 1

        for i in range(n_nodes):
            indptr[i + 1] = indptr[i] + edge_counts[i]

        # Sort edges by source
        edges_with_idx = [
            (src, tgt, i) for i, (src, tgt) in enumerate(zip(all_sources, all_targets))
        ]
        edges_with_idx.sort(key=lambda x: x[0])
        sorted_targets = [x[1] for x in edges_with_idx]

        # Create Arrow arrays
        indices_arrow = pa.array(sorted_targets, type=pa.uint64())
        indptr_arrow = pa.array(indptr, type=pa.uint64())

        # Create graph
        graph = nk.Graph.fromCSR(n_nodes, False, indices_arrow, indptr_arrow)

        # CRITICAL: Keep Arrow arrays alive by storing them in a registry keyed by graph id
        # The C++ CSR graph holds raw pointers to the Arrow array data
        graph_id = id(graph)
        _arrow_registry[graph_id] = {
            "indices": indices_arrow,
            "indptr": indptr_arrow,
        }

        return graph, entities, embeddings_norm

    def _run_leiden_clustering(self, graph: nk.Graph):
        """Run parallel leiden clustering on entity graph.

        Args:
            graph: Networkit graph

        Returns:
            Partition object with cluster assignments
        """
        # Create ParallelLeidenView instance
        # Use minimal iterations for speed
        leiden = nk.community.ParallelLeidenView(
            graph, iterations=1, randomize=False, gamma=0.5
        )

        # Run clustering
        leiden.run()

        # Get partition
        partition = leiden.getPartition()

        return partition

    def _analyze_clusters(
        self,
        entities: list[Entity],
        partition,
        all_embeddings: np.ndarray,
    ) -> list[DiscoveredSchema]:
        """Analyze clusters to discover schema types.

        Args:
            entities: List of entities
            partition: Cluster partition from leiden
            all_embeddings: Pre-computed normalized embeddings

        Returns:
            List of discovered schemas
        """
        discovered_schemas = []

        # Group entities by cluster (store indices for embedding lookup)
        cluster_entities: dict[int, list[tuple[int, Entity]]] = {}
        for node_id in range(len(entities)):
            cluster_id = partition[node_id]
            if cluster_id not in cluster_entities:
                cluster_entities[cluster_id] = []
            cluster_entities[cluster_id].append((node_id, entities[node_id]))

        # Analyze each cluster
        for cluster_id, cluster_items in cluster_entities.items():
            if len(cluster_items) < self.min_cluster_size:
                continue

            # Extract just entities for type inference
            cluster_ents = [e for _, e in cluster_items]

            # Calculate confidence based on cluster cohesion
            if len(cluster_items) > 1 and len(all_embeddings) > 0:
                # Get embeddings for this cluster using original indices
                indices = [idx for idx, _ in cluster_items]
                cluster_embs = all_embeddings[indices]
                similarity_matrix = np.dot(cluster_embs, cluster_embs.T)
                # Get upper triangle (excluding diagonal)
                similarities = similarity_matrix[
                    np.triu_indices_from(similarity_matrix, k=1)
                ]
                confidence = float(np.mean(similarities))
            else:
                confidence = 1.0

            if confidence < self.min_confidence:
                continue

            # Infer type name from GLiNER2 entity types
            sample_texts = [e.text for e in cluster_ents[:5]]
            type_name = self._infer_type_name(cluster_ents)

            discovered_schemas.append(
                DiscoveredSchema(
                    type_name=type_name,
                    confidence=confidence,
                    sample_entities=sample_texts,
                    cluster_id=cluster_id,
                    size=len(cluster_ents),
                )
            )

        # Sort by confidence
        discovered_schemas.sort(key=lambda x: x.confidence, reverse=True)

        return discovered_schemas

    def _infer_type_name(self, entities: list[Entity]) -> str:
        """Infer a descriptive type name from cluster entities.

        Uses LLM to generate a specific type name based on the entities.
        Falls back to the GLiNER2 entity_type if LLM is disabled or fails.
        """
        from collections import Counter

        # Get base type from GLiNER2
        type_counts = Counter([e.entity_type for e in entities])
        base_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"

        # If LLM is enabled, generate a descriptive name
        if self.llm_model:
            entity_texts = [e.text for e in entities[:20]]
            return _generate_type_name_with_llm(entity_texts, base_type, self.llm_model)

        # Fallback: use capitalized base type
        return base_type.capitalize()

    def discover_schema(
        self, entities: list[Entity]
    ) -> tuple[list[DiscoveredSchema], dict[str, str]]:
        """Discover schema types from entities.

        Args:
            entities: List of extracted entities

        Returns:
            Tuple of (discovered schemas, entity to type mapping)
        """
        if len(entities) == 0:
            return [], {}

        # Build similarity graph
        graph, filtered_entities, embeddings_norm = self._build_similarity_graph(
            entities
        )

        # Run leiden clustering
        partition = self._run_leiden_clustering(graph)

        # Analyze clusters (pass embeddings to avoid recomputing)
        schemas = self._analyze_clusters(filtered_entities, partition, embeddings_norm)

        # Create entity to type mapping
        entity_to_type: dict[str, str] = {}
        cluster_to_type: dict[int, str] = {s.cluster_id: s.type_name for s in schemas}

        for node_id in range(len(filtered_entities)):
            cluster_id = partition[node_id]
            if cluster_id in cluster_to_type:
                entity = filtered_entities[node_id]
                entity_to_type[entity.id] = cluster_to_type[cluster_id]

        return schemas, entity_to_type
