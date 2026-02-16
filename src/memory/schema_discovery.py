"""Dynamic schema discovery using icebug parallel leiden clustering.

This module discovers entity schemas dynamically by:
1. Extracting entities without predefined types (or with broad types)
2. Building an entity similarity graph based on embeddings/context
3. Running parallel leiden clustering to group similar entities
4. Analyzing clusters to discover high-confidence schema types
5. Creating dynamic tables in ladybug for discovered schemas
"""

from dataclasses import dataclass
from typing import Any
import uuid
import numpy as np
import pyarrow as pa
import polars as pl

# Note: icebug provides networkit-compatible API
import networkit as nk
from fastembed import TextEmbedding

from memory.entities import Entity


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
    ):
        """Initialize dynamic schema discovery.

        Args:
            similarity_threshold: Minimum cosine similarity to create edge
            min_cluster_size: Minimum entities in a cluster to be a schema type
            min_confidence: Minimum confidence for discovered schema
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.min_confidence = min_confidence
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
    ) -> tuple[nk.Graph, list[Entity]]:
        """Build entity similarity graph using embeddings.

        Returns:
            Tuple of (networkit graph, filtered entity list)
        """
        if len(entities) < 2:
            # Create empty graph with single node if only one entity
            graph = nk.Graph(1)
            return graph, entities

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
            return graph, entities

        df = pl.DataFrame(
            {
                "source": pl.Series(sources, dtype=pl.UInt64),
                "target": pl.Series(targets, dtype=pl.UInt64),
            }
        )

        # Build CSR format
        sources_list = df["source"].to_list()
        targets_list = df["target"].to_list()

        # Undirected graph - add reverse edges
        all_sources = sources_list + targets_list
        all_targets = targets_list + sources_list

        n_nodes = len(entities)

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

        return graph, entities

    def _run_leiden_clustering(self, graph: nk.Graph):
        """Run parallel leiden clustering on entity graph.

        Args:
            graph: Networkit graph

        Returns:
            Partition object with cluster assignments
        """
        # Create ParallelLeidenView instance
        leiden = nk.community.ParallelLeidenView(
            graph, iterations=3, randomize=True, gamma=1.0
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
    ) -> list[DiscoveredSchema]:
        """Analyze clusters to discover schema types.

        Args:
            entities: List of entities
            partition: Cluster partition from leiden

        Returns:
            List of discovered schemas
        """
        discovered_schemas = []

        # Group entities by cluster
        cluster_entities: dict[int, list[Entity]] = {}
        for node_id in range(len(entities)):
            cluster_id = partition[node_id]
            if cluster_id not in cluster_entities:
                cluster_entities[cluster_id] = []
            cluster_entities[cluster_id].append(entities[node_id])

        # Analyze each cluster
        for cluster_id, cluster_ents in cluster_entities.items():
            if len(cluster_ents) < self.min_cluster_size:
                continue

            # Calculate confidence based on cluster cohesion
            # (average similarity within cluster)
            if len(cluster_ents) > 1:
                embeddings = self._get_embeddings(cluster_ents)
                embeddings_norm = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
                # Get upper triangle (excluding diagonal)
                similarities = similarity_matrix[
                    np.triu_indices_from(similarity_matrix, k=1)
                ]
                confidence = float(np.mean(similarities))
            else:
                confidence = 1.0

            if confidence < self.min_confidence:
                continue

            # Infer type name from most common words in entity texts
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
        """Infer a type name from cluster entities.

        Uses heuristics based on entity text patterns.
        """
        # Get all entity texts
        texts = [e.text.lower() for e in entities]

        # Check for common patterns
        # Person names (typically 2-3 capitalized words)
        person_pattern = sum(
            1
            for t in texts
            if len(t.split()) in [1, 2] and all(c.isalpha() or c.isspace() for c in t)
        )

        # Locations (contain location indicators)
        location_indicators = [
            "city",
            "state",
            "country",
            "street",
            "avenue",
            "road",
            "blvd",
        ]
        location_pattern = sum(
            1 for t in texts for ind in location_indicators if ind in t
        )

        # Organizations (common suffixes)
        org_suffixes = ["inc", "corp", "llc", "ltd", "company", "co.", "gmbh"]
        org_pattern = sum(1 for t in texts for suffix in org_suffixes if suffix in t)

        # Dates (numbers with separators)
        date_pattern = sum(
            1
            for t in texts
            if any(c.isdigit() for c in t) and any(c in t for c in ["/", "-", ".", ","])
        )

        # Products (often contain version numbers or proper nouns)
        product_pattern = sum(
            1 for t in texts if any(c.isdigit() for c in t) and len(t.split()) <= 3
        )

        # Find best match
        patterns = {
            "person": person_pattern,
            "location": location_pattern,
            "organization": org_pattern,
            "date": date_pattern,
            "product": product_pattern,
        }

        best_type = max(patterns.items(), key=lambda x: x[1])[0]

        # If no strong pattern, use generic name based on most common words
        if patterns[best_type] == 0:
            # Extract words and find most common
            words = []
            for t in texts:
                words.extend(t.split())

            if words:
                from collections import Counter

                most_common = Counter(words).most_common(1)[0][0]
                return f"{most_common}_entity"
            else:
                return f"entity_type_{uuid.uuid4().hex[:8]}"

        return best_type

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
        graph, filtered_entities = self._build_similarity_graph(entities)

        # Run leiden clustering
        partition = self._run_leiden_clustering(graph)

        # Analyze clusters
        schemas = self._analyze_clusters(filtered_entities, partition)

        # Create entity to type mapping
        entity_to_type: dict[str, str] = {}
        cluster_to_type: dict[int, str] = {s.cluster_id: s.type_name for s in schemas}

        for node_id in range(len(filtered_entities)):
            cluster_id = partition[node_id]
            if cluster_id in cluster_to_type:
                entity = filtered_entities[node_id]
                entity_to_type[entity.id] = cluster_to_type[cluster_id]

        return schemas, entity_to_type
