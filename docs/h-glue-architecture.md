# Hierarchical Graph-based Logical Unit Extraction (H-GLUE) Architecture

An alternative to Fixed Entity Architecture (FEA) that uses LLM reasoning to create logical hierarchical structures and extracts entities from natural semantic units rather than fixed-size chunks.

## Overview

This document proposes the **Hierarchical Graph-based Logical Unit Extraction (H-GLUE)** architecture—a cost-efficient RAG system for knowledge graphs. Unlike FEA's fixed-size chunking, H-GLUE analyzes document structure to identify natural logical boundaries and uses LLM reasoning hierarchically for document segmentation, optimizing cost by using cheaper models (GLiNER2) for routine extraction and reserving LLMs for complex disambiguation tasks.

## Key Differences from FEA

| Aspect | FEA | H-GLUE |
|--------|-----|------|
| **Chunking** | Fixed 1000-character windows | Dynamic logical units based on structure |
| **Hierarchy** | Layered fixed chunks | Semantic hierarchical decomposition |
| **Structure Detection** | N/A (always chunks) | Detects pre-segmented docs (Markdown, HTML) to skip LLM |
| **Extraction** | Same process for all chunks | Adaptive: GLiNER2 for standard entities, LLM for complex cases |
| **Disambiguation** | Graph-based co-occurrence | Cosine similarity of entity embeddings |
| **Cost Model** | Uniform processing | Tiered: GLiNER2 (cheap) + LLM (expensive, sparing) |

## Architecture

### 1. Document Analysis & Hierarchical Segmentation

**Process:**
```
Document → Has Structure? ──Yes──┐
     │                           │
    No                           │
     ↓                           ↓
LLM Structure Analysis    Parse Existing Structure
     │                           │
     └──────────┬────────────────┘
                ↓
      Hierarchical Tree → Logical Units
```

The LLM analyzes the document to identify:
- **Sections** (major topical divisions)
- **Subsections** (nested logical groupings)
- **Paragraphs** (atomic semantic units)
- **Lists/Tables** (structured data)

**Example Output:**
```json
{
  "type": "document",
  "title": "Annual Report 2024",
  "children": [
    {
      "type": "section",
      "title": "Executive Summary",
      "children": [
        {"type": "paragraph", "text": "...", "semantic_focus": "overview"},
        {"type": "paragraph", "text": "...", "semantic_focus": "key_findings"}
      ]
    },
    {
      "type": "section",
      "title": "Financial Results",
      "children": [
        {
          "type": "subsection",
          "title": "Q1 Performance",
          "children": [...]
        }
      ]
    }
  ]
}
```

**Cost Optimization & Structure Detection:**

Before invoking an LLM for structure analysis, the system first checks if the document already has clear logical segmentation:

1. **Pre-segmented Documents (No LLM needed):**
   - Markdown with clear headers (`#`, `##`, `###`)
   - HTML with semantic tags (`<section>`, `<article>`, `<h1>`-`<h6>`)
   - JSON/XML with nested structures
   - LaTeX with `\section`, `\subsection` commands
   - Word documents with heading styles

   For these formats, extract the hierarchy directly using pattern matching or parsing libraries.

2. **Unstructured Documents (LLM required):**
   - Plain text without clear delimiters
   - PDFs without semantic markup
   - Scanned documents (OCR output)
   - Mixed format documents

3. **Additional Optimizations:**
   - Use a lightweight LLM (e.g., GPT-3.5-turbo, Llama 3.1 8B) only when needed
   - Cache structure trees for repeated queries
   - Reuse structure from previous processing runs if document unchanged

### 2. Entity Extraction Pipeline

**Two-Tier System:**

```
Logical Unit → GLiNER2 (Primary) → Confidence Check → [Low Confidence] → LLM (Fallback)
                                    ↓
                              [High Confidence] → Entity Output
```

#### Tier 1: GLiNER2 Entity Extraction

**When to use:**
- Standard entity types (person, organization, location, date, product)
- Clear, unambiguous text
- High-confidence predictions (>0.85)

**Process:**
```python
entities = gliner_model.predict(
    text=logical_unit.text,
    labels=["person", "organization", "location", "product", "date", "event"],
    threshold=0.85
)
```

**Advantages:**
- 10-100x cheaper than LLM API calls
- Fast inference (can run locally)
- No rate limiting concerns

#### Tier 2: LLM Extraction (Conditional)

**Trigger conditions:**
1. GLiNER2 confidence < 0.85 for critical entity types
2. Complex entity relationships requiring reasoning
3. Domain-specific entities not in GLiNER2 training
4. Ambiguous references (pronouns, abbreviations)
5. Implicit entities requiring context understanding

**Prompt Template:**
```
Given the following context from "{document_title}" > "{section_title}":

Context: {logical_unit_text}

Extract entities with their types, confidence, and any relationships mentioned.
Focus on entities that are ambiguous or require domain knowledge.

Output format: JSON with entity, type, confidence, rationale
```

### 3. Entity Disambiguation via Cosine Similarity

**Problem:** Same entity mentioned multiple times with variations:
- "Apple Inc." vs "Apple" vs "the Cupertino company"
- "John Smith" vs "Smith" vs "he"

**Solution: Embedding-based Clustering**

```python
# Generate embeddings for each entity mention
entity_embeddings = embedding_model.encode(entity_mentions)

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(entity_embeddings)

# Cluster entities with similarity > threshold (typically 0.85-0.90)
clusters = hierarchical_clustering(similarity_matrix, threshold=0.85)

# Merge clusters into canonical entities
for cluster in clusters:
    canonical_entity = select_canonical_name(cluster)
    merge_entities(cluster, canonical_entity)
```

**Embedding Strategy:**
- Use sentence transformers (e.g., `all-MiniLM-L6-v2` for cost, `BAAI/bge-large-en` for accuracy)
- Include context window around entity mention for better disambiguation
- Store entity embeddings in vector database for efficient similarity search

**Context-Aware Similarity:**
```python
def get_entity_context_embedding(entity, window_size=50):
    """Get embedding of entity with surrounding context."""
    start = max(0, entity.start - window_size)
    end = min(len(text), entity.end + window_size)
    context = text[start:end]
    return embedding_model.encode(context)
```

### 4. Knowledge Graph Construction

**Graph Schema:**

```
Nodes:
- Entity (id, canonical_name, type, mentions[], embedding)
- LogicalUnit (id, type, text, parent_id, document_id)
- Document (id, title, metadata)

Edges:
- MENTIONED_IN (Entity → LogicalUnit, frequency, confidence)
- CONTAINS (LogicalUnit → LogicalUnit, hierarchical)
- SIMILAR_TO (Entity → Entity, cosine_similarity_score)
- COREFERENCE (Entity → Entity, from disambiguation)
```

**Relationship Extraction:**
- Primary: GLiNER2 with relation labels
- Fallback to LLM for complex relationships only
- Store relation confidence scores for filtering

## Cost Analysis

### Comparison with FEA

Assuming 1000 documents, 5000 characters each:

| Approach | GLiNER2 Calls | LLM Calls | Est. Cost |
|----------|--------------|-----------|-----------|
| **FEA** | 0 | 5000 (one per chunk) | $50-250 |
| **H-GLUE** | 3000 (logical units) | 200 (complex cases only) | $10-40 |

**Savings:** 60-80% reduction in LLM costs

### Breakdown

**H-GLUE Costs (Worst Case - All Unstructured):**
1. **Structure Analysis:** 1000 LLM calls (one per document) = ~$2-10
2. **GLiNER2 Extraction:** 3000 calls (local/free or $0.001 each) = ~$0-3
3. **LLM Fallback:** 200 calls for complex entities = ~$5-20
4. **Embedding Generation:** 5000 entity mentions = ~$1-5

**H-GLUE Costs (Best Case - Pre-segmented Documents):**
1. **Structure Analysis:** 0 LLM calls (use existing Markdown/HTML structure) = $0
2. **GLiNER2 Extraction:** 3000 calls = ~$0-3
3. **LLM Fallback:** 200 calls = ~$5-20
4. **Embedding Generation:** 5000 entity mentions = ~$1-5

**Best Case Savings:** Up to 90% reduction vs FEA when processing well-structured documents (Markdown, HTML, etc.)

## Implementation Phases

### Phase 1: Hierarchical Segmentation
- Implement document structure analyzer
- Support common formats (Markdown, HTML, plain text)
- Cache structure trees

### Phase 2: Two-Tier Extraction
- Integrate GLiNER2
- Build confidence-based routing to LLM
- Implement extraction result merging

### Phase 3: Disambiguation
- Set up embedding pipeline
- Implement cosine similarity clustering
- Build canonical entity resolution

### Phase 4: Graph Storage
- Choose graph database (Neo4j, Amazon Neptune, or RDF store)
- Implement schema
- Add querying interface

## Advantages Over FEA

1. **Semantic Coherence:** Logical units preserve context better than arbitrary chunks
2. **Cost Efficiency:** GLiNER2 handles 80-90% of extraction at fraction of LLM cost
3. **Scalability:** Local models reduce API dependency
4. **Accuracy:** Hierarchical context improves entity resolution
5. **Flexibility:** Easy to adapt to new domains by fine-tuning GLiNER2

## Trade-offs

1. **Complexity:** More moving parts than FEA
2. **Latency:** Structure analysis adds initial processing time
3. **Tuning Required:** Thresholds for GLiNER2 confidence and similarity need calibration per domain
4. **Storage:** Embeddings require additional storage (~1KB per entity)

## Future Enhancements

1. **Active Learning:** Use LLM extractions to improve GLiNER2 fine-tuning
2. **Incremental Processing:** Only re-analyze changed sections
3. **Multi-modal:** Extend to images, tables, charts within documents
4. **Real-time:** Stream processing for live document feeds

## References

1. [GLiNER: Generalist Model for Named Entity Recognition](https://github.com/urchade/GLiNER)
2. FEA Architecture: [Three-Layer Fixed Entity Architecture for Efficient RAG on Graphs](https://medium.com/@irina.karkkanen/three-layer-fixed-entity-architecture-for-efficient-rag-on-graphs-787c70e3151a)
3. Sentence Transformers for Embeddings
4. Hierarchical Document Structure Analysis

## Appendix: Sample Code

### Structure Analyzer

```python
async def analyze_document_structure(document: str, model: str = "gpt-3.5-turbo") -> DocumentTree:
    """Use LLM to identify logical hierarchical structure."""
    prompt = f"""
    Analyze the following document and return a JSON tree structure of its logical components.
    Identify sections, subsections, paragraphs, lists, and tables.

    Document:
    {document[:4000]}  # Truncate for token limits

    Return format:
    {{
      "type": "document",
      "title": "...",
      "children": [
        {{"type": "section", "title": "...", "start": 0, "end": 500, "children": [...]}}
      ]
    }}
    """

    response = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return DocumentTree.parse(response.choices[0].message.content)
```

### Adaptive Extractor

```python
class AdaptiveEntityExtractor:
    def __init__(self, gliner_model, llm_client):
        self.gliner = gliner_model
        self.llm = llm_client
        self.confidence_threshold = 0.85

    async def extract(self, text: str, context: Dict) -> List[Entity]:
        # Try GLiNER2 first
        gliner_entities = self.gliner.predict(text, labels=ENTITY_TYPES)

        entities = []
        low_confidence_spans = []

        for ent in gliner_entities:
            if ent.confidence >= self.confidence_threshold:
                entities.append(Entity.from_gliner(ent, context))
            else:
                low_confidence_spans.append((ent.start, ent.end))

        # Use LLM for low-confidence regions
        if low_confidence_spans:
            llm_entities = await self._llm_extract(text, low_confidence_spans, context)
            entities.extend(llm_entities)

        return entities

    async def _llm_extract(self, text: str, spans: List[Tuple], context: Dict) -> List[Entity]:
        # Extract text around low-confidence spans
        regions = [text[max(0, s-50):min(len(text), e+50)] for s, e in spans]

        prompt = f"""
        Extract entities from these text regions:
        {json.dumps(regions)}

        Context: {context.get('section_title', 'Unknown')}
        """

        response = await self.llm.chat.completions.create(...)
        return parse_llm_entities(response)
```

### Cosine Similarity Disambiguation

```python
class EntityDisambiguator:
    def __init__(self, embedding_model, similarity_threshold=0.85):
        self.embedder = embedding_model
        self.threshold = similarity_threshold

    def disambiguate(self, entities: List[Entity]) -> List[EntityCluster]:
        # Generate embeddings with context
        embeddings = []
        for ent in entities:
            context = ent.get_surrounding_text(window=50)
            emb = self.embedder.encode(f"{ent.text} | Context: {context}")
            embeddings.append(emb)

        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Hierarchical clustering
        clusters = self._cluster_entities(entities, sim_matrix)

        # Merge each cluster into canonical entity
        return [self._create_canonical_entity(c) for c in clusters]
```

---

*Last updated: 2026*
