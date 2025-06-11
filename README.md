# Sentence Transformer Requirements Analysis

A Python-based tool for analyzing and mapping requirements using hybrid similarity computation combining TF-IDF and Sentence Transformers with Neo4j integration.

## Overview

This project implements a hybrid approach to requirements traceability using both TF-IDF and Sentence Transformers. It supports multiple transformer models optimized for sentence embeddings and provides detailed analysis with configurable parameters. Results are stored in a Neo4j graph database for advanced querying and analysis.

## Key Components

### Research Notebooks (`/notebooks`)

| Notebook | Purpose |
|----------|---------|
| `00_Multi_Stage_Error_Analysis.ipynb` | Multi-stage LLM error analysis calculating performance metrics across TF-IDF, Claude 2.1, and Claude 3.5 processing stages |
| `01_Claude_Enhanced_Requirements.ipynb` | Claude AI-based requirements analysis and enhancement workflows with automated requirement processing |
| `02_Sentence_Transformer_Matching.ipynb` | Sentence transformer model analysis for requirements similarity matching with threshold optimization |
| `03_Similarity_Score_Performance.ipynb` | Comprehensive similarity score performance analysis with precision-recall curves and ROC analysis |
| `04_Claude_Requirements_Discovery.ipynb` | Claude AI-powered requirements discovery workflows for automated requirement identification |
| `05_Advanced_Transformer_Analysis.ipynb` | Advanced sentence transformer analysis with multi-model comparison and performance optimization |
| `06_ML_Similarity_Analysis.ipynb` | Machine learning similarity analysis using Random Forest classifiers with feature importance analysis |
| `07_HuggingFace_API_Testing.ipynb` | HuggingFace Transformers API testing and model validation workflows |
| `08_Enhanced_Claude_Finder.ipynb` | Enhanced Claude-based requirement finding with advanced natural language processing |
| `09_Multi_Modal_Analysis_Pipeline.ipynb` | Multi-modal analysis pipeline combining different AI models for comprehensive requirements analysis |
| `10_Multi_Model_Transformer_Processing.ipynb` | Multi-model transformer processing with batch optimization and performance comparison |

### Code (`/src/praxis_sentence_transformer`)

- **Analyzers (`/analyzers`)** - Core requirements analysis engines including sentence transformer analyzers and comprehensive error analysis frameworks
- **Clients (`/clients`)** - External service integrations for Claude AI and other LLM providers with authentication and rate limiting
- **Neo4j Operations (`/neo4j_operations`)** - Graph database client, operations, and data models for storing and querying requirements relationships
- **Preprocessors (`/preprocessors`)** - Data preprocessing pipelines for cleaning and normalizing requirements text data
- **Loaders (`/loaders`)** - Document and requirements data loading utilities supporting XML, JSON, and other formats
- **Visualization (`/visualization`)** - Performance visualization tools for generating charts, graphs, and analysis reports
- **Utilities (`/utils`)** - Helper functions for data manipulation, mathematical operations, and system utilities
- **Logger (`/logger`)** - Centralized logging system with configurable levels and structured output formatting

## Supported Models

The tool currently supports several Sentence Transformer models:
- `all-mpnet-base-v2`: Microsoft MPNet-based model, optimized for semantic similarity
- `all-MiniLM-L6-v2`: Lightweight model based on MiniLM architecture
- `all-distilroberta-v1`: Distilled RoBERTa-based model
- `multi-qa-mpnet-base-dot-v1`: MPNet optimized for question-answering tasks
- `multi-qa-mpnet-base-cos-v1`: MPNet variant using cosine similarity

## Key Parameters

### Similarity Computation (α)

The hybrid similarity score between two requirements is computed using a weighted combination of TF-IDF and transformer-based similarities:

$similarity = \alpha \cdot sim_{TF-IDF}(req_1, req_2) + (1-\alpha) \cdot sim_{transformer}(req_1, req_2)$

Where:
- $\alpha \in [0,1]$ is the weighting parameter
- $sim_{TF-IDF}$ is the cosine similarity between TF-IDF vectors
- $sim_{transformer}$ is the cosine similarity between transformer embeddings

#### α Parameter Effects:
- α = 0: Uses only transformer-based similarity
- α = 1: Uses only TF-IDF similarity
- α = Optimal range, giving more weight to transformer embeddings while still considering TF-IDF likely ideal

### Threshold (τ)

The threshold τ determines whether two requirements are considered related:

$isRelated(req_1, req_2) = \begin{cases} 
True & \text{if } similarity(req_1, req_2) \geq \tau \\
False & \text{otherwise}
\end{cases}$

#### Threshold Effects:
- Higher τ: More selective matching, higher precision but lower recall
- Lower τ: More inclusive matching, higher recall but lower precision

#### Coverage and Precision Trade-off:
The minimum coverage threshold (default 0.9 or 90%) ensures sufficient requirement coverage:

$coverage = \frac{|correctly\_matched\_requirements|}{|total\_requirements|} \geq min\_coverage$

### Evaluation Metrics and Scoring

For a given (α, τ) pair:

1. **Precision**:
   $P = \frac{TP}{TP + FP}$
   - Measures accuracy of identified links

2. **Recall**:
   $R = \frac{TP}{TP + FN}$
   - Measures completeness of identified links

3. **F1 Score**:
   $F1 = 2 \cdot \frac{P \cdot R}{P + R}$
   - Harmonic mean of precision and recall

4. **False Negative Rate**:
   $FNR = \frac{FN}{TP + FN}$
   - Rate of missed valid links

Where:
- TP: True Positives (correctly identified links)
- FP: False Positives (incorrectly identified links)
- FN: False Negatives (missed links)
- TN: True Negatives (correctly identified non-links)

### Neo4j Querying Examples

#### Finding True Positives
```cypher
// Find requirements that match both predicted and ground truth
MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
MATCH (s)-[g:GROUND_TRUTH]->(t)
RETURN s.id as source, t.id as target, p.score as similarity_score
ORDER BY p.score DESC
```

#### Finding False Positives
```cypher
// Find predicted links that don't exist in ground truth
MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
WHERE NOT EXISTS((s)-[:GROUND_TRUTH]->(t))
RETURN s.id as source, t.id as target, p.score as similarity_score
ORDER BY p.score DESC
```

#### Finding False Negatives
```cypher
// Find ground truth links that weren't predicted
MATCH (s:Requirement)-[g:GROUND_TRUTH]->(t:Requirement)
WHERE NOT EXISTS((s)-[:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t))
RETURN s.id as source, t.id as target
```

#### Getting Counts for Evaluation Metrics
```cypher
// Get counts for TP, FP, FN, TN
MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
WITH collect(distinct {source: s.id, target: t.id}) as predicted
MATCH (s:Requirement)-[g:GROUND_TRUTH]->(t:Requirement)
WITH predicted, collect(distinct {source: s.id, target: t.id}) as ground_truth
WITH predicted, ground_truth,
     [x in predicted WHERE x in ground_truth] as true_positives,
     [x in predicted WHERE NOT x in ground_truth] as false_positives,
     [x in ground_truth WHERE NOT x in predicted] as false_negatives
RETURN 
    size(true_positives) as TP,
    size(false_positives) as FP,
    size(false_negatives) as FN,
    size([x in predicted WHERE NOT x in ground_truth]) as TN
```

#### Analyzing High Confidence False Positives
```cypher
// Find false positives with high similarity scores
MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
WHERE NOT EXISTS((s)-[:GROUND_TRUTH]->(t))
  AND p.score > 0.8
RETURN s.id as source, s.content as source_content,
       t.id as target, t.content as target_content,
       p.score as similarity
ORDER BY p.score DESC
LIMIT 10
```

#### Finding Requirements with Most False Positives
```cypher
// Identify requirements generating many false positives
MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
WHERE NOT EXISTS((s)-[:GROUND_TRUTH]->(t))
WITH s, count(t) as false_positive_count
RETURN s.id as source_id, 
       s.content as content,
       false_positive_count
ORDER BY false_positive_count DESC
LIMIT 10
```

### Example Analysis Workflow

1. Run initial analysis:
```python
analyzer = SentenceTransformerAnalyzer(
    model_name="sentence-transformers/all-mpnet-base-v2",
    alpha=0.3
)
analyzer.initialize()
results = analyzer.analyze_requirements(
    source_file="path/to/source.xml",
    target_file="path/to/target.xml",
    threshold=0.15
)
```

2. Query Neo4j for detailed analysis:
```python
with GraphDatabase.driver(uri, auth=(user, password)) as driver:
    # Get evaluation metrics
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Requirement)-[p:SIMILAR_ALL_MPNET_BASE_V2_A0_3_T0_15]->(t:Requirement)
            WITH collect(distinct {source: s.id, target: t.id}) as predicted
            MATCH (s:Requirement)-[g:GROUND_TRUTH]->(t:Requirement)
            WITH predicted, collect(distinct {source: s.id, target: t.id}) as ground_truth
            RETURN 
                size([x in predicted WHERE x in ground_truth]) as true_positives,
                size([x in predicted WHERE NOT x in ground_truth]) as false_positives,
                size([x in ground_truth WHERE NOT x in predicted]) as false_negatives
        """)
        metrics = result.single()
        
        # Calculate precision and recall
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
```

## Memory Usage Monitoring

### Neo4j Memory Usage

#### List Available Procedures
```cypher
// List all available procedures
CALL dbms.procedures()
YIELD name, signature, description
WHERE name CONTAINS 'memory' OR name CONTAINS 'system'
RETURN name, signature, description
ORDER BY name;
```

#### Basic Memory Stats (No APOC Required)
```cypher
// Get basic system memory info
CALL dbms.systemInfo()
YIELD name, value 
WHERE name CONTAINS 'memory' OR name CONTAINS 'heap'
RETURN name, value;
```

#### Query Memory Usage
```cypher
// Monitor memory usage of running queries
CALL dbms.listQueries()
YIELD 
    queryId,
    query,
    allocatedBytes,
    pageCacheHits,
    pageCacheMisses
RETURN 
    queryId,
    left(query, 100) as queryPreview,
    allocatedBytes/1024/1024.0 as memoryMB,
    pageCacheHits,
    pageCacheMisses,
    CASE WHEN pageCacheHits + pageCacheMisses > 0 
         THEN round(100.0 * pageCacheHits / (pageCacheHits + pageCacheMisses), 2)
         ELSE 0 
    END as cacheHitRatio
ORDER BY allocatedBytes DESC;
```

#### Database Size and Storage
```cypher
// Get database size information
CALL dbms.database.state()
YIELD name, currentStatus, role
WITH name, currentStatus, role
CALL db.stats()
YIELD entityCount, labelCount, relTypeCount, propertyKeyCount
RETURN 
    name as database,
    currentStatus as status,
    role,
    entityCount as totalEntities,
    labelCount as labels,
    relTypeCount as relationshipTypes,
    propertyKeyCount as properties;
```

#### Node and Relationship Statistics
```cypher
// Get detailed node and relationship counts
MATCH (n)
WITH count(n) as nodeCount
MATCH ()-[r]->()
WITH nodeCount, count(r) as relCount
RETURN 
    nodeCount as totalNodes,
    relCount as totalRelationships,
    nodeCount + relCount as totalElements,
    round(1.0 * relCount / nodeCount, 2) as avgRelationshipsPerNode,
    round((nodeCount * 64 + relCount * 96)/1024/1024.0, 2) as estimatedBaseSizeMB;
```

#### Label Distribution
```cypher
// Get distribution of nodes by label
MATCH (n)
WITH labels(n) as nodeLabels
UNWIND nodeLabels as label
WITH label, count(*) as count
WITH label, count, sum(count) as total
RETURN 
    label,
    count,
    round(100.0 * count / total, 2) as percentage,
    round(count * 64/1024/1024.0, 2) as estimatedSizeMB
ORDER BY count DESC;
```

#### Relationship Type Distribution
```cypher
// Get distribution of relationship types
MATCH ()-[r]->()
WITH type(r) as relType, count(*) as count
WITH relType, count, sum(count) as total
RETURN 
    relType,
    count,
    round(100.0 * count / total, 2) as percentage,
    round(count * 96/1024/1024.0, 2) as estimatedSizeMB
ORDER BY count DESC;
```

#### Transaction and Query Stats
```cypher
// Get transaction and query statistics
CALL dbms.listTransactions()
YIELD transactionId, currentQueryId, status, elapsedTimeMillis
WITH count(*) as activeTransactions, 
     sum(elapsedTimeMillis) as totalTimeMillis
CALL dbms.listQueries()
YIELD queryId
WITH activeTransactions, totalTimeMillis, count(*) as activeQueries
RETURN 
    activeTransactions,
    activeQueries,
    totalTimeMillis/1000.0 as totalRuntimeSeconds;
```

### Python Memory Monitoring

#### Monitor CUDA Memory
```python
def print_cuda_memory_stats():
    """Print CUDA memory statistics"""
    if torch.cuda.is_available():
        print("\nCUDA Memory Summary:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        print("\nMemory by Device:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB total")
```

#### Monitor Process Memory
```python
def get_process_memory():
    """Get current process memory usage"""
    import psutil
    process = psutil.Process()
    return {
        'rss': process.memory_info().rss / 1024**2,  # RSS in MB
        'vms': process.memory_info().vms / 1024**2,  # VMS in MB
        'percent': process.memory_percent()
    }

# Usage example
memory_stats = get_process_memory()
print(f"RSS Memory: {memory_stats['rss']:.2f} MB")
print(f"Virtual Memory: {memory_stats['vms']:.2f} MB")
print(f"Memory Usage: {memory_stats['percent']:.1f}%")
```

#### Memory Profiling Decorator
```python
import functools
import tracemalloc

def profile_memory(func):
    """Decorator to profile memory usage of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_mem = get_process_memory()
        
        try:
            result = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            end_mem = get_process_memory()
            
            print(f"\nMemory Profile for {func.__name__}:")
            print(f"Current memory usage: {current/1024**2:.2f} MB")
            print(f"Peak memory usage: {peak/1024**2:.2f} MB")
            print(f"Memory change: {(end_mem['rss'] - start_mem['rss']):.2f} MB")
            
            return result
        finally:
            tracemalloc.stop()
            
    return wrapper

# Usage example
@profile_memory
def analyze_requirements(source_file, target_file):
    # Your analysis code here
    pass
```

### Memory Optimization Tips

1. **Batch Processing**
   - Use batch sizes appropriate for your GPU memory
   - Monitor memory usage and adjust batch size dynamically
   ```python
   batch_size = min(32, max(1, available_gpu_memory // (768 * 4)))  # For 768-dim embeddings
   ```

2. **Clear Cache Regularly**
   ```python
   # Clear CUDA cache
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   
   # Clear Neo4j query cache
   with driver.session() as session:
       session.run("CALL db.clearQueryCaches()")
   ```

3. **Monitor Long-Running Operations**
   ```python
   def monitor_memory_usage(interval=60):
       """Monitor memory usage every interval seconds"""
       while True:
           cuda_stats = print_cuda_memory_stats()
           process_stats = get_process_memory()
           time.sleep(interval)
   
   # Run in separate thread
   import threading
   monitor_thread = threading.Thread(target=monitor_memory_usage)
   monitor_thread.daemon = True
   monitor_thread.start()
   ```

4. **Neo4j Memory Configuration**
   Add to neo4j.conf:
   ```conf
   # Memory settings
   dbms.memory.heap.initial_size=2g
   dbms.memory.heap.max_size=4g
   dbms.memory.pagecache.size=2g
   ```

## Configuration

Create a `.env` file with:

```env
# Model Selection
MODEL_LIST=[
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
]

# Analysis Parameters
ALPHA_VALUES=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Weight between TF-IDF and transformer
THRESHOLD_RANGE=[0.05, 0.075, ..., 0.5]  # Similarity thresholds to test
MIN_COVERAGE_THRESHOLD=0.9  # Minimum required coverage (90%)

# CUDA Configuration
CUDA_LAUNCH_BLOCKING=0
TORCH_USE_CUDA_DSA=0
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8

# Input Files
SOURCE_FILE=datasets/CM1/CM1-sourceArtifacts.xml
TARGET_FILE=datasets/CM1/CM1-targetArtifacts.xml
ANSWER_FILE=datasets/CM1/CM1-answerSet.xml
```

## Usage Example

```python
analyzer = SentenceTransformerAnalyzer(
    model_name="sentence-transformers/all-mpnet-base-v2",
    alpha=0.3  # 30% TF-IDF, 70% transformer-based similarity
)

# Initialize and analyze
analyzer.initialize()
results = analyzer.analyze_requirements(
    source_file="path/to/source.xml",
    target_file="path/to/target.xml",
    threshold=0.15  # Minimum similarity threshold
)
```

## Output Analysis

The tool generates comprehensive analysis including:

1. **Similarity Distribution**:
   - Histogram of similarity scores
   - Cumulative distribution function

2. **Performance Curves**:
   - Precision-Recall curves
   - ROC curves
   - F1 score vs threshold plots

3. **Coverage Analysis**:
   - Requirement coverage at different thresholds
   - False negative analysis

## Prerequisites

- Python 3.8+
- PyTorch
- spaCy
- Sentence Transformers
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Attribution

This project includes code that was generated or assisted by [Cursor AI](https://cursor.ai/) tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

Copyright (c) 2024-2025 Evan Montgomery-Recht

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.