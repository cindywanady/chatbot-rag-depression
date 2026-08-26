"""depression_rag - retrieval evaluation for a depression case-management RAG.

A reproducible pipeline over the Indonesian MoH primary-care mental-health
guideline: extraction -> chunking (three strategies) -> embedding + FAISS
indexing (four models) -> gold-QA retrieval evaluation, built as decoupled
modules behind Protocol interfaces (see README.md).

    domain/          frozen domain models (Chunk, Segment, PageText, configs)
    ports/           Protocol interfaces (Chunker, Embedder, TokenCounter, TextCleaner)
    config/          typed YAML config loading
    observability/   structured JSON logging + run-manifest writer
    reproducibility/ global seed control
    extraction/      PDF loading -> cleaning -> segmentation
    chunking/        fixed / recursive / structure-aware chunkers
    embedding/       sentence-transformers + raw-transformers backends
    indexing/        exact cosine FAISS index
    evaluation/      gold passages, questions, relevance mapping, metrics, stats
    output/          JSONL + Parquet chunk sinks
    pipeline/        orchestration wiring all of the above
    cli, cli_index   command-line entry points
"""

__version__ = "0.1.0"
