# 🔍 RAG Technology Index

A comprehensive repository of 100+ RAG (Retrieval-Augmented Generation) libraries, frameworks, and tools organized by category.

<p align="center">
  <a href="https://www.linkedin.com/in/esra-da%C4%9F-49653115b/">
    <img src="https://custom-icon-badges.demolab.com/badge/Esra%20Da%C4%9F-0A66C2?logo=linkedin-white&logoColor=fff" alt="LinkedIn">
  </a>
</p>

## Quick Links
||||
|---|---|---|
| [🏗️ RAG Frameworks](#rag-frameworks) | [📚 Document Processing](#document-processing) | [🔎 Vector Databases](#vector-databases) |
| [🧠 Embedding Models](#embedding-models) | [🔄 Chunking Strategies](#chunking-strategies) | [📊 Retrieval Methods](#retrieval-methods) |
| [🤖 Advanced RAG Architecture](#advanced-rag-architecture) | [🧪 Evaluation & Testing](#evaluation-and-testing) | [📈 RAG Optimization](#rag-optimization) |
| [🔧 RAG Tools](#rag-tools) | [📝 Query Transformation](#query-transformation) | [🧩 Multi-Modal RAG](#multi-modal-rag) |
| [🚀 Production & Deployment](#production-and-deployment) | [🔐 Security & Compliance](#security-and-compliance) | [📖 Learning Resources](#learning-resources) |


## RAG Frameworks

| Library | Description | Link |
|---------|-------------|------|
| LangChain | Leading framework for building RAG applications with document loaders, retrievers, and agents | [Link](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Widely used data framework that simplifies connecting custom data sources to LLMs | [Link](https://github.com/run-llama/llama_index) |
| Haystack | End-to-end NLP framework with extensive RAG functionality and pipeline architecture | [Link](https://github.com/deepset-ai/haystack) |
| DSPy | Stanford NLP's framework with declarative API for programming language models | [Link](https://github.com/stanfordnlp/dspy) |
| Llmware | Enterprise RAG framework with production-ready capabilities for specialized models | [Link](https://github.com/llmware-ai/llmware) |
| Embedchain | Open source RAG framework to create bots from any data source with minimal setup | [Link](https://github.com/embedchain/embedchain) |
| AutoChain | Framework for building and running reliable LLM applications with structured outputs | [Link](https://github.com/Forethought-Technologies/AutoChain) |
| Langflow | Visual low-code tool for building and prototyping RAG applications quickly | [Link](https://github.com/langflow-ai/langflow) |
| CrewAI | Framework for orchestrating role-playing agents in complex RAG workflows | [Link](https://github.com/joaomdmoura/crewAI) |
| FastRAG | Intel's research framework for efficient and optimized RAG pipelines | [Link](https://github.com/IntelLabs/fastRAG) |

## Document Processing

| Library | Description | Link |
|---------|-------------|------|
| Unstructured | Leading library for pre-processing and extracting information from unstructured data | [Link](https://github.com/Unstructured-IO/unstructured) |
| LangChain Document Loaders | Comprehensive collection of document loaders for various file types and data sources | [Link](https://python.langchain.com/docs/integrations/document_loaders/) |
| DocArray | Powerful library for representing, sending, and storing nested multimodal data | [Link](https://github.com/docarray/docarray) |
| PyMuPDF | Industry-standard library for extracting PDF content for LLM & RAG environments | [Link](https://pymupdf.readthedocs.io/en/latest/) |
| LlamaHub | Extensive collection of data loaders and readers for 100+ data sources | [Link](https://github.com/run-llama/llama-hub) |
| Txtai | All-in-one embeddings database with document processing capabilities | [Link](https://github.com/neuml/txtai) |
| Chroma | Open-source embedding database with powerful document processors | [Link](https://github.com/chroma-core/chroma) |
| Semantic Router | Intelligent document routing and processing system | [Link](https://github.com/aurelio-labs/semantic-router) |
| Detectron2 | Facebook AI's advanced document layout analysis system | [Link](https://github.com/facebookresearch/detectron2) |
| Document Understanding | Microsoft's comprehensive document processing toolkit | [Link](https://github.com/microsoft/unilm/tree/master/dit) |

## Vector Databases

| Library | Description | Link |
|---------|-------------|------|
| Faiss | Meta's industry-standard library for efficient similarity search with massive adoption | [Link](https://github.com/facebookresearch/faiss) |
| Milvus | Enterprise-grade vector database built for scalable similarity search | [Link](https://github.com/milvus-io/milvus) |
| Pinecone | Leading serverless vector database for production AI applications | [Link](https://www.pinecone.io/) |
| Weaviate | Open source vector database with integrated neural search capabilities | [Link](https://github.com/weaviate/weaviate) |
| Qdrant | High-performance vector similarity search engine with extended filtering | [Link](https://github.com/qdrant/qdrant) |
| Chroma | Developer-friendly embedding database built specifically for RAG applications | [Link](https://github.com/chroma-core/chroma) |
| PGVector | PostgreSQL extension for vector similarity search with SQL integration | [Link](https://github.com/pgvector/pgvector) |
| Vespa | Scalable engine for low-latency computation over large vector datasets | [Link](https://github.com/vespa-engine/vespa) |
| LanceDB | Vector database for machine learning applications with fast local processing | [Link](https://github.com/lancedb/lancedb) |
| Elasticsearch | Enterprise search platform with vector search capabilities | [Link](https://github.com/elastic/elasticsearch) |

## Embedding Models

| Library | Description | Link |
|---------|-------------|------|
| Sentence-Transformers | Python framework for state-of-the-art text embeddings | [Link](https://github.com/UKPLab/sentence-transformers) |
| Text Embedding Inference | Fast inference solution for text embedding models | [Link](https://github.com/huggingface/text-embeddings-inference) |
| E5 | Text embeddings optimized for retrieval tasks | [Link](https://huggingface.co/intfloat/e5-base-v2) |
| GTE | General Text Embeddings developed by Alibaba DAMO Academy | [Link](https://huggingface.co/Alibaba-NLP/gte-base) |
| BAAI Embeddings | Open-source embeddings optimized for various retrieval tasks | [Link](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Instructor | Compute embeddings with specific instructions for different use cases | [Link](https://github.com/HKUNLP/instructor-embedding) |
| OpenAI Embeddings | Embeddings from OpenAI's models like text-embedding-3-large | [Link](https://platform.openai.com/docs/guides/embeddings) |
| Cohere Embeddings | Text embeddings optimized for RAG applications | [Link](https://cohere.com/embeddings) |
| Nomic Embeddings | Open source text embeddings optimized for enterprise data | [Link](https://github.com/nomic-ai/nomic-embed-text) |
| LangChain Embeddings | Interfaces to various embedding models | [Link](https://python.langchain.com/docs/modules/data_connection/text_embedding/) |

## Chunking Strategies

| Library | Description | Link |
|---------|-------------|------|
| Chonkie | RAG chunking library that is lightweight, lightning-fast, and easy to use | [Link](https://github.com/chonkie-ai/chonkie) |
| LlamaIndex Chunking | Text splitters with various chunking strategies | [Link](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) |
| LangChain Text Splitters | Collection of chunking methods like recursive, semantic, and token-based | [Link](https://python.langchain.com/docs/modules/data_connection/document_transformers/) |
| Haystack Preprocessors | Text preprocessing and chunking components | [Link](https://docs.haystack.deepset.ai/docs/preprocessor) |
| Text Splitter Collection | Library dedicated to different text splitting strategies | [Link](https://github.com/langchain-ai/text-splitter) |
| Adaptive Chunking | Advanced chunking strategies that adapt to document structure | [Link](https://github.com/AkshitIreddy/Adaptive-RAG-Methods) |
| Semantic Chunking | Tool for creating semantically coherent chunks | [Link](https://github.com/Canner/semantic-chunking) |
| TextMesh | Toolkit for RAG chunking based on various boundaries | [Link](https://github.com/akdavid1/textmesh) |
| ChunkViz | Visualization tool for document chunking strategies | [Link](https://github.com/FullStackRetrieval-com/chunkviz) |
| Split by Header | Specialized chunker for documents with hierarchical structure | [Link](https://github.com/llamahub/split-by-header) |

## Retrieval Methods

| Library | Description | Link |
|---------|-------------|------|
| Rerankers | Lightweight unified API for various reranking models | [Link](https://github.com/AnswerDotAI/rerankers) |
| ColBERT | State-of-the-art neural search using late interaction | [Link](https://github.com/stanford-futuredata/ColBERT) |
| TART | Toolkit for Advanced Retrieval Techniques | [Link](https://github.com/facebookresearch/tart) |
| SBERT Reranker | Bidirectional cross-encoder for reranking search results | [Link](https://www.sbert.net/examples/applications/cross-encoder/README.html) |
| Hydra | Multi-headed approach to retrieval with ensemble methods | [Link](https://github.com/primeqa/primeqa) |
| DeepImpact | Learning deep document impact for dense retrieval | [Link](https://github.com/AmenRa/retriv) |
| SPLADE | SParse Lexical AnD Expansion model for retrieval | [Link](https://github.com/naver/splade) |
| HyDE | Hypothetical Document Embeddings for improved retrieval | [Link](https://github.com/texttron/hyde) |
| RAGatouille | Collection of advanced retrieval methods for RAG | [Link](https://github.com/bclavie/RAGatouille) |
| Parent Document Retriever | Hierarchical retrieval for long documents | [Link](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/parent_document_retriever.py) |

## Advanced RAG Architecture

| Library | Description | Link |
|---------|-------------|------|
| Agentic RAG | Framework for building agentic RAG applications with Vectara | [Link](https://vectara.github.io/py-vectara-agentic/latest/) |
| GraphRAG | Graph-based RAG implementation for complex reasoning | [Link](https://github.com/langchain-ai/langgraph) |
| Self-RAG | Self-reflective retrieval-augmented generation | [Link](https://github.com/AkariAsai/self-rag) |
| RAPTOR | Recursive Abstractive Processing for Tree-Organized Retrieval | [Link](https://github.com/McGill-NLP/RAPTOR) |
| FLARE | Forward-Looking Active REtrieval augmented generation | [Link](https://github.com/jzbjyb/FLARE) |
| CRAG | Contextual RAG with memory for long conversations | [Link](https://github.com/predibase/crag) |
| Adaptive RAG | RAG systems that dynamically adjust to query complexity | [Link](https://github.com/FullStackRetrieval-com/adaptive-rag) |
| RecurrentRAG | Implementation of recurrent memory mechanisms in RAG | [Link](https://github.com/IntelLabs/fastRAG) |
| KATE | Knowledge-Adaptive Text Extraction for RAG | [Link](https://github.com/neulab/kate) |
| IRAG | In-Context RAG with self-reflection and learning | [Link](https://github.com/wangyuxi96/IRAG) |

## Evaluation and Testing

| Library | Description | Link |
|---------|-------------|------|
| Ragas | Comprehensive evaluation framework for RAG systems with 3.5k+ stars | [Link](https://github.com/explodinggradients/ragas) |
| TruLens | Popular evaluation and observability toolkit for LLM applications with 1.7k+ stars | [Link](https://github.com/truera/trulens) |
| LangChain Evaluation | Official LangChain evaluation framework for RAG pipelines | [Link](https://python.langchain.com/docs/guides/evaluation/) |
| DeepEval | Industry-standard LLM evaluation framework with 1.4k+ stars | [Link](https://github.com/confident-ai/deepeval) |
| BEIR | Widely-used benchmark for information retrieval with 800+ stars | [Link](https://github.com/beir-cellar/beir) |
| LangSmith | Production-grade evaluation platform by LangChain with 1k+ stars | [Link](https://github.com/langchain-ai/langsmith-sdk) |
| Prometheus | Stanford CRFM's evaluation framework for generative AI with 1.2k+ stars | [Link](https://github.com/stanford-crfm/prometheus) |
| MLflow Evaluation | Evaluation capabilities in the popular MLflow platform | [Link](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html) |
| Giskard | Open-source LLM testing framework with 3k+ stars | [Link](https://github.com/Giskard-AI/giskard) |
| Promptfoo | Popular LLM prompt testing tool with 4.6k+ stars | [Link](https://github.com/promptfoo/promptfoo) |

## RAG Optimization

| Library | Description | Link |
|---------|-------------|------|
| LLMLingua | Library for compressing prompts to accelerate RAG performance | [Link](https://github.com/microsoft/LLMLingua) |
| Selective Context | Compress prompt and context to allow LLMs to process 2x more content | [Link](https://pypi.org/project/selective-context/) |
| GPTCache | Library for creating semantic cache for LLM queries | [Link](https://github.com/zilliztech/gptcache) |
| PCToolkit | Unified plug-and-play prompt compression toolkit | [Link](https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression) |
| FastRAG | Framework focusing on RAG optimization | [Link](https://github.com/IntelLabs/fastRAG) |
| ThreadSplit | Thread-based parallel chunking and processing for RAG | [Link](https://github.com/mcmonkeyprojects/threadedchunking) |
| RAGFlow | End-to-end optimization pipeline for RAG architectures | [Link](https://github.com/CarperAI/RAGFlow) |
| Semantic Router | Tool to optimize routing of queries to appropriate RAG components | [Link](https://github.com/aurelio-labs/semantic-router) |
| Token Tunnel | Optimization technique for maximizing context within token limits | [Link](https://github.com/hwchase17/token-tunnel) |
| Distill-Embed | Embedding distillation to optimize retrieval performance | [Link](https://github.com/embeddings-benchmark/mteb) |

## RAG Tools

| Library | Description | Link |
|---------|-------------|------|
| LangFlow | Low-code app builder for RAG and multi-agent AI applications | [Link](https://github.com/langflow-ai/langflow) |
| Vectara | Vector search platform with built-in RAG capabilities | [Link](https://github.com/vectara/vectara-python) |
| Arize Phoenix | Open-source AI observability platform for RAG experimentation | [Link](https://github.com/Arize-ai/phoenix) |
| Flowise | Open source drag & drop UI for building RAG workflows | [Link](https://github.com/FlowiseAI/Flowise) |
| Verba | RAG toolkit for interacting with documentation and knowledge bases | [Link](https://github.com/weaviate/Verba) |
| Embedchain | Framework to create RAG bots from any data source | [Link](https://github.com/embedchain/embedchain) |
| PrivateGPT | Privacy-focused RAG application that runs locally | [Link](https://github.com/imartinez/privateGPT) |
| LocalGPT | Interface for using LLMs with your files completely offline | [Link](https://github.com/PromtEngineer/localGPT) |
| H2O LLM Studio | Framework for fine-tuning LLMs for RAG tasks | [Link](https://github.com/h2oai/h2o-llmstudio) |
| DoctranSlim | Library for document transformation to enhance RAG | [Link](https://github.com/Arize-ai/doctran_slim) |

## Query Transformation

| Library | Description | Link |
|---------|-------------|------|
| Query Transformation | Library to modify and optimize user queries for better retrieval | [Link](https://github.com/hwchase17/query-transformation) |
| HyDE | Hypothetical Document Embeddings for query expansion | [Link](https://github.com/texttron/hyde) |
| Query Decomposer | Breaking complex queries into sub-queries for improved retrieval | [Link](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/multi_query.py) |
| Query Rewriting | Framework for transforming queries to improve RAG results | [Link](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_rewriter/base.py) |
| Step-Back Prompting | Technique to take a step back and generalize before query execution | [Link](https://github.com/google-research/big-bench/tree/main/notebooks/StepBack_Prompting) |
| Auto-Merging Retriever | Automatically merges and de-duplicates results from multiple retrievers | [Link](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/merger.py) |
| Active Retrieval | Framework for interactive query refinement | [Link](https://github.com/stanfordnlp/dspy) |
| Zero-Shot Query Generator | Generate diverse search queries without training examples | [Link](https://github.com/vlaca/zero-shot-query-generation) |
| Self-Query | Self-querying retriever that optimizes filter parameters | [Link](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/self_query/base.py) |
| Query Reformulation | System for iterative query reformulation based on feedback | [Link](https://github.com/castorini/pyserini) |

## Multi-Modal RAG

| Library | Description | Link |
|---------|-------------|------|
| MultiModal-GPT | Create and deploy multi-modal RAG applications | [Link](https://github.com/openai/openai-multimodal-cookbook) |
| Image-RAG | Framework for RAG with images and text | [Link](https://github.com/vishwa-rn/image-rag) |
| CLIP-RAG | CLIP-based multi-modal retrieval for RAG | [Link](https://github.com/prithivida/clip-retrieval) |
| Video-RAG | Processing and retrieving from video content | [Link](https://github.com/showlab/VideoChat) |
| Audio-RAG | Framework for RAG with audio content | [Link](https://github.com/m-bain/whisperX) |
| DocRAG | Specialized RAG framework for documents with images and text | [Link](https://github.com/impira/docquery) |
| Multimodal LlamaIndex | Multi-modal extensions for LlamaIndex | [Link](https://docs.llamaindex.ai/en/stable/module_guides/loading/multimedia/) |
| MultiBERTers | Model for processing text alongside other modalities for retrieval | [Link](https://github.com/castorini/multiBERTers) |
| FlashAttention-2 | Library enabling efficient attention for multimodal inputs | [Link](https://github.com/Dao-AILab/flash-attention) |
| Multimodal Haystack | Pipelines for building multimodal RAG in Haystack | [Link](https://github.com/deepset-ai/haystack-core-integrations) |

## Production and Deployment

| Library | Description | Link |
|---------|-------------|------|
| LlamaCloud | Cloud infrastructure for deploying and scaling RAG applications | [Link](https://www.llamaindex.ai/llama-cloud) |
| Langcorn | Serving LangChain LLM apps and agents with FastAPI | [Link](https://github.com/msoedov/langcorn) |
| Ray | Distributed compute framework for scaling RAG applications | [Link](https://github.com/ray-project/ray) |
| MLflow | Platform for ML lifecycle with RAG tracking capabilities | [Link](https://github.com/mlflow/mlflow) |
| BentoML | Platform for building and deploying RAG applications | [Link](https://github.com/bentoml/BentoML) |
| Watsonx.ai | IBM's cloud platform for deploying enterprise-grade RAG | [Link](https://github.com/IBM/watsonx-ai-python-sdk) |
| Pinecone Serverless | Managed vector database for RAG deployment | [Link](https://www.pinecone.io/products/serverless/) |
| Azure AI Search | Microsoft's RAG deployment solution | [Link](https://github.com/Azure/azure-search-vector-samples) |
| Chainlit | Build production-ready RAG applications quickly | [Link](https://github.com/Chainlit/chainlit) |
| Potent RAG | Framework for enterprise-ready RAG solutions | [Link](https://github.com/ewfrees/Potent-RAG) |

## Security and Compliance

| Library | Description | Link |
|---------|-------------|------|
| LLM Guard | Comprehensive security toolkit for LLM interactions | [Link](https://github.com/protectai/llm-guard) |
| Guardrails AI | Framework for adding guardrails to LLMs | [Link](https://github.com/guardrails-ai/guardrails) |
| NeMo Guardrails | NVIDIA's toolkit for LLM safety | [Link](https://github.com/NVIDIA/NeMo-Guardrails) |
| Garak | LLM vulnerability scanner | [Link](https://github.com/leondz/garak) |
| Azure Content Safety | Microsoft's content filtering solution | [Link](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview) |
| Adversarial Robustness Toolkit | IBM's toolkit for LLM security testing | [Link](https://github.com/Trusted-AI/adversarial-robustness-toolbox) |
| OpenAI Moderation | API for content moderation in LLM outputs | [Link](https://platform.openai.com/docs/guides/moderation) |
| LangChain PEMA | Framework for measuring LLM alignment and safety | [Link](https://github.com/hwchase17/langchain-pema) |
| Lakera Guard | Commercial LLM security solution with free tier | [Link](https://www.lakera.ai/products/lakera-guard) |
| Allen AI Trojan Detection | Tools for detecting backdoors in LLMs | [Link](https://github.com/allenai/trojan-detection) |

## Learning Resources

| Resource | Description | Link |
|---------|-------------|------|
| RAG with LlamaIndex | Official LlamaIndex guide to RAG | [Link](https://docs.llamaindex.ai/en/stable/use_cases/query_engine/) |
| Pinecone RAG Guide | Complete guide to building RAG applications | [Link](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| Langchain Documentation | RAG patterns and implementation guides | [Link](https://python.langchain.com/docs/modules/chains/popular/chat_vector_db) |
| Awesome LLM-RAG | Curated list of RAG papers and resources | [Link](https://github.com/teacherpeterpan/Awesome-LLM-RAG) |
| Azure OpenAI RAG Demo | Practical RAG implementation example | [Link](https://github.com/Azure-Samples/azure-search-openai-demo) |
| Haystack Tutorials | Hands-on RAG tutorials | [Link](https://haystack.deepset.ai/tutorials/25_rag_pipeline) |
| Weaviate RAG Guide | Building RAG with Weaviate | [Link](https://weaviate.io/blog/semantic-search-with-weaviate) |
| Original RAG Paper | Lewis et al. research paper | [Link](https://arxiv.org/abs/2005.11401) |
| RARR Benchmark | Benchmark for evaluating RAG systems | [Link](https://github.com/amazon-science/rarr) |
| Hugging Face RAG Guide | Guide to implementing RAG | [Link](https://huggingface.co/docs/transformers/model_doc/rag) |
## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=esradag/rag-technology-index&type=Date)](https://star-history.com/#)

Please consider giving a star, if you find this repository useful.
