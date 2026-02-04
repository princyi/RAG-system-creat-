# RAG System - Retrieval-Augmented Generation

A complete, production-ready RAG (Retrieval-Augmented Generation) system that allows you to create a knowledge base from documents and query it with natural language.

## 🌟 Features

- **Document Ingestion**: Load and process text documents
- **Smart Chunking**: Automatically splits documents into manageable chunks with overlap
- **Vector Embeddings**: Creates TF-IDF embeddings (upgradeable to transformer models)
- **Similarity Search**: Finds the most relevant document chunks for queries
- **LLM Integration**: Optional integration with OpenAI or Anthropic Claude
- **Persistent Storage**: Save and load your knowledge base
- **Easy to Use**: Simple API with minimal setup

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install numpy
```

For enhanced features (optional):
```bash
pip install sentence-transformers  # Better embeddings
pip install openai                  # OpenAI integration
pip install anthropic              # Claude integration
```

### 2. Basic Usage

```python
from rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem()

# Ingest documents
rag.ingest_documents([
    "document1.txt",
    "document2.txt"
])

# Query the system
result = rag.query("What is machine learning?")
print(result['answer'])

# Save for later use
rag.save("my_knowledge_base")
```

### 3. Run the Demo

```bash
python rag_system.py
```

This will:
- Create sample documents
- Ingest them into the RAG system
- Run example queries
- Show retrieved sources and answers

## 🚀 Advanced Usage

### With LLM Integration

```python
from enhanced_rag import EnhancedRAGSystem
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize with LLM
rag = EnhancedRAGSystem(llm_provider="openai")

# Ingest documents
rag.ingest_documents(["docs/document1.txt"])

# Query with LLM-generated answers
result = rag.query_with_llm("Explain the concept in detail")
print(result['answer'])
```

### Interactive Mode

```bash
python enhanced_rag.py
```

Then enter your questions interactively!

## 📁 File Structure

```
rag_system/
├── rag_system.py          # Core RAG implementation
├── enhanced_rag.py        # LLM-enhanced version
├── requirements.txt       # Dependencies
├── README.md             # This file
├── rag_storage/          # Saved vector stores (created automatically)
└── sample_documents/     # Example documents (created by demo)
```

## 🔧 How It Works

1. **Document Processing**: 
   - Documents are split into overlapping chunks
   - Each chunk becomes a searchable unit

2. **Embedding Creation**:
   - Text is converted to numerical vectors using TF-IDF
   - Vectors capture semantic meaning

3. **Vector Storage**:
   - Embeddings are stored with metadata
   - Enables fast similarity search

4. **Query Processing**:
   - Query is converted to same vector space
   - Similar documents are retrieved using cosine similarity
   - Context is assembled from top matches

5. **Answer Generation**:
   - Basic: Returns relevant document chunks
   - Enhanced: Uses LLM to generate natural answers

## 📊 API Reference

### RAGSystem

#### `__init__(storage_dir="./rag_storage")`
Initialize the RAG system.

#### `ingest_documents(filepaths, chunk_size=500)`
Load documents into the system.
- `filepaths`: List of file paths to ingest
- `chunk_size`: Number of words per chunk

#### `query(question, top_k=3)`
Query the knowledge base.
- `question`: The question to ask
- `top_k`: Number of relevant chunks to retrieve

Returns:
```python
{
    'answer': str,
    'context': str,
    'sources': List[Dict],
    'num_sources': int
}
```

#### `save(name="rag_system")`
Save the RAG system to disk.

#### `load(name="rag_system")`
Load a saved RAG system.

### EnhancedRAGSystem

Extends RAGSystem with LLM capabilities.

#### `query_with_llm(question, top_k=3, model=None)`
Query with LLM-generated answers.

## 🎯 Use Cases

1. **Document Q&A**: Ask questions about your documents
2. **Knowledge Base**: Build a searchable company knowledge base
3. **Research Assistant**: Query research papers and articles
4. **Customer Support**: Create a support knowledge base
5. **Personal Notes**: Search your personal notes and documents

## 🔄 Upgrading

### Better Embeddings

Replace the `SimpleEmbedder` with sentence-transformers:

```python
from sentence_transformers import SentenceTransformer

class TransformerEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(self, text):
        return self.model.encode(text)
```

### Vector Database

Replace `VectorStore` with FAISS for better performance:

```python
import faiss

# Create FAISS index
dimension = 384  # embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Search
distances, indices = index.search(query_embedding, k=5)
```

## 📝 Example Output

```
❓ Query: What is Artificial Intelligence?
────────────────────────────────────────────────────────
📝 Answer:
Based on the retrieved documents, here's what I found:

[Source 1]: Artificial Intelligence (AI) is the simulation of human 
intelligence by machines. AI systems can perform tasks that typically 
require human intelligence, such as visual perception, speech 
recognition, decision-making, and language translation...

📚 Retrieved 2 relevant chunks
   Source 1: ./sample_documents/ai_overview.txt (similarity: 0.856)
   Source 2: ./sample_documents/ai_overview.txt (similarity: 0.723)
```

## ⚙️ Configuration

### Chunk Size
Adjust based on your documents:
- Small chunks (200-300 words): Better precision
- Large chunks (500-800 words): More context

### Top K
Number of chunks to retrieve:
- Fewer (2-3): More focused answers
- More (5-7): Broader context

## 🐛 Troubleshooting

**Problem**: "No documents loaded"
- Check file paths are correct
- Ensure files exist and are readable

**Problem**: "Poor retrieval quality"
- Try adjusting chunk_size
- Increase top_k parameter
- Consider using better embeddings (sentence-transformers)

**Problem**: "LLM integration not working"
- Verify API key is set correctly
- Check package is installed
- Ensure you have API credits

## 🤝 Contributing

Improvements welcome! Consider:
- Adding support for more file types (PDF, DOCX)
- Implementing better chunking strategies
- Adding metadata filtering
- Improving answer generation

## 📄 License

This is a demonstration/educational project. Use freely!

## 🔗 Resources

- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📧 Support

For issues or questions, please check the code comments or modify as needed for your use case.
