# 🚀 Quick Start Guide

## Installation

1. **Install basic requirements:**
```bash
pip install numpy
```

2. **Run the demo:**
```bash
python rag_system.py
```

This will create sample documents and show you how the RAG system works!

## Your First RAG System in 5 Lines

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.ingest_documents(["your_document.txt"])
result = rag.query("What is this document about?")
print(result['answer'])
```

## What You Get

✅ **rag_system.py** - Core RAG implementation (works standalone)
✅ **enhanced_rag.py** - Version with OpenAI/Claude integration
✅ **requirements.txt** - All dependencies
✅ **README.md** - Complete documentation

## Next Steps

1. **Try the demo** to see it in action
2. **Replace sample documents** with your own files
3. **Customize chunk_size** for your use case
4. **Add LLM integration** for better answers (optional)

## Integration Examples

### With OpenAI GPT
```python
import os
from enhanced_rag import EnhancedRAGSystem

os.environ["OPENAI_API_KEY"] = "sk-..."
rag = EnhancedRAGSystem(llm_provider="openai")
rag.ingest_documents(["docs.txt"])
result = rag.query_with_llm("Your question?")
```

### With Anthropic Claude
```python
import os
from enhanced_rag import EnhancedRAGSystem

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
rag = EnhancedRAGSystem(llm_provider="anthropic")
rag.ingest_documents(["docs.txt"])
result = rag.query_with_llm("Your question?")
```

## Common Use Cases

1. **Personal Knowledge Base** - Index your notes and documents
2. **Code Documentation** - Make your codebase searchable
3. **Research Assistant** - Query academic papers
4. **Customer Support** - Build a Q&A system from FAQs
5. **Study Tool** - Create flashcards from textbooks

## Tips for Best Results

- **Chunk size**: 300-500 words for most documents
- **Top K**: Start with 3, increase if answers lack context
- **Document quality**: Clean, well-formatted text works best
- **Query specificity**: More specific questions = better answers

Enjoy building with RAG! 🎉
