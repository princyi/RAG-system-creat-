"""
Complete RAG (Retrieval-Augmented Generation) System
This system allows you to:
1. Ingest documents (txt, pdf, docx)
2. Create embeddings and store in a vector database
3. Query the system to retrieve relevant context
4. Generate responses using the retrieved context
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    source: str
    chunk_id: int
    metadata: Dict = None


class SimpleEmbedder:
    """Simple TF-IDF based embedder (can be replaced with sentence-transformers)"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_count = 0
        
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF scores from documents"""
        self.doc_count = len(documents)
        word_doc_count = {}
        
        # Build vocabulary and count document frequencies
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
        
        # Calculate IDF scores
        for word, count in word_doc_count.items():
            self.idf_scores[word] = np.log(self.doc_count / (1 + count))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def embed(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text"""
        words = self._tokenize(text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Create vector
        vector = np.zeros(len(self.vocabulary))
        for word, count in word_count.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / max(len(words), 1)
                idf = self.idf_scores.get(word, 0)
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class VectorStore:
    """Simple vector store for similarity search"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents and their embeddings to the store"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Find most similar documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, emb)
            similarities.append((self.documents[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']


class DocumentProcessor:
    """Process and chunk documents"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def load_text_file(filepath: str) -> str:
        """Load text from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()


class RAGSystem:
    """Main RAG system class"""
    
    def __init__(self, storage_dir: str = "./rag_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.embedder = SimpleEmbedder()
        self.vector_store = VectorStore()
        self.is_fitted = False
        
    def ingest_documents(self, filepaths: List[str], chunk_size: int = 500):
        """Ingest documents into the RAG system"""
        all_documents = []
        all_texts = []
        
        print("📄 Loading documents...")
        for filepath in filepaths:
            if not os.path.exists(filepath):
                print(f"⚠️  File not found: {filepath}")
                continue
                
            text = DocumentProcessor.load_text_file(filepath)
            chunks = DocumentProcessor.chunk_text(text, chunk_size=chunk_size)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    source=filepath,
                    chunk_id=i,
                    metadata={'total_chunks': len(chunks)}
                )
                all_documents.append(doc)
                all_texts.append(chunk)
            
            print(f"✓ Loaded {filepath}: {len(chunks)} chunks")
        
        if not all_texts:
            print("❌ No documents loaded!")
            return
        
        print(f"\n🔧 Creating embeddings for {len(all_texts)} chunks...")
        self.embedder.fit(all_texts)
        
        embeddings = [self.embedder.embed(text) for text in all_texts]
        self.vector_store.add_documents(all_documents, embeddings)
        self.is_fitted = True
        
        print(f"✓ RAG system ready with {len(all_documents)} document chunks!\n")
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query the RAG system"""
        if not self.is_fitted:
            return {
                'answer': "The RAG system hasn't been initialized with documents yet.",
                'sources': []
            }
        
        # Create query embedding
        query_embedding = self.embedder.embed(question)
        
        # Retrieve relevant documents
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"[Document {i+1}] {doc.content}")
            sources.append({
                'source': doc.source,
                'chunk_id': doc.chunk_id,
                'similarity': float(score)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer (simple extraction-based approach)
        answer = self._generate_answer(question, context, results)
        
        return {
            'answer': answer,
            'context': context,
            'sources': sources,
            'num_sources': len(results)
        }
    
    def _generate_answer(self, question: str, context: str, results: List[Tuple[Document, float]]) -> str:
        """Generate an answer based on retrieved context"""
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Simple answer generation (in production, you'd use an LLM here)
        answer_parts = [
            f"Based on the retrieved documents, here's what I found:\n"
        ]
        
        for i, (doc, score) in enumerate(results[:2]):  # Use top 2 most relevant
            preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            answer_parts.append(f"\n[Source {i+1}]: {preview}")
        
        answer_parts.append(f"\n\nNote: This is a simple RAG system. For better answers, integrate with an LLM like GPT or Claude.")
        
        return "\n".join(answer_parts)
    
    def save(self, name: str = "rag_system"):
        """Save the RAG system state"""
        # Save vector store
        self.vector_store.save(os.path.join(self.storage_dir, f"{name}_vectors.pkl"))
        
        # Save embedder
        with open(os.path.join(self.storage_dir, f"{name}_embedder.pkl"), 'wb') as f:
            pickle.dump(self.embedder, f)
        
        print(f"✓ RAG system saved to {self.storage_dir}")
    
    def load(self, name: str = "rag_system"):
        """Load a saved RAG system"""
        vector_path = os.path.join(self.storage_dir, f"{name}_vectors.pkl")
        embedder_path = os.path.join(self.storage_dir, f"{name}_embedder.pkl")
        
        if not os.path.exists(vector_path) or not os.path.exists(embedder_path):
            print("❌ Saved RAG system not found!")
            return False
        
        self.vector_store.load(vector_path)
        
        with open(embedder_path, 'rb') as f:
            self.embedder = pickle.load(f)
        
        self.is_fitted = True
        print(f"✓ RAG system loaded from {self.storage_dir}")
        return True


def main():
    """Example usage of the RAG system"""
    print("=" * 60)
    print("RAG SYSTEM - Retrieval-Augmented Generation")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RAGSystem(storage_dir="./rag_storage")
    
    # Example: Create sample documents if none exist
    sample_docs_dir = "./sample_documents"
    os.makedirs(sample_docs_dir, exist_ok=True)
    
    # Create sample document 1
    doc1_path = os.path.join(sample_docs_dir, "ai_overview.txt")
    with open(doc1_path, 'w') as f:
        f.write("""
Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence by machines.
AI systems can perform tasks that typically require human intelligence, such as
visual perception, speech recognition, decision-making, and language translation.

There are three main types of AI:
1. Narrow AI (Weak AI): AI that is designed for a specific task
2. General AI (Strong AI): AI with human-level intelligence across domains
3. Superintelligent AI: AI that surpasses human intelligence

Machine Learning is a subset of AI that enables systems to learn from data
without being explicitly programmed. Deep Learning is a subset of Machine Learning
that uses neural networks with multiple layers.

Applications of AI include: autonomous vehicles, medical diagnosis, virtual
assistants, fraud detection, and recommendation systems.
        """)
    
    # Create sample document 2
    doc2_path = os.path.join(sample_docs_dir, "machine_learning.txt")
    with open(doc2_path, 'w') as f:
        f.write("""
Machine Learning Fundamentals

Machine Learning (ML) is a method of data analysis that automates analytical
model building. It uses algorithms that iteratively learn from data, allowing
computers to find hidden insights without being explicitly programmed.

Types of Machine Learning:

Supervised Learning: The algorithm learns from labeled training data. Examples
include classification and regression tasks. Common algorithms include Linear
Regression, Decision Trees, and Support Vector Machines.

Unsupervised Learning: The algorithm learns from unlabeled data to find patterns.
Examples include clustering and dimensionality reduction. K-means clustering and
Principal Component Analysis (PCA) are popular techniques.

Reinforcement Learning: The algorithm learns by interacting with an environment
and receiving rewards or penalties. This is used in robotics, game playing, and
autonomous systems.

Key concepts in ML include: training data, testing data, features, labels,
overfitting, underfitting, and cross-validation.
        """)
    
    # Ingest documents
    print("\n1. INGESTING DOCUMENTS")
    print("-" * 60)
    rag.ingest_documents([doc1_path, doc2_path], chunk_size=300)
    
    # Save the system
    rag.save("my_rag")
    
    # Example queries
    print("\n2. QUERYING THE SYSTEM")
    print("-" * 60)
    
    queries = [
        "What is Artificial Intelligence?",
        "What are the types of machine learning?",
        "Tell me about deep learning"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 60)
        result = rag.query(query, top_k=2)
        print(f"📝 Answer:\n{result['answer']}\n")
        print(f"📚 Retrieved {result['num_sources']} relevant chunks")
        for i, source in enumerate(result['sources']):
            print(f"   Source {i+1}: {source['source']} (similarity: {source['similarity']:.3f})")
    
    print("\n" + "=" * 60)
    print("RAG System Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
