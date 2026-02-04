"""
Enhanced RAG System with LLM Integration
This version supports integration with OpenAI, Anthropic Claude, or local models
"""

import os
import json
from typing import List, Dict, Optional
from rag_system import RAGSystem, Document


class EnhancedRAGSystem(RAGSystem):
    """Extended RAG system with LLM integration"""
    
    def __init__(self, storage_dir: str = "./rag_storage", llm_provider: str = None):
        super().__init__(storage_dir)
        self.llm_provider = llm_provider
        self.llm_client = None
        
        # Initialize LLM client if provider is specified
        if llm_provider:
            self._initialize_llm(llm_provider)
    
    def _initialize_llm(self, provider: str):
        """Initialize LLM client based on provider"""
        if provider.lower() == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = openai.OpenAI(api_key=api_key)
                    print("✓ OpenAI client initialized")
                else:
                    print("⚠️  OPENAI_API_KEY not found in environment")
            except ImportError:
                print("⚠️  OpenAI package not installed. Run: pip install openai")
        
        elif provider.lower() == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.llm_client = anthropic.Anthropic(api_key=api_key)
                    print("✓ Anthropic client initialized")
                else:
                    print("⚠️  ANTHROPIC_API_KEY not found in environment")
            except ImportError:
                print("⚠️  Anthropic package not installed. Run: pip install anthropic")
    
    def query_with_llm(self, question: str, top_k: int = 3, model: str = None) -> Dict:
        """Query the RAG system and generate answer using LLM"""
        
        # Retrieve relevant documents
        query_embedding = self.embedder.embed(question)
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'context': ''
            }
        
        # Build context
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"Document {i+1}:\n{doc.content}")
            sources.append({
                'source': doc.source,
                'chunk_id': doc.chunk_id,
                'similarity': float(score)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with LLM
        answer = self._generate_llm_answer(question, context, model)
        
        return {
            'answer': answer,
            'context': context,
            'sources': sources,
            'num_sources': len(results)
        }
    
    def _generate_llm_answer(self, question: str, context: str, model: Optional[str]) -> str:
        """Generate answer using LLM"""
        
        if not self.llm_client:
            return self._generate_answer(question, context, [])
        
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            if self.llm_provider == "openai":
                model = model or "gpt-4o-mini"
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                model = model or "claude-sonnet-4-5-20250929"
                response = self.llm_client.messages.create(
                    model=model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return self._generate_answer(question, context, [])
        
        return "LLM provider not supported"


def demo_enhanced_rag():
    """Demo the enhanced RAG system"""
    print("=" * 70)
    print("ENHANCED RAG SYSTEM - With LLM Integration")
    print("=" * 70)
    
    # You can initialize with a provider if you have API keys set
    # rag = EnhancedRAGSystem(llm_provider="openai")
    # or
    # rag = EnhancedRAGSystem(llm_provider="anthropic")
    
    rag = EnhancedRAGSystem()
    
    # Load previously saved system or create new
    if not rag.load("my_rag"):
        print("\nNo saved RAG system found. Please run rag_system.py first.")
        return
    
    print("\n" + "=" * 70)
    print("RAG System Loaded!")
    print("=" * 70)
    
    # Interactive query loop
    print("\nEnter your questions (or 'quit' to exit):")
    print("-" * 70)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        result = rag.query(question, top_k=3)
        
        print("\n📝 Answer:")
        print("-" * 70)
        print(result['answer'])
        print("\n📚 Sources:")
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['source']} (similarity: {source['similarity']:.3f})")
        print("-" * 70)


if __name__ == "__main__":
    demo_enhanced_rag()
