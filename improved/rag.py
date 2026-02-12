"""
RAG (Retrieval-Augmented Generation) Extension
Add document search to your voice assistant for factual accuracy
"""

import os
from typing import List, Dict
from llama_cpp import Llama
import chromadb
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """Simple RAG implementation for the voice assistant"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        print("✓ RAG system ready!")
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        doc_id = f"doc_{self.collection.count()}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
    
    def add_documents_from_file(self, filepath: str):
        """Load documents from a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks (simple paragraph splitting)
        chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, chunk in enumerate(chunks):
            self.add_document(
                chunk,
                metadata={'source': filepath, 'chunk': i}
            )
        
        print(f"✓ Added {len(chunks)} chunks from {filepath}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return documents
    
    def augment_prompt(self, user_query: str, system_prompt: str) -> str:
        """Augment user query with relevant context"""
        # Search for relevant documents
        relevant_docs = self.search(user_query, top_k=2)
        
        if not relevant_docs:
            return system_prompt + f"\n\nUser: {user_query}\nAssistant:"
        
        # Build context
        context_parts = ["Here is relevant information:"]
        for doc in relevant_docs:
            context_parts.append(f"- {doc['text'][:200]}...")
        
        context = "\n".join(context_parts)
        
        # Build augmented prompt
        augmented = f"""{system_prompt}

{context}

User: {user_query}
Assistant:"""
        
        return augmented


# --- Example Integration with Voice Assistant ---

def create_rag_enabled_assistant():
    """Example: Voice assistant with RAG"""
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Add some knowledge (example)
    rag.add_document(
        "The capital of France is Paris. Paris is known for the Eiffel Tower.",
        metadata={'topic': 'geography'}
    )
    rag.add_document(
        "Python was created by Guido van Rossum in 1991.",
        metadata={'topic': 'programming'}
    )
    
    # You can also load from files
    # rag.add_documents_from_file("knowledge/facts.txt")
    
    # Initialize LLM
    llm = Llama(
        model_path="./models/llama-3.2-3b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    
    # System prompt
    system_prompt = "You are a helpful assistant. Use the provided context to answer accurately."
    
    # Interactive loop
    print("\n🤖 RAG-Enabled Voice Assistant")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Augment prompt with RAG
        augmented_prompt = rag.augment_prompt(user_input, system_prompt)
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        response = llm(
            augmented_prompt,
            max_tokens=150,
            temperature=0.7,
            stream=True,
            stop=["User:"]
        )
        
        for chunk in response:
            print(chunk['choices'][0]['text'], end="", flush=True)
        
        print()


# --- Advanced: Integration with voice_assistant_improved.py ---

class RAGEnabledLLMEngine:
    """Drop-in replacement for LLMEngine with RAG support"""
    
    def __init__(self, llm_model_path: str):
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        self.rag = RAGSystem()
    
    def add_knowledge(self, text: str, metadata: Dict = None):
        """Add knowledge to RAG system"""
        self.rag.add_document(text, metadata)
    
    def load_knowledge_from_file(self, filepath: str):
        """Load knowledge from file"""
        self.rag.add_documents_from_file(filepath)
    
    def stream_response(self, user_input: str, system_prompt: str, 
                       conversation_history: List[Dict] = None) -> str:
        """Generate response with RAG augmentation"""
        
        # Build conversation context
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 exchanges
                history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
        
        # Search for relevant context
        relevant_docs = self.rag.search(user_input, top_k=2)
        
        context = ""
        if relevant_docs:
            context = "\n\nRelevant information:\n"
            for doc in relevant_docs:
                context += f"- {doc['text'][:150]}...\n"
        
        # Build prompt
        prompt = f"""{system_prompt}

{context}

{history_text}
User: {user_input}
Assistant:"""
        
        # Stream response
        buffer = ""
        full_response = ""
        
        for chunk in self.llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            stream=True,
            stop=["User:", "\n\n"]
        ):
            token = chunk['choices'][0]['text']
            print(token, end="", flush=True)
            
            buffer += token
            full_response += token
            
            # You can yield/queue to TTS here
            if any(p in token for p in ".!?"):
                # Send to TTS
                pass
        
        print()
        return full_response.strip()


# --- Usage Example ---

if __name__ == "__main__":
    print("\n=== RAG System Demo ===\n")
    
    # Option 1: Simple RAG demo
    # create_rag_enabled_assistant()
    
    # Option 2: RAG with knowledge base
    rag = RAGSystem()
    
    # Add knowledge
    print("Adding knowledge...")
    rag.add_document(
        "Anthropic created Claude, an AI assistant focused on being helpful, harmless, and honest.",
        {'topic': 'AI', 'company': 'Anthropic'}
    )
    rag.add_document(
        "llama.cpp is a C++ implementation of Meta's LLaMA models for efficient inference.",
        {'topic': 'AI', 'tool': 'llama.cpp'}
    )
    
    # Test search
    query = "Tell me about Claude"
    print(f"\nQuery: {query}")
    results = rag.search(query, top_k=2)
    
    print("\nRelevant documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc['text'][:100]}...")
        print(f"   Distance: {doc['distance']:.3f}")
    
    print("\n✓ RAG system working!")
    print("\nTo integrate with voice assistant:")
    print("1. Add this file to your project")
    print("2. Replace LLMEngine with RAGEnabledLLMEngine")
    print("3. Load your knowledge base")
    print("4. Enjoy more accurate responses!")