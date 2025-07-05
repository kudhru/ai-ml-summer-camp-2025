import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Simple document loader for text files."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """Load all text files from a directory and split them into chunks."""
        documents = []
        doc_dir = Path(directory)
        
        if not doc_dir.exists():
            print(f"Directory {directory} doesn't exist. Creating sample documents...")
            self._create_sample_documents(doc_dir)
        
        for file_path in doc_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    chunks = self._split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'source': str(file_path),
                            'chunk_id': i
                        })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return documents
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # If we're not at the end, try to split at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _create_sample_documents(self, doc_dir: Path):
        """Create sample documents for testing."""
        doc_dir.mkdir(exist_ok=True)
        
        sample_docs = {
            "artificial_intelligence.txt": """
            Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

            AI can be categorized into two main types: narrow AI and general AI. Narrow AI, also known as weak AI, is designed to perform a narrow task such as facial recognition, internet searches, or driving a car. General AI, also known as strong AI, refers to machines that possess the ability to understand, learn, and apply knowledge across a wide range of tasks at a level comparable to human intelligence.

            Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to analyze various factors of data.

            The applications of AI are numerous and growing rapidly. In healthcare, AI is used for drug discovery, medical imaging, and personalized treatment plans. In finance, AI helps with fraud detection, algorithmic trading, and risk assessment. In transportation, AI powers autonomous vehicles and traffic management systems.

            However, AI also raises important ethical concerns including privacy, job displacement, and the potential for bias in decision-making algorithms. As AI technology continues to advance, it's crucial to address these challenges while maximizing the benefits of this powerful technology.
            """,
            
            "machine_learning.txt": """
            Machine Learning (ML) is a branch of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience without being explicitly programmed for that task.

            There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common supervised learning tasks include classification and regression. Unsupervised learning finds patterns in data without labeled examples, such as clustering and dimensionality reduction. Reinforcement learning involves an agent learning to make decisions through interaction with an environment, receiving rewards or penalties for its actions.

            Popular machine learning algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks. Each algorithm has its strengths and is suited for different types of problems.

            The machine learning process typically involves several steps: data collection, data preprocessing, feature selection, model training, model evaluation, and model deployment. Data quality is crucial for successful machine learning projects, as poor quality data can lead to inaccurate models.

            Machine learning has revolutionized many industries. In technology, it powers recommendation systems, search engines, and natural language processing. In business, it enables predictive analytics, customer segmentation, and automated decision-making. The field continues to evolve rapidly with new techniques and applications emerging regularly.
            """,
            
            "data_science.txt": """
            Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines aspects of statistics, computer science, information science, and domain expertise to analyze and interpret complex data.

            The data science process typically follows these stages: data collection, data cleaning and preprocessing, exploratory data analysis, model building, model validation, and deployment. This process is often iterative, with data scientists going back and forth between different stages as they refine their understanding of the data and improve their models.

            Key skills for data scientists include programming languages like Python and R, statistical analysis, data visualization, machine learning, and domain knowledge. Tools commonly used in data science include Jupyter notebooks, pandas, NumPy, scikit-learn, TensorFlow, and various visualization libraries.

            Data visualization is a crucial aspect of data science, as it helps communicate findings to stakeholders and can reveal patterns that might not be apparent in raw data. Common visualization techniques include histograms, scatter plots, heat maps, and interactive dashboards.

            Big data technologies have become increasingly important in data science, as organizations deal with ever-growing volumes of data. Technologies like Hadoop, Spark, and cloud computing platforms enable data scientists to work with datasets that would be impossible to process on a single machine.

            The applications of data science are vast and continue to grow. From predicting customer behavior to optimizing supply chains, from detecting fraud to improving medical diagnoses, data science is transforming how organizations make decisions and operate.
            """
        }
        
        for filename, content in sample_docs.items():
            with open(doc_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        print(f"Created sample documents in {doc_dir}")

class EmbeddingService:
    """Service for generating embeddings using Gemini."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "text-embedding-004"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(768)  # Default embedding size
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)

class VectorStore:
    """Simple in-memory vector store."""
    
    def __init__(self):
        self.embeddings = None
        self.documents = []
    
    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        """Add documents and their embeddings to the store."""
        self.documents = documents
        self.embeddings = embeddings
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """Find most similar documents to the query."""
        if self.embeddings is None:
            return []
        
        # Reshape query embedding for cosine similarity calculation
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], similarities[idx]))
        
        return results

class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""
    
    def __init__(self, api_key: str, documents_dir: str = "rag/sample_documents"):
        self.embedding_service = EmbeddingService(api_key)
        self.vector_store = VectorStore()
        self.client = genai.Client(api_key=api_key)
        self.generation_model = "gemini-2.0-flash-001"
        self.documents_dir = documents_dir
        
        # Load and index documents
        self._load_and_index_documents()
    
    def _load_and_index_documents(self):
        """Load documents and create embeddings."""
        print("Loading and indexing documents...")
        
        # Load documents
        loader = DocumentLoader(chunk_size=800, chunk_overlap=100)
        documents = loader.load_documents(self.documents_dir)
        
        if not documents:
            print("No documents found!")
            return
        
        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_service.get_embeddings(texts)
        
        # Store in vector store
        self.vector_store.add_documents(documents, embeddings)
        
        print(f"Successfully indexed {len(documents)} document chunks")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Answer a question using RAG."""
        print(f"\nQuery: {question}")
        
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(question)
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query_embedding, top_k)
        
        if not relevant_docs:
            return "I don't have any relevant information to answer your question."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (similarity: {score:.3f}):\n{doc['content']}"
            for i, (doc, score) in enumerate(relevant_docs)
        ])
        
        # Generate answer using Gemini
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=prompt
            )
            
            answer = response.text
            
            # Display retrieval information
            print("\n--- Retrieved Documents ---")
            for i, (doc, score) in enumerate(relevant_docs):
                print(f"{i+1}. Similarity: {score:.3f} | Source: {doc['source']} | Chunk: {doc['chunk_id']}")
                print(f"   Content: {doc['content'][:100]}...")
            
            print(f"\n--- Generated Answer ---")
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer."

def main():
    """Main function to demonstrate the RAG pipeline."""
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return
    
    print("🚀 Starting Simple RAG Pipeline with Gemini")
    print("=" * 50)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(api_key)
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        # "What are the main types of machine learning?",
        # "How does data science differ from machine learning?",
        # "What are the applications of AI in healthcare?",
        # "What tools are commonly used in data science?"
    ]
    
    print("\n🔍 Running example queries...")
    for query in queries:
        answer = rag.query(query)
        print(f"\n💬 Answer: {answer}")
        print("-" * 80)
    
    # Interactive mode
    print("\n🎯 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                answer = rag.query(user_query)
                print(f"\n💬 Answer: {answer}")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thank you for using the RAG pipeline!")

if __name__ == "__main__":
    main()
