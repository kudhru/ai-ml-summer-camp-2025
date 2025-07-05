# Simple RAG Pipeline with Gemini

This is a simple Retrieval-Augmented Generation (RAG) pipeline implementation using Google's Gemini API for both embeddings and text generation.

## Features

- 🧠 **Embeddings**: Uses Gemini's `text-embedding-004` model for semantic embeddings
- 🎯 **Generation**: Uses Gemini's `gemini-2.0-flash-001` model for answer generation
- 🔍 **Retrieval**: Cosine similarity search using scikit-learn
- 💾 **Storage**: Simple in-memory vector store with NumPy arrays
- 📄 **Documents**: Loads and chunks local text files
- 🚀 **Easy to use**: Minimal setup with automatic sample document creation

## Architecture

```
Query → Embedding → Vector Search → Context Retrieval → Answer Generation
```

### Components

1. **DocumentLoader**: Loads text files and splits them into overlapping chunks
2. **EmbeddingService**: Generates embeddings using Gemini's embedding model
3. **VectorStore**: In-memory storage with cosine similarity search
4. **RAGPipeline**: Orchestrates the entire RAG process

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Get a Gemini API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new project and get your API key
   - Add the key to your `.env` file

## Usage

### Basic Usage

```python
from rag.example_rag_101 import RAGPipeline
import os

# Initialize the pipeline
api_key = os.getenv("GEMINI_API_KEY")
rag = RAGPipeline(api_key)

# Ask a question
answer = rag.query("What is artificial intelligence?")
print(answer)
```

### Run the Example

```bash
python rag/example_rag_101.py
```

This will:
1. Create sample documents (if they don't exist)
2. Index the documents with embeddings
3. Run example queries
4. Start an interactive mode for custom questions

### Sample Documents

The pipeline automatically creates sample documents about:
- **Artificial Intelligence**: Overview, types, applications, and ethical concerns
- **Machine Learning**: Types, algorithms, process, and applications
- **Data Science**: Process, skills, tools, and applications

You can also add your own `.txt` files to the `rag/sample_documents/` directory.

## Configuration

### Chunk Settings

```python
# Adjust chunk size and overlap
loader = DocumentLoader(chunk_size=800, chunk_overlap=100)
```

### Retrieval Settings

```python
# Number of documents to retrieve
relevant_docs = rag.query("your question", top_k=3)
```

### Custom Document Directory

```python
# Use your own documents
rag = RAGPipeline(api_key, documents_dir="path/to/your/documents")
```

## Output Format

When you ask a question, the pipeline shows:

1. **Query**: Your question
2. **Retrieved Documents**: Top matching chunks with similarity scores
3. **Generated Answer**: Gemini's response based on the context

Example output:
```
Query: What is artificial intelligence?

--- Retrieved Documents ---
1. Similarity: 0.892 | Source: rag/sample_documents/artificial_intelligence.txt | Chunk: 0
   Content: Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed...

--- Generated Answer ---
Artificial Intelligence (AI) is the simulation of human intelligence in machines...
```

## How It Works

1. **Document Processing**: Text files are loaded and split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to a vector using Gemini's embedding model
3. **Query Processing**: User questions are also embedded using the same model
4. **Similarity Search**: Cosine similarity finds the most relevant document chunks
5. **Answer Generation**: Retrieved context is sent to Gemini for answer generation

## Limitations

- **In-memory storage**: Not suitable for large document collections
- **No persistence**: Embeddings are recalculated on each run
- **Sequential processing**: Embeddings are generated one at a time
- **Simple chunking**: Basic text splitting without semantic awareness

## Possible Improvements

- Add persistent storage (e.g., ChromaDB, Pinecone)
- Implement batch embedding generation
- Add more sophisticated chunking strategies
- Include document metadata and filtering
- Add conversation history for multi-turn dialogues
- Implement caching for embeddings

## Dependencies

- `google-genai`: Gemini API client
- `numpy`: Vector operations
- `scikit-learn`: Cosine similarity
- `python-dotenv`: Environment variable management

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```
   Error: GEMINI_API_KEY not found in environment variables
   ```
   Solution: Create a `.env` file with your API key

2. **No Documents Found**:
   The pipeline automatically creates sample documents if none exist

3. **Import Errors**:
   Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Performance Tips

- Use smaller chunk sizes for better precision
- Increase `top_k` for more context (but higher API costs)
- Consider the trade-off between chunk size and retrieval accuracy

## License

This is a simple educational example for learning RAG concepts with Gemini API. 