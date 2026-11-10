# 🔎 DeepScout AI - Agentic AI Research Assistant

DeepScout AI is an intelligent research assistant that combines document analysis with web research to provide comprehensive, evidence-based answers. It features advanced RAG (Retrieval-Augmented Generation), evidence sufficiency checking, and intelligent query routing.

## 🚀 Features

### Core Capabilities
- **Four Query Routes**: Intelligently routes questions to the appropriate research path
  - **General Questions**: Direct AI knowledge without external sources
  - **Document Research**: Answers based solely on uploaded PDF content
  - **Web Research**: Current information from the open web
  - **Mixed Research**: Combines PDF analysis with modern web research

### Advanced Research Pipeline
- **Evidence Critic System**: Evaluates whether retrieved evidence actually answers the question
- **Targeted Retrieval**: Automatically refines queries when evidence is insufficient
- **PDF Attribution**: Every PDF claim includes exact page numbers and source identification
- **Source Distinction**: Clearly separates PDF evidence from web evidence in mixed answers

### Technical Features
- **Multi-PDF Support**: Upload and analyze multiple documents simultaneously
- **Intelligent Chunking**: Context-aware text segmentation with source preservation
- **Semantic Search**: FAISS-based vector similarity search with reranking
- **Conversational Memory**: Maintains context across follow-up questions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Query Router  │───▶│  Route Execution│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │                      │
                                 ▼                      ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │   PDF Upload    │    │  Web Search     │
                        │   & Chunking    │    │  (Tavily API)   │
                        └─────────────────┘    └─────────────────┘
                                 │                      │
                                 ▼                      ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │   RAG Pipeline  │◀───┤  Evidence       │
                        │                 │    │  Critic System  │
                        └─────────────────┘    └─────────────────┘
                                 │                      │
                                 ▼                      ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │   LLM Synthesis │    │  Targeted       │
                        │   with Strict   │    │  Retrieval      │
                        │   Attribution   │    └─────────────────┘
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Final Answer  │
                        │  with Sources   │
                        └─────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Python 3.10+
- **LLM Integration**: Google Gemini API (with fallback support)
- **Vector Database**: FAISS for semantic search
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Web Search**: Tavily API for current information
- **PDF Processing**: PyPDF2 for text extraction
- **Frontend**: Streamlit for interactive UI
- **Testing**: Pytest with comprehensive test suite

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git
- API keys for:
  - Google Gemini API
  - Tavily API (for web research)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DeepScout
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LLM_PROVIDER=gemini  # Options: gemini, mock
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage Examples

### 1. Document Research
```
Upload: "Attention Is All You Need" (Transformer paper)
Question: "What architecture does this paper propose?"
Answer: "Based on the uploaded document (page 2), the paper proposes a Transformer architecture that uses self-attention instead of recurrent layers..."
```

### 2. Mixed Research
```
Upload: Same Transformer paper
Question: "How does this compare with modern Transformer architectures?"
Answer: "From the uploaded document (page 2): The original Transformer uses self-attention... From web sources: Modern Transformers use innovations like RMSNorm and RoPE... Comparison: While the original paper focused on..."
```

### 3. Web Research
```
Question: "What are the latest developments in AI?"
Answer: "According to recent sources [Source 1], the latest developments include..."
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest test_*.py rag/test_*.py -v

# Run specific test modules
pytest test_evidence_critic.py -v
pytest test_evidence_sufficiency.py -v
pytest rag/test_query_rewriter.py -v
```

## 🔧 Project Structure

```
DeepScout/
├── app.py                    # Streamlit web application
├── router.py                 # Query classification and routing
├── research/                 # Core research logic
│   ├── orchestrator.py       # Main research pipeline controller
│   ├── evidence_critic.py    # Evidence sufficiency evaluation
│   └── __init__.py
├── rag/                      # Retrieval-Augmented Generation
│   ├── chunker.py           # PDF text segmentation
│   ├── retriever.py         # Vector similarity search
│   ├── embeddings.py        # Text embedding generation
│   ├── vector_store.py      # FAISS vector storage
│   ├── context_builder.py   # Prompt construction with attribution
│   ├── query_rewriter.py    # Conversational query refinement
│   └── test_*.py           # RAG module tests
├── llm/                     # LLM provider integrations
│   ├── llm_service.py       # LLM provider router
│   ├── gemini_client.py     # Google Gemini API client
│   ├── mock_llm.py          # Mock LLM for testing
│   └── test_llm.py
├── web/                     # Web research module
│   ├── web_search.py        # Tavily API integration
│   └── test_web_search.py
├── tests/                   # Integration and unit tests
│   ├── test_evidence_critic.py
│   ├── test_evidence_sufficiency.py
│   ├── test_router.py
│   └── test_orchestrator_integration.py
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🎨 UI Screenshots

### Main Interface
![Main Interface](docs/screenshots/main-interface.png)

### Multi-PDF Upload
![Multi-PDF Upload](docs/screenshots/multi-pdf-upload.png)

### Research Results with Sources
![Research Results](docs/screenshots/research-results.png)

### Evidence Critic in Action
![Evidence Critic](docs/screenshots/evidence-critic.png)

## 📊 Performance

- **Query Processing**: < 5 seconds for most queries
- **PDF Processing**: ~2 seconds per page
- **Embedding Generation**: ~0.5 seconds per chunk
- **Web Search**: ~3 seconds per query
- **Answer Generation**: ~2-10 seconds depending on complexity

## 🔍 Evidence Critic System

The Evidence Critic is a key innovation that evaluates whether retrieved evidence actually answers the user's question:

1. **Evidence Analysis**: Examines retrieved chunks for relevance and completeness
2. **Missing Aspect Detection**: Identifies what information is lacking
3. **Query Refinement**: Generates better queries for targeted retrieval
4. **Iterative Improvement**: Up to 2 retrieval attempts with refinement

## 📝 Citation & Attribution

DeepScout ensures rigorous academic standards:
- **PDF Claims**: Always include exact page numbers: "(page 2)"
- **Web Claims**: Attribute to specific sources: "According to [Source Title]..."
- **Mixed Answers**: Clearly separate PDF vs web evidence
- **No Invention**: Never creates unsupported claims or citations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for LLM capabilities
- SentenceTransformers for embeddings
- FAISS team for vector search
- Tavily for web search API
- Streamlit for the web framework

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for implementation examples

---

**DeepScout AI** - Making research intelligent, reliable, and attribution-perfect. 🔎