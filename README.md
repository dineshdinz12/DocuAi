# Production-Ready PDF RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over multiple PDF documents. This production-grade implementation combines state-of-the-art natural language processing with persistent vector storage to deliver accurate, contextual responses with full source attribution.

## Overview

This chatbot leverages advanced machine learning techniques to extract, process, and retrieve information from PDF documents. Built with enterprise-grade components, it provides a scalable solution for document-based question answering with conversational memory and metadata-driven search capabilities.

## Architecture

The system follows a modular architecture pattern with clear separation of concerns:

### Core Components

- **Document Processing Pipeline**: Handles PDF text extraction, chunking, and metadata extraction
- **Embedding Generation**: Utilizes Google's Gemini embedding models for semantic understanding
- **Vector Database**: Pinecone-based storage for scalable similarity search
- **Language Model Interface**: Google Gemini Flash for response generation
- **Conversation Management**: Session-based memory with configurable retention
- **Web Interface**: Streamlit-based user interface with file upload and chat functionality

### Data Flow

1. **Document Ingestion**: PDFs are uploaded and parsed into structured text chunks
2. **Semantic Encoding**: Text chunks are converted to high-dimensional embeddings
3. **Vector Storage**: Embeddings and metadata are persisted in Pinecone vector database
4. **Query Processing**: User queries are embedded and matched against stored documents
5. **Context Retrieval**: Most relevant document chunks are retrieved based on semantic similarity
6. **Response Generation**: Language model generates contextual responses using retrieved information
7. **Source Attribution**: Responses include citations with document names and page numbers

## Features

### Document Management
- Multi-PDF upload and processing
- Automatic text extraction with page-level granularity
- Intelligent chunking with configurable overlap
- Metadata extraction including document names, page numbers, and section titles

### Intelligent Retrieval
- Semantic search using state-of-the-art embeddings
- Configurable similarity thresholds
- Metadata-based filtering for targeted searches
- Support for document-specific queries

### Conversational Interface
- Multi-turn conversation support
- Context preservation across interactions
- Session-based memory management
- Conversation history with source tracking

### Enterprise Features
- Environment-based configuration management
- Comprehensive error handling and logging
- Scalable vector database architecture
- Production-ready deployment patterns

## Technology Stack

### Core Dependencies
- **Python 3.8+**: Primary development language
- **Streamlit**: Web application framework
- **LangChain**: Document processing and text manipulation
- **Pinecone**: Vector database for similarity search
- **Google Generative AI**: Embedding generation and language modeling

### Supporting Libraries
- **PyPDF2**: PDF text extraction
- **Sentence Transformers**: Alternative embedding support
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing operations

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip package manager
- Active internet connection for API access

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root directory with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
GEMINI_API_KEY=your_gemini_api_key
```

### API Key Setup

#### Pinecone Configuration
1. Create an account at [Pinecone](https://www.pinecone.io/)
2. Generate an API key from the console
3. Note your environment region
4. Choose an index name (will be created automatically)

#### Google Gemini Setup
1. Access [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Ensure the Gemini API is enabled for your project

## Usage

### Starting the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Document Upload Process

1. Navigate to the sidebar "Document Management" section
2. Use the file uploader to select PDF documents
3. Click "Process" for each uploaded document
4. Wait for processing completion (indicated by success message)

### Querying Documents

1. Type questions in the chat input field
2. Use the document filter dropdown to restrict searches to specific files
3. Review responses with automatic source citations
4. Expand the "Sources" section to view relevant document excerpts

### Session Management

- **Clear Conversation**: Removes chat history while preserving document knowledge base
- **Document Filtering**: Restrict queries to specific uploaded documents
- **Source Attribution**: All responses include document and page references

## API Reference

### Core Classes

#### RAGChatbot
Main orchestration class that coordinates all system components.

```python
chatbot = RAGChatbot()
success = chatbot.upload_pdf("document.pdf")
response, sources = chatbot.chat("What is the main topic?")
```

#### ConfigManager
Handles environment variable loading and validation.

```python
config = ConfigManager()  # Automatically loads .env variables
```

#### PDFProcessor
Manages document parsing and text extraction.

```python
processor = PDFProcessor()
chunks = processor.extract_text_from_pdf("document.pdf")
```

### Key Methods

#### Document Processing
```python
upload_pdf(pdf_path: str) -> bool
```
Processes a PDF document and stores it in the vector database.

#### Query Processing
```python
chat(user_query: str, document_filter: str = None) -> Tuple[str, List[Dict]]
```
Processes user queries and returns responses with source information.

#### Search Operations
```python
search(query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]
```
Performs semantic search over stored documents.

## Configuration Options

### Text Processing
- **Chunk Size**: Configurable text segment length (default: 1000 characters)
- **Chunk Overlap**: Overlap between consecutive chunks (default: 200 characters)
- **Embedding Dimensions**: Vector dimensionality (768 for Google embeddings)

### Retrieval Settings
- **Top-K Results**: Number of similar chunks retrieved (default: 5)
- **Similarity Threshold**: Minimum similarity score for inclusion
- **Context Window**: Maximum tokens sent to language model

### Memory Management
- **Session Length**: Maximum conversation history retained (default: 20 messages)
- **Context Messages**: Recent messages included in responses (default: 6)

## Performance Considerations

### Scalability
- Vector database supports millions of documents
- Batch processing for large document sets
- Configurable embedding generation rates

### Optimization
- Text chunking optimized for semantic coherence
- Efficient vector storage and retrieval
- Memory management for long conversations

### Resource Management
- API rate limiting and error handling
- Graceful degradation for service interruptions
- Comprehensive logging for debugging

## Deployment

### Local Deployment
The application runs locally using Streamlit's development server, suitable for personal use and testing.

### Production Deployment
For production environments, consider:
- Container deployment using Docker
- Cloud platforms such as AWS, GCP, or Azure
- Load balancing for multiple concurrent users
- Database backup and recovery procedures

### Environment Variables for Production
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Troubleshooting

### Common Issues

#### Connection Errors
- Verify API keys are correctly configured
- Check network connectivity
- Confirm service availability

#### Processing Failures
- Ensure PDF files are not corrupted or password-protected
- Verify sufficient system memory for large documents
- Check file format compatibility

#### Performance Issues
- Monitor API rate limits
- Consider document size and complexity
- Review system resource utilization

### Logging and Debugging

The application includes comprehensive logging at multiple levels:
- INFO: Normal operation status
- WARNING: Non-critical issues
- ERROR: Processing failures and exceptions

Logs are output to the console and can be redirected to files for production use.

## Security Considerations

### API Key Management
- Store sensitive credentials in environment variables
- Use secure key rotation practices
- Implement access controls for production deployments

### Data Privacy
- PDF content is processed and stored in external services
- Consider data residency requirements
- Implement appropriate data retention policies

### Network Security
- Use HTTPS in production environments
- Implement proper authentication mechanisms
- Consider VPN or private network deployment

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting pull requests

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Maintain high test coverage
- Use type hints where appropriate

### Testing
```bash
pytest tests/
python -m pytest --cov=app tests/
```

## Support

For technical support and questions:
- Review the troubleshooting section
- Check existing issues in the repository
- Create detailed bug reports with reproduction steps

## Acknowledgments

This project leverages several open-source technologies and cloud services:
- Google's Generative AI platform for embeddings and language modeling
- Pinecone for vector database services
- LangChain for document processing utilities
- Streamlit for rapid web application development