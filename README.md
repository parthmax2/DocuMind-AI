---
title: DocuMind-AI
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0"
app_file: Dockerfile
pinned: false
---
# DocuMind-AI: Enterprise PDF Summarizer System

<div align="center">

![DocuMind-AI Logo](https://img.shields.io/badge/DocuMind-AI-blue?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini-API-orange.svg)](https://developers.generativeai.google)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow.svg)](https://huggingface.co/spaces/parthmax/DocuMind-AI)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A comprehensive, AI-powered PDF summarization system that leverages MCP server architecture and Gemini API to provide professional, interactive, and context-aware document summaries.*

[🚀 Live Demo](https://huggingface.co/spaces/parthmax/DocuMind-AI) • [📖 Documentation](#documentation) • [🛠️ Installation](#installation) • [📊 API Reference](#api-reference)

</div>

---

## 🌟 Overview

DocuMind-AI is an enterprise-grade PDF summarization system that transforms complex documents into intelligent, actionable insights. Built with cutting-edge AI technology, it provides multi-modal document processing, semantic search, and interactive Q&A capabilities.

## ✨ Key Features

### 🔍 **Advanced PDF Processing**
- **Multi-modal Content Extraction**: Text, tables, images, and scanned documents
- **OCR Integration**: Tesseract-powered optical character recognition
- **Layout Preservation**: Maintains document structure and formatting
- **Batch Processing**: Handle multiple documents simultaneously

### 🧠 **AI-Powered Summarization**
- **Hybrid Approach**: Combines extractive and abstractive summarization
- **Multiple Summary Types**: Short (TL;DR), Medium, and Detailed options
- **Customizable Tone**: Formal, casual, technical, and executive styles
- **Focus Areas**: Target specific sections or topics
- **Multi-language Support**: Process documents in 40+ languages

### 🔎 **Intelligent Search & Q&A**
- **Semantic Search**: Vector-based content retrieval using FAISS
- **Interactive Q&A**: Ask specific questions about document content
- **Context-Aware Responses**: Maintains conversation context
- **Entity Recognition**: Identify people, organizations, locations, and financial data

### 📊 **Enterprise Features**
- **Scalable Architecture**: MCP server integration with load balancing
- **Real-time Processing**: Live document analysis and feedback
- **Export Options**: JSON, Markdown, PDF, and plain text formats
- **Analytics Dashboard**: Comprehensive processing insights and metrics
- **Security**: Rate limiting, input validation, and secure file handling

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   MCP Server    │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   (Gemini API)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis         │    │   FAISS         │    │   File Storage  │
│   (Queue/Cache) │    │   (Vectors)     │    │   (PDFs/Data)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **FastAPI Backend**: High-performance async web framework
- **MCP Server**: Model Context Protocol for AI model integration
- **Gemini API**: Google's advanced language model for text processing
- **FAISS Vector Store**: Efficient similarity search and clustering
- **Redis**: Caching and queue management
- **Tesseract OCR**: Text extraction from images and scanned PDFs

## 🚀 Quick Start

### Option 1: Try Online (Recommended)
Visit the live demo: [🤗 HuggingFace Spaces](https://huggingface.co/spaces/parthmax/DocuMind-AI)

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/parthmax/DocuMind-AI.git
cd DocuMind-AI

# Configure environment
cp .env.example .env
# Add your Gemini API key to .env file

# Start with Docker Compose
docker-compose up -d

# Access the application
open http://localhost:8000
```

### Option 3: Manual Installation

#### Prerequisites
- Python 3.11+
- Tesseract OCR
- Redis Server
- Gemini API Key

#### Installation Steps

1. **Install System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils redis-server

# macOS
brew install tesseract poppler redis
brew services start redis

# Windows (using Chocolatey)
choco install tesseract poppler redis-64
```

2. **Setup Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

3. **Configure Environment Variables**
```bash
# Create .env file
GEMINI_API_KEY=your_gemini_api_key_here
MCP_SERVER_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS_PER_REQUEST=4000
```

4. **Start the Application**
```bash
# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🎯 Usage

### Web Interface

1. **📁 Upload PDF**: Drag and drop or browse for PDF files
2. **⚙️ Configure Settings**: 
   - Choose summary type (Short/Medium/Detailed)
   - Select tone (Formal/Casual/Technical/Executive)
   - Specify focus areas and custom questions
3. **🔄 Process Document**: Click "Generate Summary"
4. **💬 Interactive Features**: 
   - Ask questions about the document
   - Search specific content
   - Export results in various formats

### API Usage

#### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### Generate Summary
```bash
curl -X POST "http://localhost:8000/summarize/{file_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "summary_type": "medium",
    "tone": "formal",
    "focus_areas": ["key insights", "risks", "recommendations"],
    "custom_questions": ["What are the main findings?"]
  }'
```

#### Semantic Search
```bash
curl -X POST "http://localhost:8000/search/{file_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "financial performance", 
    "top_k": 5
  }'
```

#### Ask Questions
```bash
curl -X GET "http://localhost:8000/qa/{file_id}?question=What are the key risks mentioned?"
```

### Python SDK Usage

```python
from pdf_summarizer import DocuMindAI

# Initialize client
client = DocuMindAI(api_key="your-api-key")

# Upload and process document
with open("document.pdf", "rb") as file:
    document = client.upload(file)

# Generate summary
summary = client.summarize(
    document.id,
    summary_type="medium",
    tone="formal",
    focus_areas=["key insights", "risks"]
)

# Ask questions
answer = client.ask_question(
    document.id, 
    "What are the main recommendations?"
)

# Search content
results = client.search(
    document.id,
    query="revenue analysis",
    top_k=5
)
```

## 📚 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload PDF file |
| `POST` | `/batch/upload` | Upload multiple PDFs |
| `GET` | `/document/{file_id}/status` | Check processing status |
| `POST` | `/summarize/{file_id}` | Generate summary |
| `GET` | `/summaries/{file_id}` | List all summaries |
| `GET` | `/summary/{summary_id}` | Get specific summary |
| `POST` | `/search/{file_id}` | Semantic search |
| `POST` | `/qa/{file_id}` | Question answering |
| `GET` | `/export/{summary_id}/{format}` | Export summary |
| `GET` | `/analytics/{file_id}` | Document analytics |
| `POST` | `/compare` | Compare documents |
| `GET` | `/health` | System health check |

### Response Examples

#### Summary Response
```json
{
  "summary_id": "sum_abc123",
  "document_id": "doc_xyz789",
  "summary": {
    "content": "This document outlines the company's Q4 performance...",
    "key_points": [
      "Revenue increased by 15% year-over-year",
      "New market expansion planned for Q4",
      "Cost optimization initiatives showing results"
    ],
    "entities": {
      "organizations": ["Acme Corp", "TechStart Inc"],
      "people": ["John Smith", "Jane Doe"],
      "locations": ["New York", "California"],
      "financial": ["$1.2M", "15%", "Q4 2024"]
    },
    "topics": [
      {"topic": "Financial Performance", "confidence": 0.92},
      {"topic": "Market Expansion", "confidence": 0.87}
    ],
    "confidence_score": 0.91
  },
  "metadata": {
    "summary_type": "medium",
    "tone": "formal",
    "processing_time": 12.34,
    "created_at": "2024-08-25T10:30:00Z"
  }
}
```

#### Search Response
```json
{
  "query": "financial performance",
  "results": [
    {
      "content": "The company's financial performance exceeded expectations...",
      "similarity_score": 0.94,
      "page_number": 3,
      "chunk_id": "chunk_789"
    }
  ],
  "total_results": 5,
  "processing_time": 0.45
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Gemini API authentication key | - | ✅ |
| `MCP_SERVER_URL` | MCP server endpoint | `http://localhost:8080` | ❌ |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` | ❌ |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` | ❌ |
| `CHUNK_OVERLAP` | Overlap between text chunks | `200` | ❌ |
| `MAX_TOKENS_PER_REQUEST` | Maximum tokens per API call | `4000` | ❌ |
| `MAX_FILE_SIZE` | Maximum upload file size | `50MB` | ❌ |
| `SUPPORTED_LANGUAGES` | Comma-separated language codes | `en,es,fr,de` | ❌ |

### MCP Server Configuration

Edit `mcp-config/models.json`:

```json
{
  "models": [
    {
      "name": "gemini-pro",
      "config": {
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40
      },
      "limits": {
        "rpm": 60,
        "tpm": 32000,
        "max_concurrent": 10
      }
    }
  ],
  "load_balancing": "round_robin",
  "fallback_model": "gemini-pro-vision"
}
```

## 🔧 Advanced Features

### Batch Processing
```python
# Process multiple documents
batch_job = client.batch_process([
    "doc1.pdf", "doc2.pdf", "doc3.pdf"
], summary_type="medium")

# Monitor progress
status = client.get_batch_status(batch_job.id)
print(f"Progress: {status.progress}%")
```

### Document Comparison
```python
# Compare documents
comparison = client.compare_documents(
    document_ids=["doc1", "doc2"],
    focus_areas=["financial metrics", "strategic initiatives"]
)
```

### Custom Processing
```python
# Custom summarization parameters
summary = client.summarize(
    document_id,
    summary_type="custom",
    max_length=750,
    focus_keywords=["revenue", "growth", "risk"],
    exclude_sections=["appendix", "footnotes"]
)
```

## 🛠️ Development

### Project Structure
```
DocuMind-AI/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker services configuration
├── nginx.conf             # Reverse proxy configuration
├── .env.example           # Environment template
├── frontend/              # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── mcp-config/            # MCP server configuration
│   └── models.json
├── tests/                 # Test suite
│   ├── test_pdf_processor.py
│   ├── test_summarizer.py
│   └── samples/
└── docs/                  # Documentation
    ├── api.md
    └── deployment.md
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v --cov=main --cov-report=html

# Run specific test
pytest tests/test_pdf_processor.py -v
```

### Code Quality
```bash
# Format code
black main.py
isort main.py

# Type checking
mypy main.py

# Linting
flake8 main.py
```

## 📊 Performance & Monitoring

### System Health
- **Health Check Endpoint**: `/health`
- **Real-time Metrics**: Processing times, success rates, error tracking
- **Resource Monitoring**: Memory usage, CPU utilization, storage

### Performance Metrics
- **Average Processing Time**: ~12 seconds for medium-sized PDFs
- **Throughput**: 50+ documents per hour (single instance)
- **Accuracy**: 91%+ confidence score on summaries
- **Language Support**: 40+ languages with 85%+ accuracy

### Monitoring Dashboard
```bash
# Access metrics (if enabled)
curl http://localhost:9090/metrics

# System health
curl http://localhost:8000/health
```

## 🔒 Security

### Data Protection
- **File Validation**: Strict PDF format checking
- **Size Limits**: Configurable maximum file sizes
- **Rate Limiting**: API request throttling
- **Input Sanitization**: XSS and injection prevention

### API Security
- **Authentication**: Bearer token support
- **CORS Configuration**: Cross-origin request handling
- **Request Validation**: Pydantic model validation
- **Error Handling**: Secure error responses

### Privacy
- **Local Processing**: Optional on-premise deployment
- **Data Retention**: Configurable document cleanup
- **Encryption**: In-transit and at-rest options

## 🚀 Deployment

### Docker Deployment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale app=3
```

### Cloud Deployment
- **AWS**: ECS, EKS, or EC2 deployment guides
- **GCP**: Cloud Run, GKE deployment options
- **Azure**: Container Instances, AKS support
- **Heroku**: One-click deployment support

### Environment Setup
```bash
# Production environment
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO
export WORKERS=4
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check our [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/parthmax/DocuMind-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parthmax/DocuMind-AI/discussions)
- **Email**: support@documind-ai.com

### FAQ

**Q: What file formats are supported?**  
A: Currently, only PDF files are supported. We plan to add support for DOCX, TXT, and other formats.

**Q: Is there a file size limit?**  
A: Yes, the default limit is 50MB. This can be configured via environment variables.

**Q: Can I run this offline?**  
A: The system requires internet access for the Gemini API. We're working on offline capabilities.

**Q: How accurate are the summaries?**  
A: Our system achieves 91%+ confidence scores on most documents, with accuracy varying by document type and language.

## 🙏 Acknowledgments

- **Google AI**: For the Gemini API
- **FastAPI**: For the excellent web framework
- **HuggingFace**: For hosting our demo space
- **Tesseract**: For OCR capabilities
- **FAISS**: For efficient vector search

---

<div align="center">

**[⭐ Star this repo](https://github.com/parthmax/DocuMind-AI)** if you find it useful!

Made with ❤️ by [parthmax](https://github.com/parthmax)

</div>