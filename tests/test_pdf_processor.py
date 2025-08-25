# tests/test_pdf_processor.py
import pytest
import tempfile
import os
from pathlib import Path
import asyncio
from app import PDFProcessor, GeminiSummarizer, SummaryRequest

class TestPDFProcessor:
    """Test suite for PDF processing functionality"""
    
    @pytest.fixture
    async def pdf_processor(self):
        return PDFProcessor()
    
    @pytest.fixture
    def sample_pdf_path(self):
        # This would be a path to a test PDF file
        return "tests/samples/test_document.pdf"
    
    @pytest.mark.asyncio
    async def test_pdf_processing(self, pdf_processor, sample_pdf_path):
        """Test basic PDF processing"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("Sample PDF not found")
        
        chunks, metadata = await pdf_processor.process_pdf(sample_pdf_path)
        
        assert len(chunks) > 0
        assert "file_name" in metadata
        assert "page_count" in metadata
        assert metadata["total_chunks"] == len(chunks)
    
    @pytest.mark.asyncio
    async def test_text_chunking(self, pdf_processor):
        """Test text chunking functionality"""
        test_text = "This is a test document. " * 200  # Long text
        chunks = pdf_processor._split_text_into_chunks(test_text, 1, "Test Section")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(chunk.section == "Test Section" for chunk in chunks)
        assert all(chunk.page_number == 1 for chunk in chunks)
    
    def test_table_to_text_conversion(self, pdf_processor):
        """Test table to text conversion"""
        import pandas as pd
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })
        
        text = pdf_processor._table_to_text(df)
        
        assert "Name | Age | City" in text
        assert "Alice | 25 | New York" in text
        assert len(text.split('\n')) >= 4  # Headers + 3 rows

class TestGeminiSummarizer:
    """Test suite for Gemini summarization"""
    
    @pytest.fixture
    def summarizer(self):
        return GeminiSummarizer("test-api-key")
    
    def test_prompt_creation(self, summarizer):
        """Test prompt creation for different request types"""
        from app import DocumentChunk, SummaryRequest
        
        chunk = DocumentChunk(
            id="test-chunk",
            content="This is test content for summarization.",
            page_number=1,
            section="Test Section",
            chunk_type="text"
        )
        
        request = SummaryRequest(
            summary_type="medium",
            tone="formal",
            focus_areas=["key insights"],
            custom_questions=["What are the main points?"]
        )
        
        prompt = summarizer._create_chunk_prompt(chunk, request)
        
        assert "This is test content for summarization." in prompt
        assert "formal" in prompt.lower()
        assert "key insights" in prompt
        assert "What are the main points?" in prompt

class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
    
    def test_upload_validation(self, client):
        """Test file upload validation"""
        # Test non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"This is not a PDF")
            tmp.seek(0)
            
            response = client.post(
                "/upload",
                files={"file": ("test.txt", tmp, "text/plain")}
            )
            
            assert response.status_code == 400
            assert "PDF files" in response.json()["detail"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])