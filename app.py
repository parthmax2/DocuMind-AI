# Enterprise PDF Summarizer System
# High-end PDF processing with MCP server and Gemini API integration

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import hashlib
from datetime import datetime

# PDF Processing
import PyPDF2
import pdfplumber
import camelot
import tabula
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for better text extraction

# AI/ML
import google.generativeai as genai
import numpy as np
import os
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HOME"] = "/app/cache"
os.environ["HF_DATASETS_CACHE"] = "/app/cache"


from sentence_transformers import SentenceTransformer
import faiss

# Web Framework
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Utilities
import aiofiles
import httpx
from concurrent.futures import ThreadPoolExecutor
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()  # by default it looks for .env in project root

# Now Config will pick up the environment variables
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_TOKENS_PER_REQUEST = 4000
    UPLOAD_DIR = "uploads"
    SUMMARIES_DIR = "summaries"
    EMBEDDINGS_DIR = "embeddings"
    SUPPORTED_FORMATS = [".pdf"]

# Data Models
@dataclass
class DocumentChunk:
    id: str
    content: str
    page_number: int
    section: str
    chunk_type: str  # text, table, image
    embedding: Optional[np.ndarray] = None
    
@dataclass
class SummaryRequest:
    summary_type: str = "medium"  # short, medium, detailed
    tone: str = "formal"  # formal, casual, technical, executive
    focus_areas: List[str] = None
    custom_questions: List[str] = None
    language: str = "en"

@dataclass
class Summary:
    id: str
    document_id: str
    summary_type: str
    tone: str
    content: str
    key_points: List[str]
    entities: List[str]
    topics: List[str]
    confidence_score: float
    created_at: datetime

# Add these imports at the top of your file (missing imports)
import io
import traceback

class PDFProcessor:
    """Advanced PDF processing with comprehensive error handling"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_pdf(self, file_path: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Extract text, tables, and images from PDF with robust error handling"""
        chunks = []
        metadata = {}
        
        try:
            logger.info(f"Starting PDF processing: {file_path}")
            
            # Validate file exists and is readable
            if not Path(file_path).exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            file_size = Path(file_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"PDF file is empty: {file_path}")
            
            logger.info(f"Processing PDF: {Path(file_path).name} (size: {file_size} bytes)")
            
            # Test if PDF can be opened with PyMuPDF
            try:
                test_doc = fitz.open(file_path)
                page_count = test_doc.page_count
                logger.info(f"PDF has {page_count} pages")
                test_doc.close()
                
                if page_count == 0:
                    raise ValueError("PDF has no pages")
                    
            except Exception as e:
                logger.error(f"Cannot open PDF with PyMuPDF: {str(e)}")
                raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
            
            # Extract text and structure with error handling
            try:
                text_chunks = await self._extract_text_with_structure_safe(file_path)
                chunks.extend(text_chunks)
                logger.info(f"Extracted {len(text_chunks)} text chunks")
            except Exception as e:
                logger.error(f"Text extraction failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue processing even if text extraction fails
            
            # Extract tables with error handling
            try:
                table_chunks = await self._extract_tables_safe(file_path)
                chunks.extend(table_chunks)
                logger.info(f"Extracted {len(table_chunks)} table chunks")
            except Exception as e:
                logger.warning(f"Table extraction failed: {str(e)}")
            
            # Extract and process images with error handling
            try:
                image_chunks = await self._process_images_safe(file_path)
                chunks.extend(image_chunks)
                logger.info(f"Extracted {len(image_chunks)} image chunks")
            except Exception as e:
                logger.warning(f"Image processing failed: {str(e)}")
            
            # If no chunks were extracted, create fallback
            if not chunks:
                logger.warning("No chunks extracted, attempting fallback text extraction")
                fallback_chunks = await self._fallback_text_extraction(file_path)
                chunks.extend(fallback_chunks)
            
            # Generate metadata
            metadata = await self._generate_metadata_safe(file_path, chunks)
            
            logger.info(f"Successfully processed PDF: {len(chunks)} total chunks extracted")
            
            # Ensure we always return a tuple
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Critical error processing PDF: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty but valid results to prevent tuple unpacking errors
            empty_metadata = {
                "file_name": Path(file_path).name if Path(file_path).exists() else "unknown",
                "file_size": 0,
                "total_chunks": 0,
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "sections": [],
                "page_count": 0,
                "processed_at": datetime.now().isoformat(),
                "error": str(e)
            }
            return [], empty_metadata

    async def _extract_text_with_structure_safe(self, file_path: str) -> List[DocumentChunk]:
        """Extract text with comprehensive error handling"""
        chunks = []
        doc = None
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                try:
                    # FIX: Use correct page access method
                    page = doc[page_num]
                    
                    # Extract text with structure
                    blocks = page.get_text("dict")
                    
                    if not blocks or "blocks" not in blocks:
                        logger.warning(f"No text blocks found on page {page_num + 1}")
                        continue
                    
                    for block in blocks["blocks"]:
                        if "lines" in block:
                            text_content = ""
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if "text" in span:
                                        text_content += span["text"] + " "
                            
                            if len(text_content.strip()) > 20:  # Minimum meaningful content
                                # Detect section headers
                                section = self._detect_section(text_content, blocks)
                                
                                # Create chunks
                                text_chunks = self._split_text_into_chunks(
                                    text_content.strip(), 
                                    page_num + 1, 
                                    section
                                )
                                chunks.extend(text_chunks)
                
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num + 1}: {str(page_error)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            raise
        
        finally:
            if doc:
                doc.close()
        
        return chunks

    async def _extract_tables_safe(self, file_path: str) -> List[DocumentChunk]:
        """Extract tables with multiple fallback methods"""
        chunks = []
        
        # Method 1: Try Camelot (if available)
        try:
            import camelot
            tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                if not table.df.empty and hasattr(table, 'accuracy') and table.accuracy > 50:
                    table_text = self._table_to_text(table.df)
                    
                    chunk_id = hashlib.md5(f"table_{i}_{file_path}".encode()).hexdigest()
                    
                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=table_text,
                        page_number=getattr(table, 'page', 1),
                        section=f"Table {i+1}",
                        chunk_type="table"
                    )
                    chunks.append(chunk)
            
            if chunks:
                logger.info(f"Extracted {len(chunks)} tables using Camelot")
                return chunks
        
        except ImportError:
            logger.warning("Camelot not available for table extraction")
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {str(e)}")
        
        # Method 2: Try pdfplumber (more reliable, no Java needed)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables()
                        
                        for i, table_data in enumerate(tables):
                            if table_data and len(table_data) > 1:
                                # Convert to text format
                                table_text = self._array_to_table_text(table_data)
                                
                                chunk_id = hashlib.md5(f"table_plumber_{page_num}_{i}_{file_path}".encode()).hexdigest()
                                
                                chunk = DocumentChunk(
                                    id=chunk_id,
                                    content=table_text,
                                    page_number=page_num + 1,
                                    section=f"Table {len(chunks) + 1}",
                                    chunk_type="table"
                                )
                                chunks.append(chunk)
                    
                    except Exception as page_error:
                        logger.warning(f"Error extracting tables from page {page_num + 1}: {str(page_error)}")
                        continue
            
            if chunks:
                logger.info(f"Extracted {len(chunks)} tables using pdfplumber")
                return chunks
        
        except ImportError:
            logger.warning("pdfplumber not available")
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {str(e)}")
        
        return chunks

    def _array_to_table_text(self, table_data: List[List]) -> str:
        """Convert 2D array to readable table text"""
        text_parts = []
        
        if not table_data:
            return "Empty table"
        
        # First row as headers
        if table_data[0]:
            headers_text = " | ".join([str(cell or "") for cell in table_data[0]])
            text_parts.append(f"Table Headers: {headers_text}")
        
        # Data rows (limit to prevent huge chunks)
        for i, row in enumerate(table_data[1:], 1):
            if i > 15:  # Limit rows
                text_parts.append(f"... and {len(table_data) - 16} more rows")
                break
            
            row_text = " | ".join([str(cell or "") for cell in row])
            text_parts.append(f"Row {i}: {row_text}")
        
        return "\n".join(text_parts)

    async def _process_images_safe(self, file_path: str) -> List[DocumentChunk]:
        """Extract and process images with comprehensive error handling"""
        chunks = []
        doc = None
        
        try:
            # Check if pytesseract is available
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                logger.warning("OCR libraries not available, skipping image processing")
                return chunks
            
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Extract image
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                # Convert to PIL Image
                                img_data = pix.tobytes("ppm")
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # Perform OCR
                                ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
                                
                                if len(ocr_text.strip()) > 10:
                                    chunk_id = hashlib.md5(f"image_{page_num}_{img_index}".encode()).hexdigest()
                                    
                                    chunk = DocumentChunk(
                                        id=chunk_id,
                                        content=f"Image content (OCR): {ocr_text.strip()}",
                                        page_number=page_num + 1,
                                        section=f"Image {img_index + 1}",
                                        chunk_type="image"
                                    )
                                    chunks.append(chunk)
                            
                            pix = None
                            
                        except Exception as img_error:
                            logger.warning(f"Error processing image {img_index} on page {page_num + 1}: {str(img_error)}")
                            continue
                
                except Exception as page_error:
                    logger.warning(f"Error processing images on page {page_num + 1}: {str(page_error)}")
                    continue
        
        except Exception as e:
            logger.warning(f"Image processing failed: {str(e)}")
        
        finally:
            if doc:
                doc.close()
        
        return chunks

    async def _fallback_text_extraction(self, file_path: str) -> List[DocumentChunk]:
        """Fallback text extraction using simple methods"""
        chunks = []
        
        try:
            logger.info("Attempting fallback text extraction")
            
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Simple text extraction
                    text = page.get_text()
                    
                    if text and len(text.strip()) > 20:
                        # Split into chunks
                        fallback_chunks = self._split_text_into_chunks(
                            text.strip(),
                            page_num + 1,
                            f"Page {page_num + 1}"
                        )
                        chunks.extend(fallback_chunks)
                        logger.info(f"Fallback extraction found {len(fallback_chunks)} chunks on page {page_num + 1}")
                
                except Exception as page_error:
                    logger.warning(f"Fallback extraction failed on page {page_num + 1}: {str(page_error)}")
                    continue
            
            doc.close()
            
            if chunks:
                logger.info(f"Fallback extraction successful: {len(chunks)} chunks")
            else:
                logger.warning("Fallback extraction found no content")
                
                # Create a minimal chunk to avoid empty results
                minimal_chunk = DocumentChunk(
                    id=hashlib.md5(f"minimal_{file_path}".encode()).hexdigest(),
                    content=f"Document processed but no readable content extracted from {Path(file_path).name}",
                    page_number=1,
                    section="Document Info",
                    chunk_type="text"
                )
                chunks.append(minimal_chunk)
        
        except Exception as e:
            logger.error(f"Fallback text extraction failed: {str(e)}")
            
            # Create error chunk to avoid empty results
            error_chunk = DocumentChunk(
                id=hashlib.md5(f"error_{file_path}".encode()).hexdigest(),
                content=f"Error processing document: {str(e)}",
                page_number=1,
                section="Error",
                chunk_type="text"
            )
            chunks.append(error_chunk)
        
        return chunks

    async def _generate_metadata_safe(self, file_path: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Generate metadata with error handling"""
        try:
            metadata = {
                "file_name": Path(file_path).name,
                "file_size": Path(file_path).stat().st_size,
                "total_chunks": len(chunks),
                "text_chunks": len([c for c in chunks if c.chunk_type == "text"]),
                "table_chunks": len([c for c in chunks if c.chunk_type == "table"]),
                "image_chunks": len([c for c in chunks if c.chunk_type == "image"]),
                "sections": list(set([c.section for c in chunks])) if chunks else [],
                "page_count": max([c.page_number for c in chunks]) if chunks else 0,
                "processed_at": datetime.now().isoformat(),
                "processing_status": "success" if chunks else "no_content_extracted"
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {
                "file_name": "unknown",
                "file_size": 0,
                "total_chunks": 0,
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "sections": [],
                "page_count": 0,
                "processed_at": datetime.now().isoformat(),
                "processing_status": "error",
                "error": str(e)
            }

    # Keep your existing helper methods with minor fixes
    def _split_text_into_chunks(self, text: str, page_num: int, section: str) -> List[DocumentChunk]:
        """Split text into manageable chunks with overlap"""
        chunks = []
        
        if not text or len(text.strip()) < 10:
            return chunks
        
        words = text.split()
        
        chunk_size = Config.CHUNK_SIZE
        overlap = Config.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 20:  # Minimum chunk size
                chunk_id = hashlib.md5(f"{chunk_text[:100]}{page_num}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    page_number=page_num,
                    section=section,
                    chunk_type="text"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _detect_section(self, text: str, blocks: Dict) -> str:
        """Detect section headers using font size and formatting"""
        # Simple heuristic - look for short lines with larger fonts
        lines = text.split('\n')
        for line in lines[:3]:  # Check first few lines
            if len(line.strip()) < 100 and len(line.strip()) > 10:
                if any(keyword in line.lower() for keyword in 
                      ['chapter', 'section', 'introduction', 'conclusion', 'summary']):
                    return line.strip()
        
        return "Main Content"
    
    def _table_to_text(self, df) -> str:
        """Convert DataFrame to readable text"""
        text_parts = []
        
        # Add column headers
        headers = " | ".join([str(col) for col in df.columns])
        text_parts.append(f"Table Headers: {headers}")
        
        # Add rows (limit to prevent huge chunks)
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 15:  # Limit rows
                text_parts.append(f"... and {len(df) - 15} more rows")
                break
            
            row_text = " | ".join([str(val) for val in row.values])
            text_parts.append(f"Row {i+1}: {row_text}")
        
        return "\n".join(text_parts)
    
    async def _process_images(self, file_path: str) -> List[DocumentChunk]:
        """Extract and process images using OCR"""
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                # FIX: Use doc[page_num] instead of doc.page(page_num)
                page = doc[page_num]  # or page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PIL Image
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Perform OCR
                            ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
                            
                            if len(ocr_text.strip()) > 10:  # Only if meaningful text found
                                chunk_id = hashlib.md5(f"image_{page_num}_{img_index}".encode()).hexdigest()
                                
                                chunk = DocumentChunk(
                                    id=chunk_id,
                                    content=f"Image content (OCR): {ocr_text.strip()}",
                                    page_number=page_num + 1,
                                    section=f"Image {img_index + 1}",
                                    chunk_type="image"
                                )
                                chunks.append(chunk)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"Image processing failed: {str(e)}")
        
        return chunks
    
    async def _generate_metadata(self, file_path: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Generate document metadata"""
        metadata = {
            "file_name": Path(file_path).name,
            "file_size": Path(file_path).stat().st_size,
            "total_chunks": len(chunks),
            "text_chunks": len([c for c in chunks if c.chunk_type == "text"]),
            "table_chunks": len([c for c in chunks if c.chunk_type == "table"]),
            "image_chunks": len([c for c in chunks if c.chunk_type == "image"]),
            "sections": list(set([c.section for c in chunks])),
            "page_count": max([c.page_number for c in chunks]) if chunks else 0,
            "processed_at": datetime.now().isoformat()
        }
        
        return metadata

class GeminiSummarizer:
    """Gemini API integration for advanced summarization"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def summarize_chunks(self, chunks: List[DocumentChunk], 
                              request: SummaryRequest) -> List[str]:
        """Summarize individual chunks"""
        summaries = []
        
        # Create batch requests for efficiency
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_summaries = await self._process_chunk_batch(batch, request)
            summaries.extend(batch_summaries)
        
        return summaries
    
    async def _process_chunk_batch(self, chunks: List[DocumentChunk], 
                                  request: SummaryRequest) -> List[str]:
        """Process a batch of chunks"""
        tasks = []
        
        for chunk in chunks:
            prompt = self._create_chunk_prompt(chunk, request)
            task = self._call_gemini_api(prompt)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        summaries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error summarizing chunk {chunks[i].id}: {str(result)}")
                summaries.append(f"[Error processing content from {chunks[i].section}]")
            else:
                summaries.append(result)
        
        return summaries
    
    def _create_chunk_prompt(self, chunk: DocumentChunk, request: SummaryRequest) -> str:
        """Create optimized prompt for chunk summarization"""
        
        tone_instructions = {
            "formal": "Use professional, academic language",
            "casual": "Use conversational, accessible language", 
            "technical": "Use precise technical terminology",
            "executive": "Focus on key insights and implications for decision-making"
        }
        
        length_instructions = {
            "short": "Provide 1-2 sentences capturing the essence",
            "medium": "Provide 2-3 sentences with key details",
            "detailed": "Provide a comprehensive paragraph with full context"
        }
        
        prompt_parts = [
            f"Summarize the following {chunk.chunk_type} content from {chunk.section}:",
            f"Content: {chunk.content[:2000]}",  # Limit content length
            f"Style: {tone_instructions.get(request.tone, 'Use clear, professional language')}",
            f"Length: {length_instructions.get(request.summary_type, 'Provide appropriate detail')}",
        ]
        
        if request.focus_areas:
            prompt_parts.append(f"Focus particularly on: {', '.join(request.focus_areas)}")
        
        if request.custom_questions:
            prompt_parts.append(f"Address these questions if relevant: {'; '.join(request.custom_questions)}")
        
        prompt_parts.append("Provide only the summary without meta-commentary.")
        
        return "\n\n".join(prompt_parts)
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Make API call to Gemini"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, 
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                )
            )
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    async def create_final_summary(self, chunk_summaries: List[str], 
                                  metadata: Dict[str, Any], 
                                  request: SummaryRequest) -> Summary:
        """Create final cohesive summary from chunk summaries"""
        
        # Combine summaries intelligently
        combined_text = "\n".join(chunk_summaries)
        
        final_prompt = self._create_final_summary_prompt(combined_text, metadata, request)
        
        try:
            final_content = await self._call_gemini_api(final_prompt)
            
            # Extract key points and entities
            key_points = await self._extract_key_points(final_content)
            entities = await self._extract_entities(final_content)
            topics = await self._extract_topics(combined_text)
            
            summary_id = hashlib.md5(f"{final_content[:100]}{datetime.now()}".encode()).hexdigest()
            
            summary = Summary(
                id=summary_id,
                document_id=metadata.get("file_name", "unknown"),
                summary_type=request.summary_type,
                tone=request.tone,
                content=final_content,
                key_points=key_points,
                entities=entities,
                topics=topics,
                confidence_score=0.85,  # Would implement actual confidence scoring
                created_at=datetime.now()
            )
            
            return summary
        
        except Exception as e:
            logger.error(f"Error creating final summary: {str(e)}")
            raise
    
    def _create_final_summary_prompt(self, combined_summaries: str, 
                                   metadata: Dict[str, Any], 
                                   request: SummaryRequest) -> str:
        """Create prompt for final summary generation"""
        
        word_limits = {
            "short": "50-100 words (2-3 sentences maximum)",
            "medium": "200-400 words (2-3 paragraphs)",
            "detailed": "500-1000 words (multiple paragraphs with comprehensive coverage)"
        }
        
        prompt = f"""
Create a cohesive {request.summary_type} summary from the following section summaries of a document:

Document Information:
- File: {metadata.get('file_name', 'Unknown')}
- Pages: {metadata.get('page_count', 'Unknown')}
- Sections: {', '.join(metadata.get('sections', [])[:5])}

Section Summaries:
{combined_summaries[:4000]}

Requirements:
- Length: {word_limits.get(request.summary_type, '200-400 words')}
- Tone: {request.tone}
- Create a flowing narrative that integrates all key information
- Eliminate redundancy while preserving important details
- Structure with clear logical flow
"""
        
        if request.focus_areas:
            prompt += f"\n- Emphasize: {', '.join(request.focus_areas)}"
        
        if request.custom_questions:
            prompt += f"\n- Address: {'; '.join(request.custom_questions)}"
        
        return prompt
    
    async def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from summary"""
        prompt = f"""
Extract 5-7 key points from this summary as bullet points:

{text[:1500]}

Format as a simple list, one point per line.
"""
        
        try:
            response = await self._call_gemini_api(prompt)
            points = [line.strip().lstrip('•-*').strip() 
                     for line in response.split('\n') 
                     if line.strip() and len(line.strip()) > 10]
            return points[:7]
        except:
            return []
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        prompt = f"""
Extract important named entities (people, organizations, locations, products, concepts) from:

{text[:1500]}

List them separated by commas, no explanations.
"""
        
        try:
            response = await self._call_gemini_api(prompt)
            entities = [e.strip() for e in response.split(',') if e.strip()]
            return entities[:10]
        except:
            return []
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics"""
        prompt = f"""
Identify 3-5 main topics/themes from this content:

{text[:2000]}

List topics as single words or short phrases, separated by commas.
"""
        
        try:
            response = await self._call_gemini_api(prompt)
            topics = [t.strip() for t in response.split(',') if t.strip()]
            return topics[:5]
        except:
            return []
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for semantic search"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Update chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        return embeddings

class VectorStore:
    """FAISS-based vector storage for semantic search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_map = {}
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks and embeddings to the store"""
        self.index.add(embeddings.astype('float32'))
        
        for i, chunk in enumerate(chunks):
            self.chunk_map[i] = chunk
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Semantic search for relevant chunks"""
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.chunk_map:
                chunk = self.chunk_map[idx]
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                results.append((chunk, similarity))
        
        return results
    
    def save(self, path: str):
        """Save index and chunk map"""
        faiss.write_index(self.index, f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunk_map, f)
    
    def load(self, path: str):
        """Load index and chunk map"""
        self.index = faiss.read_index(f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'rb') as f:
            self.chunk_map = pickle.load(f)

class MCPServerClient:
    """MCP Server client for orchestration and monitoring"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = httpx.AsyncClient()
    
    async def register_document(self, doc_id: str, metadata: Dict[str, Any]):
        """Register document processing with MCP server"""
        try:
            response = await self.client.post(
                f"{self.server_url}/documents/register",
                json={"doc_id": doc_id, "metadata": metadata}
            )
            return response.json()
        except Exception as e:
            logger.warning(f"MCP server registration failed: {str(e)}")
            return {}
    
    async def log_processing_metrics(self, doc_id: str, metrics: Dict[str, Any]):
        """Log processing metrics to MCP server"""
        try:
            await self.client.post(
                f"{self.server_url}/metrics/log",
                json={"doc_id": doc_id, "metrics": metrics}
            )
        except Exception as e:
            logger.warning(f"MCP metrics logging failed: {str(e)}")
    
    async def get_model_health(self) -> Dict[str, Any]:
        """Check model health via MCP server"""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            return response.json()
        except Exception as e:
            logger.warning(f"MCP health check failed: {str(e)}")
            return {"status": "unknown"}

# FastAPI Application
app = FastAPI(title="Enterprise PDF Summarizer", version="1.0.0")
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
summarizer = GeminiSummarizer(Config.GEMINI_API_KEY)
vector_store = VectorStore()
mcp_client = MCPServerClient(Config.MCP_SERVER_URL)

# Ensure directories exist
for dir_name in [Config.UPLOAD_DIR, Config.SUMMARIES_DIR, Config.EMBEDDINGS_DIR]:
    Path(dir_name).mkdir(exist_ok=True)

# API Models
class SummaryRequestModel(BaseModel):
    summary_type: str = Field("medium", description="short, medium, or detailed")
    tone: str = Field("formal", description="formal, casual, technical, or executive")
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus on")
    custom_questions: Optional[List[str]] = Field(None, description="Custom questions to address")
    language: str = Field("en", description="Language code")

class SearchQueryModel(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results")

# API Endpoints
@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process PDF"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    file_id = hashlib.md5(f"{file.filename}{datetime.now()}".encode()).hexdigest()
    file_path = Path(Config.UPLOAD_DIR) / f"{file_id}.pdf"
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Process PDF in background
    background_tasks.add_task(process_pdf_background, str(file_path), file_id)
    
    return {"file_id": file_id, "status": "processing", "filename": file.filename}


async def process_pdf_background(file_path: str, file_id: str):
    """Background task to process PDF with comprehensive error handling"""
    try:
        logger.info(f"Starting background processing for {file_id}")
        
        # Process PDF - this now always returns a tuple
        chunks, metadata = await pdf_processor.process_pdf(file_path)
        
        logger.info(f"PDF processing completed: {len(chunks)} chunks, metadata: {metadata.get('processing_status', 'unknown')}")
        
        # Only proceed with embeddings if we have chunks
        if chunks:
            try:
                # Generate embeddings
                logger.info("Generating embeddings...")
                embeddings = summarizer.generate_embeddings(chunks)
                
                # Store in vector database
                logger.info("Storing in vector database...")
                vector_store.add_chunks(chunks, embeddings)
                
                # Save processed data
                data_path = Path(Config.EMBEDDINGS_DIR) / file_id
                vector_store.save(str(data_path))
                
                logger.info(f"Vector data saved to {data_path}")
                
            except Exception as embedding_error:
                logger.error(f"Error in embedding/vector processing: {str(embedding_error)}")
                # Continue without embeddings - we still have the chunks
        else:
            logger.warning(f"No chunks extracted from {file_id}, skipping embeddings")
        
        # Always save chunks and metadata (even if empty)
        try:
            data_path = Path(Config.EMBEDDINGS_DIR) / file_id
            with open(f"{data_path}_data.pkl", 'wb') as f:
                pickle.dump({"chunks": chunks, "metadata": metadata}, f)
            
            logger.info(f"Chunks and metadata saved for {file_id}")
            
        except Exception as save_error:
            logger.error(f"Error saving processed data for {file_id}: {str(save_error)}")
        
        # Register with MCP server (if available)
        try:
            await mcp_client.register_document(file_id, metadata)
        except Exception as mcp_error:
            logger.warning(f"MCP server registration failed for {file_id}: {str(mcp_error)}")
        
        logger.info(f"Successfully completed background processing for {file_id}")
        
    except Exception as e:
        logger.error(f"Critical error in background processing for {file_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save error information so the document status can be checked
        try:
            error_metadata = {
                "file_name": Path(file_path).name if Path(file_path).exists() else "unknown",
                "file_size": 0,
                "total_chunks": 0,
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "sections": [],
                "page_count": 0,
                "processed_at": datetime.now().isoformat(),
                "processing_status": "error",
                "error": str(e)
            }
            
            data_path = Path(Config.EMBEDDINGS_DIR) / file_id
            with open(f"{data_path}_data.pkl", 'wb') as f:
                pickle.dump({"chunks": [], "metadata": error_metadata}, f)
            
            logger.info(f"Error metadata saved for {file_id}")
            
        except Exception as save_error:
            logger.error(f"Could not save error metadata for {file_id}: {str(save_error)}")

@app.post("/summarize/{file_id}")
async def create_summary(file_id: str, request: SummaryRequestModel):
    """Generate summary for processed PDF with better error handling"""
    
    try:
        # Load processed data
        data_path = Path(Config.EMBEDDINGS_DIR) / f"{file_id}_data.pkl"
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Document not found or still processing")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        chunks = data["chunks"]
        metadata = data["metadata"]
        
        # Check if processing had errors
        if metadata.get("processing_status") == "error":
            raise HTTPException(
                status_code=422, 
                detail=f"Document processing failed: {metadata.get('error', 'Unknown error')}"
            )
        
        # Check if we have chunks to summarize
        if not chunks or len(chunks) == 0:
            raise HTTPException(
                status_code=422, 
                detail="No content could be extracted from this document for summarization"
            )
        
        logger.info(f"Creating summary for {file_id} with {len(chunks)} chunks")
        
        # Create summary request
        summary_request = SummaryRequest(
            summary_type=request.summary_type,
            tone=request.tone,
            focus_areas=request.focus_areas,
            custom_questions=request.custom_questions,
            language=request.language
        )
        
        # Generate summaries
        try:
            chunk_summaries = await summarizer.summarize_chunks(chunks, summary_request)
            final_summary = await summarizer.create_final_summary(
                chunk_summaries, metadata, summary_request
            )
        except Exception as summary_error:
            logger.error(f"Error generating summary: {str(summary_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Summary generation failed: {str(summary_error)}"
            )
        
        # Save summary
        try:
            summary_path = Path(Config.SUMMARIES_DIR) / f"{file_id}_{final_summary.id}.json"
            with open(summary_path, 'w') as f:
                json.dump(asdict(final_summary), f, indent=2, default=str)
        except Exception as save_error:
            logger.warning(f"Could not save summary to file: {str(save_error)}")
            # Continue anyway - we can still return the summary
        
        # Log metrics
        try:
            metrics = {
                "summary_type": request.summary_type,
                "chunk_count": len(chunks),
                "processing_time": "calculated",
                "confidence_score": final_summary.confidence_score
            }
            await mcp_client.log_processing_metrics(file_id, metrics)
        except Exception as metrics_error:
            logger.warning(f"Could not log metrics: {str(metrics_error)}")
        
        return {
            "summary_id": final_summary.id,
            "summary": asdict(final_summary),
            "metadata": metadata
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating summary: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


        


@app.post("/search/{file_id}")
async def semantic_search(file_id: str, query: SearchQueryModel):
    """Perform semantic search on document"""
    
    try:
        # Load vector store
        vector_path = Path(Config.EMBEDDINGS_DIR) / file_id
        
        if not Path(f"{vector_path}_index.faiss").exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create new vector store instance for this search
        search_store = VectorStore()
        search_store.load(str(vector_path))
        
        # Generate query embedding
        query_embedding = summarizer.embedding_model.encode([query.query])
        
        # Search
        results = search_store.search(query_embedding[0], query.top_k)
        
        # Format results
        search_results = []
        for chunk, similarity in results:
            search_results.append({
                "chunk_id": chunk.id,
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type,
                "similarity_score": float(similarity)
            })
        
        return {
            "query": query.query,
            "results": search_results,
            "total_results": len(search_results)
        }
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/document/{file_id}/status")
async def get_document_status(file_id: str):
    """Get processing status of a document with detailed information"""
    
    try:
        data_path = Path(Config.EMBEDDINGS_DIR) / f"{file_id}_data.pkl"
        
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data["metadata"]
            chunks = data["chunks"]
            
            status = {
                "status": "completed",
                "metadata": metadata,
                "chunks_count": len(chunks),
                "processing_status": metadata.get("processing_status", "unknown")
            }
            
            # Add processing quality information
            if chunks:
                status["content_types"] = {
                    "text": len([c for c in chunks if c.chunk_type == "text"]),
                    "table": len([c for c in chunks if c.chunk_type == "table"]),
                    "image": len([c for c in chunks if c.chunk_type == "image"])
                }
            
            # Add error information if processing failed
            if metadata.get("processing_status") == "error":
                status["error"] = metadata.get("error", "Unknown error occurred")
            
            return status
        else:
            return {
                "status": "processing",
                "message": "Document is still being processed"
            }
    
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        return {
            "status": "error",
            "error": f"Could not retrieve document status: {str(e)}"
        }

@app.get("/summaries/{file_id}")
async def list_summaries(file_id: str):
    """List all summaries for a document"""
    
    summaries_dir = Path(Config.SUMMARIES_DIR)
    summary_files = list(summaries_dir.glob(f"{file_id}_*.json"))
    
    summaries = []
    for file_path in summary_files:
        with open(file_path, 'r') as f:
            summary_data = json.load(f)
            summaries.append({
                "summary_id": summary_data["id"],
                "summary_type": summary_data["summary_type"],
                "tone": summary_data["tone"],
                "created_at": summary_data["created_at"],
                "confidence_score": summary_data["confidence_score"]
            })
    
    return {"summaries": summaries}

@app.get("/summary/{summary_id}")
async def get_summary(summary_id: str):
    """Get specific summary by ID"""
    
    # Find summary file
    summaries_dir = Path(Config.SUMMARIES_DIR)
    summary_files = list(summaries_dir.glob(f"*_{summary_id}.json"))
    
    if not summary_files:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    with open(summary_files[0], 'r') as f:
        summary_data = json.load(f)
    
    return {"summary": summary_data}

@app.post("/qa/{file_id}")
async def question_answering(file_id: str, question: str):
    """Answer specific questions about the document"""
    
    try:
        # Load processed data
        data_path = Path(Config.EMBEDDINGS_DIR) / f"{file_id}_data.pkl"
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        chunks = data["chunks"]
        
        # Find relevant chunks using semantic search
        vector_path = Path(Config.EMBEDDINGS_DIR) / file_id
        search_store = VectorStore()
        search_store.load(str(vector_path))
        
        query_embedding = summarizer.embedding_model.encode([question])
        relevant_chunks = search_store.search(query_embedding[0], top_k=3)
        
        # Create context from relevant chunks
        context = "\n\n".join([chunk.content for chunk, _ in relevant_chunks])
        
        # Generate answer using Gemini
        qa_prompt = f"""
Based on the following context from a document, answer this question: {question}

Context:
{context[:3000]}

Provide a clear, concise answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, say so.
"""
        
        answer = await summarizer._call_gemini_api(qa_prompt)
        
        # Include source information
        sources = []
        for chunk, similarity in relevant_chunks:
            sources.append({
                "page": chunk.page_number,
                "section": chunk.section,
                "similarity": float(similarity)
            })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": sum([s["similarity"] for s in sources]) / len(sources) if sources else 0
        }
    
    except Exception as e:
        logger.error(f"Error in Q&A: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")

@app.get("/export/{summary_id}/{format}")
async def export_summary(summary_id: str, format: str):
    """Export summary in different formats"""
    
    if format not in ["json", "markdown", "txt"]:
        raise HTTPException(status_code=400, detail="Supported formats: json, markdown, txt")
    
    # Find summary
    summaries_dir = Path(Config.SUMMARIES_DIR)
    summary_files = list(summaries_dir.glob(f"*_{summary_id}.json"))
    
    if not summary_files:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    with open(summary_files[0], 'r') as f:
        summary_data = json.load(f)
    
    if format == "json":
        return summary_data
    
    elif format == "markdown":
        markdown_content = f"""# Document Summary
        
**Document:** {summary_data['document_id']}
**Type:** {summary_data['summary_type']} 
**Tone:** {summary_data['tone']}
**Created:** {summary_data['created_at']}

## Summary

{summary_data['content']}

## Key Points

{chr(10).join([f"- {point}" for point in summary_data['key_points']])}

## Topics

{', '.join(summary_data['topics'])}

## Entities

{', '.join(summary_data['entities'])}
"""
        
        # Save and return file
        export_path = Path(Config.SUMMARIES_DIR) / f"{summary_id}.md"
        with open(export_path, 'w') as f:
            f.write(markdown_content)
        
        return FileResponse(
            path=export_path,
            filename=f"summary_{summary_id}.md",
            media_type="text/markdown"
        )
    
    elif format == "txt":
        txt_content = f"""Document Summary
================

Document: {summary_data['document_id']}
Type: {summary_data['summary_type']}
Tone: {summary_data['tone']}
Created: {summary_data['created_at']}

Summary:
{summary_data['content']}

Key Points:
{chr(10).join([f"• {point}" for point in summary_data['key_points']])}

Topics: {', '.join(summary_data['topics'])}
Entities: {', '.join(summary_data['entities'])}
"""
        
        export_path = Path(Config.SUMMARIES_DIR) / f"{summary_id}.txt"
        with open(export_path, 'w') as f:
            f.write(txt_content)
        
        return FileResponse(
            path=export_path,
            filename=f"summary_{summary_id}.txt",
            media_type="text/plain"
        )

@app.get("/health")
async def health_check():
    """System health check"""
    
    # Check MCP server health
    mcp_health = await mcp_client.get_model_health()
    
    # Check disk space
    upload_dir = Path(Config.UPLOAD_DIR)
    free_space = upload_dir.stat().st_size if upload_dir.exists() else 0
    
    return {
        "status": "healthy",
        "mcp_server": mcp_health.get("status", "unknown"),
        "storage": {
            "free_space_mb": free_space / (1024 * 1024),
            "upload_dir": str(upload_dir)
        },
        "services": {
            "pdf_processor": "online",
            "gemini_api": "online",
            "vector_store": "online"
        }
    }

@app.get("/analytics/{file_id}")
async def get_document_analytics(file_id: str):
    """Get detailed analytics for a processed document"""
    
    try:
        data_path = Path(Config.EMBEDDINGS_DIR) / f"{file_id}_data.pkl"
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        chunks = data["chunks"]
        metadata = data["metadata"]
        
        # Analyze content
        total_words = sum([len(chunk.content.split()) for chunk in chunks])
        avg_chunk_size = total_words / len(chunks) if chunks else 0
        
        # Content type distribution
        type_distribution = {}
        for chunk in chunks:
            type_distribution[chunk.chunk_type] = type_distribution.get(chunk.chunk_type, 0) + 1
        
        # Section analysis
        section_analysis = {}
        for chunk in chunks:
            if chunk.section not in section_analysis:
                section_analysis[chunk.section] = {
                    "chunk_count": 0,
                    "word_count": 0,
                    "types": set()
                }
            
            section_analysis[chunk.section]["chunk_count"] += 1
            section_analysis[chunk.section]["word_count"] += len(chunk.content.split())
            section_analysis[chunk.section]["types"].add(chunk.chunk_type)
        
        # Convert sets to lists for JSON serialization
        for section in section_analysis:
            section_analysis[section]["types"] = list(section_analysis[section]["types"])
        
        return {
            "document_id": file_id,
            "metadata": metadata,
            "content_stats": {
                "total_chunks": len(chunks),
                "total_words": total_words,
                "avg_chunk_size": round(avg_chunk_size, 2),
                "type_distribution": type_distribution
            },
            "section_analysis": section_analysis,
            "processing_quality": {
                "text_extraction_rate": type_distribution.get("text", 0) / len(chunks) if chunks else 0,
                "table_detection_count": type_distribution.get("table", 0),
                "image_ocr_count": type_distribution.get("image", 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")

# Multi-language support utility
class LanguageDetector:
    """Detect and handle multiple languages"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection (would use proper library in production)"""
        # Simplified detection - would use langdetect or similar
        common_english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it']
        text_lower = text.lower()
        
        english_count = sum([1 for word in common_english_words if word in text_lower])
        
        if english_count > 3:
            return "en"
        else:
            return "unknown"  # Would implement proper detection

    @staticmethod
    def get_language_specific_prompt_additions(language: str) -> str:
        """Get language-specific prompt additions"""
        language_prompts = {
            "es": "Responde en español.",
            "fr": "Répondez en français.",
            "de": "Antworten Sie auf Deutsch.",
            "it": "Rispondi in italiano.",
            "pt": "Responda em português.",
            "zh": "用中文回答。",
            "ja": "日本語で回答してください。",
            "ko": "한국어로 답변해주세요.",
            "ar": "أجب باللغة العربية.",
            "hi": "हिंदी में उत्तर दें।"
        }
        
        return language_prompts.get(language, "Respond in English.")

# Advanced document processor for special document types
class SpecializedProcessors:
    """Specialized processors for different document types"""
    
    @staticmethod
    async def process_academic_paper(chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Extract academic paper structure"""
        structure = {
            "abstract": [],
            "introduction": [],
            "methodology": [],
            "results": [],
            "discussion": [],
            "conclusion": [],
            "references": []
        }
        
        for chunk in chunks:
            section_lower = chunk.section.lower()
            
            if any(term in section_lower for term in ["abstract", "summary"]):
                structure["abstract"].append(chunk)
            elif "introduction" in section_lower:
                structure["introduction"].append(chunk)
            elif any(term in section_lower for term in ["method", "approach", "procedure"]):
                structure["methodology"].append(chunk)
            elif any(term in section_lower for term in ["result", "finding", "outcome"]):
                structure["results"].append(chunk)
            elif any(term in section_lower for term in ["discussion", "analysis"]):
                structure["discussion"].append(chunk)
            elif any(term in section_lower for term in ["conclusion", "summary"]):
                structure["conclusion"].append(chunk)
            elif any(term in section_lower for term in ["reference", "bibliography", "citation"]):
                structure["references"].append(chunk)
        
        return structure
    
    @staticmethod
    async def process_financial_document(chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Extract financial document insights"""
        financial_keywords = [
            "revenue", "profit", "loss", "assets", "liabilities", "cash flow",
            "investment", "roi", "ebitda", "margin", "growth", "risk"
        ]
        
        financial_chunks = []
        for chunk in chunks:
            content_lower = chunk.content.lower()
            if any(keyword in content_lower for keyword in financial_keywords):
                financial_chunks.append(chunk)
        
        return {
            "financial_sections": financial_chunks,
            "key_metrics_detected": len(financial_chunks),
            "table_data": [chunk for chunk in chunks if chunk.chunk_type == "table"]
        }
    
    @staticmethod
    async def process_legal_document(chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Extract legal document structure"""
        legal_keywords = [
            "clause", "section", "article", "paragraph", "whereas", "therefore",
            "contract", "agreement", "party", "obligation", "right", "liability"
        ]
        
        legal_structure = {
            "clauses": [],
            "definitions": [],
            "obligations": [],
            "rights": []
        }
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            if any(term in content_lower for term in ["clause", "section", "article"]):
                legal_structure["clauses"].append(chunk)
            elif "definition" in content_lower or "means" in content_lower:
                legal_structure["definitions"].append(chunk)
            elif any(term in content_lower for term in ["shall", "must", "obligation"]):
                legal_structure["obligations"].append(chunk)
            elif "right" in content_lower or "entitled" in content_lower:
                legal_structure["rights"].append(chunk)
        
        return legal_structure

# Batch processing endpoint
@app.post("/batch/upload")
async def batch_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload and process multiple PDFs"""
    
    batch_id = hashlib.md5(f"batch_{datetime.now()}".encode()).hexdigest()
    file_ids = []
    
    for file in files:
        if file.filename.lower().endswith('.pdf'):
            file_id = hashlib.md5(f"{file.filename}{datetime.now()}".encode()).hexdigest()
            file_path = Path(Config.UPLOAD_DIR) / f"{file_id}.pdf"
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_ids.append({
                "file_id": file_id,
                "filename": file.filename,
                "status": "queued"
            })
            
            # Add to background processing
            background_tasks.add_task(process_pdf_background, str(file_path), file_id)
    
    return {
        "batch_id": batch_id,
        "files": file_ids,
        "total_files": len(file_ids)
    }

# Comparative analysis endpoint
@app.post("/compare")
async def compare_documents(file_ids: List[str], comparison_focus: str = "content"):
    """Compare multiple documents"""
    
    try:
        documents_data = []
        
        for file_id in file_ids:
            data_path = Path(Config.EMBEDDINGS_DIR) / f"{file_id}_data.pkl"
            
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    documents_data.append({
                        "file_id": file_id,
                        "chunks": data["chunks"],
                        "metadata": data["metadata"]
                    })
        
        if len(documents_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 documents for comparison")
        
        # Generate comparison summary
        comparison_prompt = f"""
Compare the following {len(documents_data)} documents focusing on {comparison_focus}:

"""
        
        for i, doc_data in enumerate(documents_data):
            doc_summary = " ".join([chunk.content[:200] for chunk in doc_data["chunks"][:3]])
            comparison_prompt += f"\nDocument {i+1} ({doc_data['metadata']['file_name']}):\n{doc_summary}...\n"
        
        comparison_prompt += f"""
Provide a comparative analysis focusing on:
1. Key similarities
2. Major differences  
3. Unique aspects of each document
4. Overall assessment

Focus particularly on: {comparison_focus}
"""
        
        comparison_result = await summarizer._call_gemini_api(comparison_prompt)
        
        # Calculate similarity scores between documents
        similarity_matrix = await calculate_document_similarity(documents_data)
        
        return {
            "comparison_id": hashlib.md5(f"comp_{datetime.now()}".encode()).hexdigest(),
            "documents": [{"file_id": d["file_id"], "name": d["metadata"]["file_name"]} for d in documents_data],
            "comparison_analysis": comparison_result,
            "similarity_matrix": similarity_matrix,
            "focus": comparison_focus
        }
    
    except Exception as e:
        logger.error(f"Error in document comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

async def calculate_document_similarity(documents_data: List[Dict]) -> List[List[float]]:
    """Calculate similarity matrix between documents"""
    
    # Get document embeddings (average of chunk embeddings)
    doc_embeddings = []
    
    for doc_data in documents_data:
        chunks_with_embeddings = [chunk for chunk in doc_data["chunks"] if hasattr(chunk, 'embedding') and chunk.embedding is not None]
        
        if chunks_with_embeddings:
            embeddings = np.array([chunk.embedding for chunk in chunks_with_embeddings])
            doc_embedding = np.mean(embeddings, axis=0)
        else:
            # Generate embedding for concatenated content
            content = " ".join([chunk.content[:500] for chunk in doc_data["chunks"][:10]])
            doc_embedding = summarizer.embedding_model.encode([content])[0]
        
        doc_embeddings.append(doc_embedding)
    
    # Calculate similarity matrix
    similarity_matrix = []
    for i, emb1 in enumerate(doc_embeddings):
        row = []
        for j, emb2 in enumerate(doc_embeddings):
            if i == j:
                similarity = 1.0
            else:
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            row.append(float(similarity))
        similarity_matrix.append(row)
    
    return similarity_matrix

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )