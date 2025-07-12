# Real Estate RAG System - Complete Integrated Version with LLM
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document
import re
from collections import defaultdict
import pickle
from pathlib import Path
import os
import uvicorn
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Hugging Face authentication
def setup_huggingface_auth():
    """Setup Hugging Face authentication using environment variables"""
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    
    if hf_token:
        logger.info("✅ Hugging Face token found, setting up authentication")
        # Login to Hugging Face Hub
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("✅ Successfully authenticated with Hugging Face Hub")
        except ImportError:
            logger.warning("huggingface_hub not installed, using token as environment variable")
            os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
    else:
        logger.warning("⚠️ No Hugging Face token found. Some models may not be accessible.")
        logger.info("💡 Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN environment variable for authentication")

# Setup HF authentication at startup
setup_huggingface_auth()

# Initialize FastAPI app
app = FastAPI(title="Real Estate RAG System", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
def get_database_connection():
    """Get database connection with fallback"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            logger.warning("DATABASE_URL not found, using SQLite fallback")
            database_url = "sqlite:///./fallback.db"
        
        logger.info(f"Connecting to database: {database_url[:50]}...")
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("✅ Database connection successful")
        return engine
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Using in-memory SQLite as fallback")
        return create_engine("sqlite:///:memory:")

# Initialize database
try:
    engine = get_database_connection()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DATABASE_AVAILABLE = True
except Exception as e:
    logger.error(f"Database setup failed: {e}")
    DATABASE_AVAILABLE = False
    SessionLocal = None

# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    use_llm: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: Optional[List[Dict[str, Any]]] = []
    metrics: Optional[List[Dict[str, Any]]] = []
    model_used: Optional[str] = None
    timestamp: datetime = None

@dataclass
class QueryResult:
    """Structure for RAG query results"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    temporal_context: List[Dict[str, Any]]
    related_metrics: List[Dict[str, Any]]
    model_used: Optional[str] = None

# LLM Configuration
class LLMConfig:
    """Configuration for different LLM options"""
    # Smaller, faster model (recommended for most use cases)
    FLAN_T5_BASE = {
        "model_name": "google/flan-t5-base",
        "model_type": "t5",
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Larger model for better quality
    FLAN_T5_LARGE = {
        "model_name": "google/flan-t5-large",
        "model_type": "t5",
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

class TransformersLLM:
    """Wrapper for Hugging Face Transformers models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = LLMConfig.FLAN_T5_BASE
        
        self.config = config
        self.device = config["device"]
        
        logger.info(f"Loading {config['model_name']} on {self.device}...")
        
        try:
            if config["model_type"] == "t5":
                self.tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
                self.model = T5ForConditionalGeneration.from_pretrained(
                    config["model_name"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            
            logger.info("✅ LLM loaded successfully")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.available = False
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        if not self.available:
            return "LLM not available - returning mock response"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config["max_length"]
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

class TemporalChunker:
    """Advanced chunking that preserves temporal relationships"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def chunk_with_temporal_context(self, documents: List[Document]) -> List[Document]:
        """Chunk documents while preserving temporal relationships"""
        if not documents:
            return []
        
        try:
            chunked_docs = []
            
            # Group documents by time periods
            temporal_groups = self._group_by_temporal_period(documents)
            
            for period, docs in temporal_groups.items():
                # Create context summary for this period
                period_context = self._create_period_context(period, docs)
                
                for doc in docs:
                    # Split document into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    
                    for i, chunk in enumerate(chunks):
                        # Add temporal context to each chunk
                        enhanced_chunk = self._enhance_chunk_with_context(
                            chunk, period_context, i, len(chunks)
                        )
                        chunked_docs.append(enhanced_chunk)
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error in temporal chunking: {e}")
            return self.text_splitter.split_documents(documents)
    
    def _group_by_temporal_period(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by time period"""
        temporal_groups = defaultdict(list)
        
        for doc in documents:
            # Extract date from metadata or content
            date = doc.metadata.get('date', datetime.now().strftime('%Y-%m'))
            period = date[:7]  # Group by year-month
            temporal_groups[period].append(doc)
        
        return dict(temporal_groups)
    
    def _create_period_context(self, period: str, docs: List[Document]) -> str:
        """Create a context summary for a time period"""
        return f"Period {period}: {len(docs)} documents"
    
    def _enhance_chunk_with_context(self, chunk: Document, period_context: str, 
                                   chunk_index: int, total_chunks: int) -> Document:
        """Enhance chunk with temporal context"""
        chunk.metadata['temporal_context'] = period_context
        chunk.metadata['chunk_index'] = chunk_index
        chunk.metadata['total_chunks'] = total_chunks
        return chunk

class RealEstateRAGSystem:
    """Main RAG system for real estate analysis with LLM integration"""
    
    def __init__(self, use_llm: bool = True):
        logger.info("Initializing Real Estate RAG System...")
        
        # Initialize embeddings for retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize temporal chunker
        self.chunker = TemporalChunker()
        
        # Initialize LLM if requested
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = TransformersLLM(LLMConfig.FLAN_T5_BASE)
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
        
        # Initialize prompt templates
        self.qa_prompt_template = """You are a real estate market analyst. Use the following context to answer the question. Be specific and cite relevant data points from the context.

Context:
{context}

Question: {question}

Instructions:
- Base your answer only on the provided context
- Include specific numbers, percentages, and dates when available
- If the context doesn't contain enough information, say so
- Be concise but comprehensive

Answer:"""

        # Load initial data
        self._load_initial_data()
        
        logger.info("✅ RAG System initialized successfully")
    
    def _load_initial_data(self):
        """Load sample real estate data into the system"""
        try:
            sample_docs = [
                Document(
                    page_content="Commercial real estate in Manhattan saw a 15% price increase in Q3 2024, driven by return-to-office mandates and foreign investment. Average price per square foot reached $1,250 in prime locations. Office vacancy rates dropped to 12.3%, the lowest since 2020.",
                    metadata={"source": "Manhattan Market Report", "date": "2024-09-30", "property_type": "commercial", "location": "Manhattan"}
                ),
                Document(
                    page_content="The industrial real estate sector continues to outperform, with warehouse properties near major ports seeing 8% annual appreciation. E-commerce growth drives demand for last-mile delivery facilities. Cap rates for prime industrial assets range from 4.5% to 5.5%.",
                    metadata={"source": "Industrial Analysis", "date": "2024-10-15", "property_type": "industrial", "location": "National"}
                ),
                Document(
                    page_content="Residential mortgage rates stabilized at 6.5% in October 2024, leading to a slight uptick in home sales. First-time buyers remain challenged by affordability constraints in major markets. The median home price nationally reached $425,000.",
                    metadata={"source": "Residential Report", "date": "2024-10-20", "property_type": "residential", "location": "National"}
                ),
                Document(
                    page_content="Green building certifications are becoming increasingly important for institutional investors. LEED-certified properties command a 7% premium on average. ESG considerations now factor into 78% of commercial real estate investment decisions.",
                    metadata={"source": "ESG Report", "date": "2024-10-10", "property_type": "commercial", "focus": "sustainability"}
                ),
                Document(
                    page_content="Retail real estate shows signs of recovery with adaptive reuse projects gaining momentum. Mixed-use developments combining retail, office, and residential are attracting significant investment. Experiential retail concepts are driving foot traffic.",
                    metadata={"source": "Retail Trends", "date": "2024-10-05", "property_type": "retail", "location": "National"}
                )
            ]
            
            # Process and index documents
            chunked_docs = self.chunker.chunk_with_temporal_context(sample_docs)
            self._index_documents(chunked_docs)
            
            logger.info(f"Loaded {len(sample_docs)} documents, created {len(chunked_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
    
    def _index_documents(self, documents: List[Document]):
        """Index documents in vector store"""
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, use_llm: Optional[bool] = None) -> QueryResult:
        """Process a query through the RAG system"""
        try:
            # Determine whether to use LLM
            use_llm = use_llm if use_llm is not None else self.use_llm
            
            # Step 1: Retrieve relevant documents
            relevant_docs = self._retrieve_documents(question, filters)
            
            if not relevant_docs:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    temporal_context=[],
                    related_metrics=[],
                    model_used=None
                )
            
            # Step 2: Build context
            context = self._build_context(relevant_docs)
            
            # Step 3: Generate answer
            if use_llm and self.llm and self.llm.available:
                # Use LLM for generation
                prompt = self.qa_prompt_template.format(
                    context=context,
                    question=question
                )
                answer = self.llm.generate(prompt, max_new_tokens=300)
                model_used = self.llm.config["model_name"]
            else:
                # Fallback to simple extraction
                answer = self._simple_answer_extraction(question, relevant_docs)
                model_used = "simple_extraction"
            
            # Step 4: Post-process and calculate metrics
            answer = self._post_process_answer(answer, question)
            confidence = self._calculate_confidence(relevant_docs, answer)
            metrics = self._extract_metrics_from_answer(answer)
            temporal_context = self._extract_temporal_context(relevant_docs)
            
            return QueryResult(
                answer=answer,
                sources=self._format_sources(relevant_docs),
                confidence=confidence,
                temporal_context=temporal_context,
                related_metrics=metrics,
                model_used=model_used
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                temporal_context=[],
                related_metrics=[],
                model_used=None
            )
    
    def _retrieve_documents(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using similarity search"""
        try:
            # Build metadata filter if provided
            where_clause = None
            if filters:
                where_clause = {}
                if 'date_from' in filters:
                    where_clause['date'] = {'$gte': filters['date_from']}
                if 'property_type' in filters:
                    where_clause['property_type'] = filters['property_type']
                if 'location' in filters:
                    where_clause['location'] = {'$contains': filters['location']}
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=where_clause
            )
            
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _build_context(self, docs: List[Document]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown')
            date = doc.metadata.get('date', 'Unknown')
            
            # Format context entry
            context_entry = f"[Document {i} - Source: {source}, Date: {date}]\n{doc.page_content}"
            context_parts.append(context_entry)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _simple_answer_extraction(self, question: str, docs: List[Document]) -> str:
        """Simple answer extraction when LLM is not available"""
        # Combine relevant content
        combined_content = "\n".join([doc.page_content for doc in docs[:3]])
        
        # Extract key sentences based on question keywords
        question_lower = question.lower()
        sentences = combined_content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check for question keywords in sentence
            if any(keyword in sentence_lower for keyword in question_lower.split() if len(keyword) > 3):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = ". ".join(relevant_sentences[:3]) + "."
            return f"Based on available data: {answer}"
        else:
            return f"Based on the search results, here's what I found: {combined_content[:300]}..."
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Clean and enhance the generated answer"""
        # Remove any repeated text
        lines = answer.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        
        answer = '\n'.join(unique_lines)
        
        return answer.strip()
    
    def _calculate_confidence(self, docs: List[Document], answer: str) -> float:
        """Calculate confidence score based on retrieval quality and answer"""
        base_confidence = 0.6
        
        # Factor 1: Number of relevant documents
        doc_score = min(len(docs) * 0.1, 0.3)
        
        # Factor 2: Answer length and quality
        answer_score = 0.1 if len(answer) > 100 else 0.0
        
        # Factor 3: Presence of specific data points
        data_score = 0.1 if any(char.isdigit() for char in answer) else 0.0
        
        return min(base_confidence + doc_score + answer_score + data_score, 0.95)
    
    def _extract_metrics_from_answer(self, answer: str) -> List[Dict[str, str]]:
        """Extract quantitative metrics from the answer"""
        metrics = []
        
        # Pattern for percentages
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percent_pattern, answer)
        for pct in percentages:
            metrics.append({"type": "percentage", "value": f"{pct}%"})
        
        # Pattern for dollar amounts
        dollar_pattern = r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([BMK])?'
        amounts = re.findall(dollar_pattern, answer)
        for amount, suffix in amounts:
            value = f"${amount}{suffix}" if suffix else f"${amount}"
            metrics.append({"type": "currency", "value": value})
        
        return metrics[:5]  # Limit to 5 metrics
    
    def _extract_temporal_context(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract temporal patterns from retrieved documents"""
        temporal_data = []
        
        for doc in docs:
            if 'date' in doc.metadata:
                temporal_data.append({
                    'date': doc.metadata['date'],
                    'source': doc.metadata.get('source', 'Unknown'),
                    'property_type': doc.metadata.get('property_type', 'Unknown')
                })
        
        return temporal_data
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        sources = []
        
        for doc in docs:
            sources.append({
                "content_preview": doc.page_content[:150] + "...",
                "metadata": doc.metadata
            })
        
        return sources

# Initialize RAG system
rag_system = None
try:
    # Try to initialize with LLM, fallback to simple mode if needed
    rag_system = RealEstateRAGSystem(use_llm=True)
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    try:
        # Try without LLM
        rag_system = RealEstateRAGSystem(use_llm=False)
    except Exception as e:
        logger.error(f"Failed to initialize even basic RAG system: {e}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <head><title>Real Estate RAG System</title></head>
        <body>
        <h1>🏢 Real Estate RAG System</h1>
        <p>Dashboard loading... Please create an index.html file for the full interface.</p>
        <p><a href="/health">Health Check</a> | <a href="/metrics">Metrics</a></p>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llm_status = "unavailable"
    if rag_system and rag_system.llm:
        llm_status = "ready" if rag_system.llm.available else "failed"
    
    return {
        "status": "healthy",
        "database_available": DATABASE_AVAILABLE,
        "rag_system": "ready" if rag_system else "unavailable",
        "llm_status": llm_status,
        "hf_token_configured": bool(os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a real estate query"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system unavailable")
        
        result = rag_system.query(request.question, request.filters, request.use_llm)
        
        return QueryResponse(
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources,
            metrics=result.related_metrics,
            model_used=result.model_used,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get current market metrics"""
    try:
        return {
            "metrics": [
                {"name": "Global Market Cap", "value": "$4.2T", "change": "+3.2%"},
                {"name": "Average Cap Rate", "value": "5.4%", "change": "+0.3%"},
                {"name": "Transaction Volume", "value": "$847B", "change": "-1.8%"},
                {"name": "Industrial Growth", "value": "5.1%", "change": "+2.1%"}
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/competitors")
async def get_competitors():
    """Get competitor information"""
    try:
        return {
            "competitors": [
                {
                    "company_name": "Blackstone",
                    "market_share": 18.5,
                    "recent_transaction": "Acquired $2.3B industrial portfolio",
                    "trend": "stable"
                },
                {
                    "company_name": "Brookfield",
                    "market_share": 15.2,
                    "recent_transaction": "Launched $8B renewable energy fund",
                    "trend": "growing"
                },
                {
                    "company_name": "Prologis",
                    "market_share": 12.8,
                    "recent_transaction": "Expanded logistics network in Asia",
                    "trend": "stable"
                }
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving competitors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_documents")
async def add_documents(documents: List[Dict[str, Any]]):
    """Add new documents to the RAG system"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system unavailable")
        
        # Convert to Document objects
        doc_objects = []
        for doc in documents:
            doc_objects.append(Document(
                page_content=doc.get("content", ""),
                metadata=doc.get("metadata", {})
            ))
        
        # Process and add to system
        chunked_docs = rag_system.chunker.chunk_with_temporal_context(doc_objects)
        rag_system._index_documents(chunked_docs)
        
        return {
            "status": "success",
            "documents_added": len(doc_objects),
            "chunks_created": len(chunked_docs),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution
if __name__ == "__main__":
    import sys
    
    # Get port from environment variable with fallback
    port = int(os.environ.get("PORT", 8000))
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI mode
        print("🌍 Real Estate RAG System - CLI Mode")
        if rag_system:
            print(f"📊 LLM Status: {'Available' if rag_system.llm and rag_system.llm.available else 'Not Available'}")
            print(f"🤗 HF Token: {'Configured' if os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN') else 'Not Configured'}")
            while True:
                try:
                    question = input("\n💬 Enter your question (or 'exit'): ")
                    if question.lower() == 'exit':
                        break
                    
                    print("🔍 Processing...")
                    result = rag_system.query(question)
                    
                    print(f"\n📊 Answer: {result.answer}")
                    print(f"🎯 Confidence: {result.confidence:.2%}")
                    print(f"🤖 Model: {result.model_used}")
                    
                    if result.related_metrics:
                        print(f"📈 Metrics found: {len(result.related_metrics)}")
                        for metric in result.related_metrics[:3]:
                            print(f"   - {metric['value']}")
                            
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
            print("\n👋 Goodbye!")
        else:
            print("❌ RAG system not available")
    else:
        # Run FastAPI server
        print(f"🚀 Starting Real Estate RAG System on port {port}")
        print(f"📊 LLM Status: {'Available' if rag_system and rag_system.llm and rag_system.llm.available else 'Not Available'}")
        print(f"🤗 HF Token: {'Configured' if os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN') else 'Not Configured'}")
        print(f"🌐 Access the dashboard at http://localhost:{port}")
        
        uvicorn.run(
            "real_estate_rag_system:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            workers=1
        )
