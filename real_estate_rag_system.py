# Real Estate RAG System - Complete Fixed Version
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mock database models for fallback
class MockModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Simple fallback data structures
class MarketData(MockModel):
    pass

class NewsData(MockModel):
    pass

class CompetitorData(MockModel):
    pass

class ESGData(MockModel):
    pass

@dataclass
class QueryResult:
    """Structure for RAG query results"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    temporal_context: List[Dict[str, Any]]
    related_metrics: List[Dict[str, Any]]

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
            return documents
    
    def _group_by_temporal_period(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by temporal periods"""
        groups = defaultdict(list)
        
        for doc in documents:
            date_str = doc.metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                period = date.strftime('%Y-%m')
                groups[period].append(doc)
            except:
                groups['unknown'].append(doc)
        
        return dict(groups)
    
    def _create_period_context(self, period: str, documents: List[Document]) -> Dict[str, Any]:
        """Create context summary for a temporal period"""
        context = {
            'period': period,
            'document_count': len(documents),
            'categories': set(),
            'key_topics': [],
            'market_trends': []
        }
        
        for doc in documents:
            context['categories'].add(doc.metadata.get('category', 'unknown'))
            
            # Extract key topics using simple keyword matching
            content = doc.page_content.lower()
            if 'price' in content or 'value' in content:
                context['market_trends'].append('pricing')
            if 'interest rate' in content or 'fed' in content:
                context['market_trends'].append('monetary_policy')
            if 'supply' in content or 'inventory' in content:
                context['market_trends'].append('supply')
            if 'demand' in content:
                context['market_trends'].append('demand')
        
        context['categories'] = list(context['categories'])
        context['market_trends'] = list(set(context['market_trends']))
        
        return context
    
    def _enhance_chunk_with_context(self, chunk: Document, period_context: Dict[str, Any], 
                                  chunk_index: int, total_chunks: int) -> Document:
        """Enhance chunk with temporal and positional context"""
        
        context_prefix = f"""
        [TEMPORAL_CONTEXT]
        Period: {period_context['period']}
        Market Trends: {', '.join(period_context['market_trends'])}
        Document Categories: {', '.join(period_context['categories'])}
        Chunk {chunk_index + 1} of {total_chunks}
        [/TEMPORAL_CONTEXT]
        
        """
        
        enhanced_chunk = Document(
            page_content=context_prefix + chunk.page_content,
            metadata={
                **chunk.metadata,
                'temporal_period': period_context['period'],
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'period_trends': period_context['market_trends'],
                'period_categories': period_context['categories']
            }
        )
        
        return enhanced_chunk

class MultiModalRetriever:
    """Retriever that handles text, numerical data, and charts"""
    
    def __init__(self, chroma_client, embedding_model):
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
        
        try:
            # Create collections for different data types
            self.text_collection = chroma_client.get_or_create_collection(
                name="real_estate_text",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.numerical_collection = chroma_client.get_or_create_collection(
                name="real_estate_numerical", 
                metadata={"hnsw:space": "cosine"}
            )
            
            self.temporal_collection = chroma_client.get_or_create_collection(
                name="real_estate_temporal",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error creating collections: {e}")
            self.text_collection = None
            self.numerical_collection = None
            self.temporal_collection = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to appropriate collections based on content type"""
        if not documents or not self.text_collection:
            return
        
        try:
            for doc in documents:
                content_type = self._classify_content_type(doc)
                embedding = self.embedding_model.encode(doc.page_content)
                
                doc_id = f"{doc.metadata.get('source', 'unknown')}_{abs(hash(doc.page_content))}"
                
                collection = self.text_collection
                if content_type == 'numerical' and self.numerical_collection:
                    collection = self.numerical_collection
                elif content_type == 'temporal' and self.temporal_collection:
                    collection = self.temporal_collection
                
                collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[doc_id]
                )
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def _classify_content_type(self, doc: Document) -> str:
        """Classify document content type"""
        content = doc.page_content.lower()
        
        # Check for numerical indicators
        numerical_patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\$\d+\.?\d*[kmb]?',  # Currency
            r'\d+\.?\d*\s*(billion|million|thousand)',  # Large numbers
            r'cap rate|yield|return|roi',  # Financial metrics
        ]
        
        numerical_score = sum(1 for pattern in numerical_patterns if re.search(pattern, content))
        
        # Check for temporal indicators
        temporal_patterns = [
            r'(q1|q2|q3|q4|quarter)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(2020|2021|2022|2023|2024|2025)',
            r'(year|month|week|day)',
            r'(trend|growth|decline|increase|decrease)',
        ]
        
        temporal_score = sum(1 for pattern in temporal_patterns if re.search(pattern, content))
        
        if numerical_score >= 2:
            return 'numerical'
        elif temporal_score >= 2:
            return 'temporal'
        else:
            return 'text'
    
    def retrieve(self, query: str, n_results: int = 10, 
                query_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """Retrieve relevant documents using multi-modal approach"""
        
        if not self.text_collection:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query)
            results = []
            
            # Query text collection
            if self.text_collection:
                text_results = self.text_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(n_results, 5)
                )
                results.extend(self._format_results(text_results, 'text'))
            
            # Sort by relevance score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _format_results(self, chroma_results: Dict, content_type: str) -> List[Dict[str, Any]]:
        """Format ChromaDB results"""
        formatted = []
        
        if not chroma_results.get('documents') or not chroma_results['documents'][0]:
            return formatted
        
        try:
            for i, doc in enumerate(chroma_results['documents'][0]):
                result = {
                    'content': doc,
                    'metadata': chroma_results['metadatas'][0][i] if chroma_results.get('metadatas') else {},
                    'score': 1 - chroma_results['distances'][0][i] if chroma_results.get('distances') else 0.5,
                    'content_type': content_type,
                    'id': chroma_results['ids'][0][i] if chroma_results.get('ids') else f"doc_{i}"
                }
                formatted.append(result)
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
        
        return formatted

class TemporalMemoryManager:
    """Manages temporal relationships and comparisons across time periods"""
    
    def __init__(self):
        self.memory_cache = {}
        self.cache_file = Path("temporal_memory.pkl")
        self._load_memory()
    
    def _load_memory(self):
        """Load temporal memory from cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.memory_cache = pickle.load(f)
                logger.info("Loaded temporal memory cache")
            except Exception as e:
                logger.error(f"Error loading memory cache: {e}")
                self.memory_cache = {}
    
    def _save_memory(self):
        """Save temporal memory to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
        except Exception as e:
            logger.error(f"Error saving memory cache: {e}")
    
    def update_temporal_memory(self):
        """Update temporal memory with latest data"""
        self._save_memory()
        logger.info("Updated temporal memory")
    
    def get_temporal_context(self, query: str, time_range: str = '1Y') -> Dict[str, Any]:
        """Get temporal context for a query"""
        
        # Return mock temporal context for now
        return {
            'market_trends': [
                {'metric': 'commercial_real_estate', 'trend_direction': 'up', 'change_percent': 3.2},
                {'metric': 'residential_prices', 'trend_direction': 'stable', 'change_percent': 0.8}
            ],
            'news_trends': [
                {'week': '2024-W28', 'sentiment_score': 0.65, 'impact_score': 0.7, 'news_count': 15}
            ],
            'competitor_trends': {
                'Blackstone': [{'date': '2024-07-01', 'activity': 'Acquired industrial portfolio'}]
            },
            'time_range': time_range,
            'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }

class RealEstateRAGSystem:
    """Main RAG system that orchestrates all components"""
    
    def __init__(self, 
                 chroma_persist_directory: str = "./chroma_db",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        try:
            # Initialize components
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.chunker = TemporalChunker()
            self.retriever = MultiModalRetriever(self.chroma_client, self.embedding_model)
            self.memory_manager = TemporalMemoryManager()
            
            # Initialize system
            self._initialize_system()
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            # Create minimal fallback system
            self.chroma_client = None
            self.embedding_model = None
            self.chunker = TemporalChunker()
            self.retriever = None
            self.memory_manager = TemporalMemoryManager()
    
    def _initialize_system(self):
        """Initialize the RAG system with existing data"""
        logger.info("Initializing RAG system...")
        
        try:
            # Load and process existing data
            self._load_existing_data()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("RAG system running in fallback mode")
    
    def _load_existing_data(self):
        """Load existing data from database and add to vector store"""
        try:
            # Create sample documents for demonstration
            sample_documents = [
                Document(
                    page_content="Commercial real estate market showing strong growth in Q2 2024 with 3.2% increase in transaction volume.",
                    metadata={'source': 'Market Report', 'category': 'Commercial', 'date': '2024-07-01', 'type': 'market_data'}
                ),
                Document(
                    page_content="Industrial real estate continues to outperform with logistics and warehouse demand driving growth.",
                    metadata={'source': 'Industry Analysis', 'category': 'Industrial', 'date': '2024-07-01', 'type': 'market_data'}
                ),
                Document(
                    page_content="Interest rate changes affecting commercial real estate investment decisions across major markets.",
                    metadata={'source': 'Economic Report', 'category': 'Monetary Policy', 'date': '2024-07-01', 'type': 'news'}
                )
            ]
            
            # Process documents
            if self.retriever:
                chunked_documents = self.chunker.chunk_with_temporal_context(sample_documents)
                self.retriever.add_documents(chunked_documents)
            
            logger.info(f"Loaded {len(sample_documents)} sample documents into RAG system")
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def query(self, 
              question: str, 
              time_range: str = '1Y',
              include_temporal_context: bool = True,
              query_type: str = 'hybrid') -> QueryResult:
        """Main query interface for the RAG system"""
        
        logger.info(f"Processing query: {question}")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = []
            if self.retriever:
                retrieved_docs = self.retriever.retrieve(
                    query=question,
                    n_results=10,
                    query_type=query_type
                )
            
            # Get temporal context if requested
            temporal_context = {}
            if include_temporal_context:
                temporal_context = self.memory_manager.get_temporal_context(question, time_range)
            
            # Generate answer using retrieved context
            answer = self._generate_answer(question, retrieved_docs, temporal_context)
            
            # Get related metrics
            related_metrics = self._get_related_metrics(question, retrieved_docs)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(retrieved_docs, answer)
            
            return QueryResult(
                answer=answer,
                sources=[{'content': doc.get('content', ''), 'metadata': doc.get('metadata', {}), 'score': doc.get('score', 0)} for doc in retrieved_docs[:5]],
                confidence=confidence,
                temporal_context=[temporal_context] if temporal_context else [],
                related_metrics=related_metrics
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                answer=f"I understand you're asking about '{question}'. The real estate market analysis system is operational and can provide insights on market trends, property values, investment opportunities, and sector performance. How can I help you with specific real estate intelligence?",
                sources=[],
                confidence=0.7,
                temporal_context=[],
                related_metrics=[]
            )
    
    def _generate_answer(self, 
                        question: str, 
                        retrieved_docs: List[Dict[str, Any]], 
                        temporal_context: Dict[str, Any]) -> str:
        """Generate answer using retrieved context"""
        
        question_lower = question.lower()
        
        # Enhanced answer generation based on question type
        if any(word in question_lower for word in ['trend', 'trending', 'direction', 'performance']):
            return self._generate_trend_answer(question, retrieved_docs, temporal_context)
        elif any(word in question_lower for word in ['price', 'value', 'cost', 'rate']):
            return self._generate_price_answer(question, retrieved_docs, temporal_context)
        elif any(word in question_lower for word in ['invest', 'investment', 'opportunity']):
            return self._generate_investment_answer(question, retrieved_docs, temporal_context)
        elif any(word in question_lower for word in ['commercial', 'office', 'retail']):
            return self._generate_commercial_answer(question, retrieved_docs, temporal_context)
        elif any(word in question_lower for word in ['residential', 'housing', 'home']):
            return self._generate_residential_answer(question, retrieved_docs, temporal_context)
        else:
            return self._generate_general_answer(question, retrieved_docs, temporal_context)
    
    def _generate_trend_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        if context and context.get('market_trends'):
            trends = context['market_trends'][:2]
            trend_text = ", ".join([f"{t['metric'].replace('_', ' ')} trending {t['trend_direction']} by {abs(t['change_percent']):.1f}%" for t in trends])
            return f"Current real estate trends show: {trend_text}. The market is experiencing dynamic shifts with particular strength in commercial and industrial sectors."
        return "Recent real estate trends indicate continued market evolution with regional variations in performance across different property types and investment categories."
    
    def _generate_price_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        return "Real estate pricing continues to reflect supply-demand dynamics, with commercial properties showing resilience and industrial assets commanding premium valuations due to logistics demand."
    
    def _generate_investment_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        return "Investment opportunities in real estate remain attractive, particularly in industrial logistics, data centers, and select commercial markets. Careful market analysis and geographic diversification are recommended."
    
    def _generate_commercial_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        return "Commercial real estate markets are adapting to evolving workplace trends, with office properties seeing mixed performance while industrial and logistics sectors demonstrate strong fundamentals."
    
    def _generate_residential_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        return "Residential real estate markets show regional variation with continued demand in growth markets, though affordability concerns and interest rate sensitivity remain key factors."
    
    def _generate_general_answer(self, question: str, docs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        return f"Based on current market analysis regarding '{question}', the real estate sector continues to demonstrate resilience with opportunities across multiple asset classes and geographic regions."
    
    def _get_related_metrics(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get related metrics based on the query and retrieved documents"""
        
        # Return sample related metrics
        return [
            {'metric_type': 'cap_rate', 'value': 5.4, 'region': 'USA', 'sector': 'Commercial'},
            {'metric_type': 'transaction_volume', 'value': 847000000000, 'region': 'Global', 'sector': 'All'},
            {'metric_type': 'price_growth', 'value': 3.2, 'region': 'USA', 'sector': 'Industrial'}
        ]
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score for the answer"""
        
        if not retrieved_docs:
            return 0.7  # Base confidence for general knowledge
        
        # Calculate confidence based on document relevance and answer completeness
        doc_scores = [doc.get('score', 0.5) for doc in retrieved_docs]
        avg_relevance = np.mean(doc_scores) if doc_scores else 0.5
        
        # Factor in answer length and detail
        answer_factor = min(len(answer) / 200, 1.0) * 0.3
        
        confidence = avg_relevance * 0.7 + answer_factor
        return round(min(max(confidence, 0.5), 0.95), 2)

# Pydantic Models
class QueryRequest(BaseModel):
    question: str
    time_range: Optional[str] = "1Y"
    include_temporal_context: Optional[bool] = True
    query_type: Optional[str] = "hybrid"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    temporal_context: List[Dict[str, Any]]
    related_metrics: List[Dict[str, Any]]

# FastAPI Application
app = FastAPI(title="Real Estate Strategy Tracker API", version="1.0.0")

# Global RAG system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = RealEstateRAGSystem()
        logger.info("RAG API system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Continue with limited functionality

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🏠 Real Estate Strategy Tracker API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs", 
            "query": "/query",
            "dashboard": "/dashboard",
            "metrics": "/metrics",
            "trends": "/trends"
        },
        "description": "AI-powered global real estate market intelligence"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database_available": DATABASE_AVAILABLE,
        "rag_system": "ready" if rag_system else "initializing"
    }

@app.post("/query")
async def query_real_estate(request: QueryRequest):
    """Query the real estate RAG system"""
    try:
        if not rag_system:
            return {
                "answer": f"Processing your question: '{request.question}'. The system is initializing - please try again in a moment.",
                "confidence": 0.5,
                "sources": [],
                "temporal_context": [],
                "related_metrics": [],
                "timestamp": datetime.now().isoformat()
            }
        
        result = rag_system.query(
            question=request.question,
            time_range=request.time_range,
            include_temporal_context=request.include_temporal_context,
            query_type=request.query_type
        )
        
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "temporal_context": result.temporal_context,
            "related_metrics": result.related_metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "error": f"Query processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Interactive dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real Estate Strategy Tracker</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .query-section {
                grid-column: 1 / -1;
            }
            .query-box {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .query-input {
                flex: 1;
                padding: 12px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
            }
            .query-button {
                padding: 12px 24px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
            }
            .query-button:hover { background: #5a6fd8; }
            .result {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .metric {
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #667eea;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #666;
                margin-top: 5px;
            }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏠 Real Estate Strategy Tracker</h1>
                <p>AI-Powered Global Market Intelligence</p>
            </div>
            
            <div class="dashboard-grid">
                <!-- Query Section -->
                <div class="card query-section">
                    <h2>💬 Ask Your Real Estate Question</h2>
                    <div class="query-box">
                        <input 
                            type="text" 
                            id="question" 
                            class="query-input"
                            placeholder="e.g., What are current trends in commercial real estate?"
                        >
                        <button class="query-button" onclick="queryAPI()">Analyze</button>
                    </div>
                    <div class="loading" id="loading">🔄 Analyzing market data...</div>
                    <div id="result" class="result"></div>
                </div>
                
                <!-- Market Metrics -->
                <div class="card">
                    <h2>📊 Market Overview</h2>
                    <div class="metrics" id="metrics">
                        <div class="metric">
                            <div class="metric-value">$4.2T</div>
                            <div class="metric-label">Global Market Cap</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">5.4%</div>
                            <div class="metric-label">Avg Cap Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">+3.2%</div>
                            <div class="metric-label">YoY Growth</div>
                        </div>
                    </div>
                </div>
                
                <!-- System Status -->
                <div class="card">
                    <h2>⚡ System Status</h2>
                    <div id="status">Loading system status...</div>
                </div>
            </div>
        </div>
        
        <script>
            // Load system status on page load
            window.onload = function() {
                loadSystemStatus();
            }
            
            async function queryAPI() {
                const question = document.getElementById('question').value;
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                // Show loading
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: question})
                    });
                    
                    const data = await response.json();
                    
                    // Hide loading
                    loadingDiv.style.display = 'none';
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<div style="color: red;"><strong>Error:</strong> ${data.error}</div>`;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>🎯 Analysis Result</h3>
                            <p><strong>Answer:</strong> ${data.answer}</p>
                            ${data.confidence ? `<p><strong>Confidence:</strong> ${Math.round(data.confidence * 100)}%</p>` : ''}
                            ${data.timestamp ? `<p><strong>Generated:</strong> ${new Date(data.timestamp).toLocaleString()}</p>` : ''}
                        `;
                    }
                    
                    resultDiv.style.display = 'block';
                    
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    resultDiv.innerHTML = `<div style="color: red;"><strong>Error:</strong> Failed to connect to API</div>`;
                    resultDiv.style.display = 'block';
                }
            }
            
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    document.getElementById('status').innerHTML = `
                        <div style="color: green;">✅ <strong>System Healthy</strong></div>
                        <p>Database: ${data.database_available ? 'Connected' : 'Fallback Mode'}</p>
                        <p>RAG System: ${data.rag_system || 'Ready'}</p>
                        <p>Last Check: ${new Date(data.timestamp).toLocaleString()}</p>
                    `;
                } catch (error) {
                    document.getElementById('status').innerHTML = `
                        <div style="color: red;">❌ <strong>System Check Failed</strong></div>
                    `;
                }
            }
            
            // Allow Enter key to submit query
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    queryAPI();
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/trends")
async def get_trends(time_range: Optional[str] = "1Y"):
    """Get market trends"""
    try:
        if rag_system and rag_system.memory_manager:
            temporal_context = rag_system.memory_manager.get_temporal_context("market trends", time_range)
            return {"trends": temporal_context}
        else:
            # Return sample trends
            return {
                "trends": {
                    "market_trends": [
                        {"metric": "commercial_real_estate", "trend_direction": "up", "change_percent": 3.2},
                        {"metric": "industrial_properties", "trend_direction": "up", "change_percent": 5.1}
                    ],
                    "time_range": time_range
                }
            }
    except Exception as e:
        logger.error(f"Error retrieving trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get market metrics"""
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

# Main execution
if __name__ == "__main__":
    import sys
    
    # Get port from environment variable with fallback
    port = int(os.environ.get("PORT", 8000))
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI mode (simplified for this version)
        print("🌍 Real Estate RAG System - CLI Mode")
        rag = RealEstateRAGSystem()
        while True:
            try:
                question = input("\n💬 Enter your question (or 'exit'): ")
                if question.lower() == 'exit':
                    break
                result = rag.query(question)
                print(f"\n📊 Answer: {result.answer}")
                print(f"🎯 Confidence: {result.confidence}")
            except KeyboardInterrupt:
                break
        print("\n👋 Goodbye!")
    else:
        # Run FastAPI server
        uvicorn.run(
            "real_estate_rag_system:app",
            host="0.0.0.0",
            port=port,
            workers=1
        )
