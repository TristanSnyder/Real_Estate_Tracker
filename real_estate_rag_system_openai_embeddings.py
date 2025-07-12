# Real Estate RAG System - OpenAI Embeddings Version (Ultra Lightweight)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
from collections import defaultdict
import pickle
from pathlib import Path
import os
import uvicorn
import openai
from openai import OpenAI
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI authentication
def setup_openai_auth():
    """Setup OpenAI authentication using environment variables"""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if openai_api_key:
        logger.info("✅ OpenAI API key found, setting up authentication")
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=openai_api_key)
            # Test the connection with a simple request
            client.models.list()
            logger.info("✅ Successfully authenticated with OpenAI API")
            return client
        except Exception as e:
            logger.error(f"Failed to authenticate with OpenAI: {e}")
            return None
    else:
        logger.warning("⚠️ No OpenAI API key found. LLM features will be disabled.")
        logger.info("💡 Set OPENAI_API_KEY environment variable for LLM functionality")
        return None

# Setup OpenAI authentication at startup
openai_client = setup_openai_auth()

# Initialize FastAPI app
app = FastAPI(title="Real Estate RAG System", version="3.0.0")

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

# OpenAI Embeddings Class
class OpenAIEmbeddings:
    """Ultra-lightweight embeddings using OpenAI API"""
    
    def __init__(self, client: OpenAI = None, model: str = "text-embedding-ada-002"):
        self.client = client or openai_client
        self.model = model
        self.available = bool(self.client)
        
        if self.available:
            logger.info(f"✅ OpenAI Embeddings initialized with {model}")
        else:
            logger.warning("⚠️ OpenAI Embeddings not available - API key missing")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        if not self.available:
            logger.warning("Using fallback embeddings")
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            # OpenAI allows up to 2048 texts per request for ada-002
            embeddings = []
            batch_size = 100  # Conservative batch size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return [self._fallback_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.available:
            return self._fallback_embedding(text)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """Simple fallback embedding using text hash and padding"""
        # Simple hash-based embedding for fallback
        text_hash = hash(text.lower())
        np.random.seed(abs(text_hash))
        embedding = np.random.normal(0, 1, dim).tolist()
        return embedding

# Simple Vector Store
class SimpleVectorStore:
    """Ultra-lightweight vector store using SQLite"""
    
    def __init__(self, embeddings: OpenAIEmbeddings, db_path: str = "./vector_store.db"):
        self.embeddings = embeddings
        self.db_path = db_path
        self.dimension = 1536  # OpenAI ada-002 dimension
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for vector storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Vector store database initialized")
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Add texts to vector store"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Get embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            cursor.execute('''
                INSERT INTO documents (text, metadata, embedding)
                VALUES (?, ?, ?)
            ''', (text, json.dumps(metadata), json.dumps(embedding)))
        
        conn.commit()
        conn.close()
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5, filter: Dict = None) -> List[Document]:
        """Search for similar documents"""
        query_embedding = self.embeddings.embed_query(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all documents (in a real system, you'd want approximate nearest neighbor search)
        cursor.execute('SELECT text, metadata, embedding FROM documents')
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return []
        
        # Calculate similarities
        similarities = []
        for text, metadata_str, embedding_str in results:
            try:
                metadata = json.loads(metadata_str)
                embedding = json.loads(embedding_str)
                
                # Apply filter if provided
                if filter:
                    skip = False
                    for key, value in filter.items():
                        if key in metadata and metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((similarity, text, metadata))
                
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        documents = []
        for similarity, text, metadata in similarities[:k]:
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except:
            return 0.0

# LLM Configuration
class LLMConfig:
    """Configuration for different OpenAI models"""
    GPT_3_5_TURBO = {
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    GPT_4 = {
        "model_name": "gpt-4",
        "max_tokens": 512,
        "temperature": 0.7
    }

class OpenAILLM:
    """Wrapper for OpenAI API models"""
    
    def __init__(self, config: Dict[str, Any] = None, client: OpenAI = None):
        if config is None:
            config = LLMConfig.GPT_3_5_TURBO
        
        self.config = config
        self.client = client or openai_client
        
        logger.info(f"Initializing OpenAI LLM with {config['model_name']}...")
        
        if self.client:
            try:
                self.client.models.list()
                logger.info("✅ OpenAI LLM initialized successfully")
                self.available = True
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}")
                self.available = False
        else:
            logger.warning("OpenAI client not available")
            self.available = False
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate text from prompt using OpenAI API"""
        if not self.available:
            return "OpenAI LLM not available - check your API key"
        
        try:
            max_tokens = max_tokens or self.config.get("max_tokens", 512)
            temperature = temperature or self.config.get("temperature", 0.7)
            
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional real estate market analyst. Provide accurate, data-driven insights based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return f"Error generating response: {str(e)}"

# Simplified RAG System
class RealEstateRAGSystem:
    """Ultra-lightweight RAG system using OpenAI embeddings"""
    
    def __init__(self, use_llm: bool = True):
        logger.info("Initializing Ultra-Lightweight Real Estate RAG System...")
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(openai_client)
        
        # Initialize simple vector store
        self.vector_store = SimpleVectorStore(self.embeddings)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize LLM if requested
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = OpenAILLM(LLMConfig.GPT_3_5_TURBO, openai_client)
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
        
        # Initialize prompt template
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
        
        logger.info("✅ Ultra-Lightweight RAG System initialized successfully")
    
    def _load_initial_data(self):
        """Load sample real estate data into the system"""
        try:
            sample_docs = [
                Document(
                    page_content="Commercial real estate in Manhattan saw a 15% price increase in Q3 2024, driven by return-to-office mandates and foreign investment. Average price per square foot reached $1,250 in prime locations. Office vacancy rates dropped to 12.3%, the lowest since 2020.",
                    metadata={"source": "Manhattan Market Report", "date": "2024-09-30", "property_type": "commercial", "location": "Manhattan", "url": "https://example.com/manhattan-market-report-q3-2024", "author": "Manhattan Real Estate Institute"}
                ),
                Document(
                    page_content="The industrial real estate sector continues to outperform, with warehouse properties near major ports seeing 8% annual appreciation. E-commerce growth drives demand for last-mile delivery facilities. Cap rates for prime industrial assets range from 4.5% to 5.5%.",
                    metadata={"source": "Industrial Analysis", "date": "2024-10-15", "property_type": "industrial", "location": "National", "url": "https://example.com/industrial-real-estate-analysis-2024", "author": "National Industrial Research Group"}
                ),
                Document(
                    page_content="Residential mortgage rates stabilized at 6.5% in October 2024, leading to a slight uptick in home sales. First-time buyers remain challenged by affordability constraints in major markets. The median home price nationally reached $425,000.",
                    metadata={"source": "Residential Report", "date": "2024-10-20", "property_type": "residential", "location": "National", "url": "https://example.com/residential-market-report-oct-2024", "author": "National Association of Realtors"}
                ),
                Document(
                    page_content="Green building certifications are becoming increasingly important for institutional investors. LEED-certified properties command a 7% premium on average. ESG considerations now factor into 78% of commercial real estate investment decisions.",
                    metadata={"source": "ESG Report", "date": "2024-10-10", "property_type": "commercial", "focus": "sustainability", "url": "https://example.com/esg-real-estate-report-2024", "author": "Green Building Council"}
                ),
                Document(
                    page_content="Retail real estate shows signs of recovery with adaptive reuse projects gaining momentum. Mixed-use developments combining retail, office, and residential are attracting significant investment. Experiential retail concepts are driving foot traffic.",
                    metadata={"source": "Retail Trends", "date": "2024-10-05", "property_type": "retail", "location": "National", "url": "https://example.com/retail-real-estate-trends-2024", "author": "Retail Property Institute"}
                )
            ]
            
            # Process and index documents
            texts = [doc.page_content for doc in sample_docs]
            metadatas = [doc.metadata for doc in sample_docs]
            self.vector_store.add_texts(texts, metadatas)
            
            logger.info(f"Loaded {len(sample_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, use_llm: Optional[bool] = None) -> QueryResult:
        """Process a query through the RAG system"""
        try:
            use_llm = use_llm if use_llm is not None else self.use_llm
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=5, filter=filters)
            
            if not relevant_docs:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    temporal_context=[],
                    related_metrics=[],
                    model_used=None
                )
            
            # Build context
            context = self._build_context(relevant_docs)
            
            # Generate answer
            if use_llm and self.llm and self.llm.available:
                prompt = self.qa_prompt_template.format(context=context, question=question)
                answer = self.llm.generate(prompt, max_tokens=300)
                model_used = self.llm.config["model_name"]
            else:
                answer = self._simple_answer_extraction(question, relevant_docs)
                model_used = "simple_extraction"
            
            # Post-process
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
    
    def _build_context(self, docs: List[Document]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            date = doc.metadata.get('date', 'Unknown')
            context_entry = f"[Document {i} - Source: {source}, Date: {date}]\n{doc.page_content}"
            context_parts.append(context_entry)
        return "\n\n---\n\n".join(context_parts)
    
    def _simple_answer_extraction(self, question: str, docs: List[Document]) -> str:
        """Simple answer extraction when LLM is not available"""
        combined_content = "\n".join([doc.page_content for doc in docs[:3]])
        question_lower = question.lower()
        sentences = combined_content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in question_lower.split() if len(keyword) > 3):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = ". ".join(relevant_sentences[:3]) + "."
            return f"Based on available data: {answer}"
        else:
            return f"Based on the search results, here's what I found: {combined_content[:300]}..."
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Clean and enhance the generated answer"""
        lines = answer.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        return '\n'.join(unique_lines).strip()
    
    def _calculate_confidence(self, docs: List[Document], answer: str) -> float:
        """Calculate confidence score"""
        base_confidence = 0.6
        doc_score = min(len(docs) * 0.1, 0.3)
        answer_score = 0.1 if len(answer) > 100 else 0.0
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
        
        return metrics[:5]
    
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
            metadata = doc.metadata
            source_name = metadata.get('source', 'Unknown Source')
            url = metadata.get('url', None)
            author = metadata.get('author', None)
            date = metadata.get('date', None)
            
            source_entry = {
                "title": source_name,
                "content_preview": doc.page_content[:150] + "...",
                "url": url,
                "author": author,
                "date": date,
                "metadata": metadata
            }
            
            if metadata.get('property_type'):
                source_entry["property_type"] = metadata['property_type']
            if metadata.get('location'):
                source_entry["location"] = metadata['location']
            
            sources.append(source_entry)
        
        return sources

# Initialize RAG system
rag_system = None
try:
    rag_system = RealEstateRAGSystem(use_llm=True)
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    try:
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
        <h1>🏢 Real Estate RAG System - Ultra Lightweight</h1>
        <p>OpenAI-powered system ready for deployment!</p>
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
        "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
        "embeddings_available": rag_system.embeddings.available if rag_system else False,
        "version": "3.0.0-ultralight",
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

# Main execution
if __name__ == "__main__":
    import sys
    
    port = int(os.environ.get("PORT", 8000))
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        print("🌍 Real Estate RAG System - Ultra Lightweight CLI Mode")
        if rag_system:
            print(f"📊 LLM Status: {'Available' if rag_system.llm and rag_system.llm.available else 'Not Available'}")
            print(f"🤖 OpenAI: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not Configured'}")
            print(f"🔍 Embeddings: {'Available' if rag_system.embeddings.available else 'Fallback Mode'}")
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
        print(f"🚀 Starting Ultra-Lightweight Real Estate RAG System on port {port}")
        print(f"📊 LLM Status: {'Available' if rag_system and rag_system.llm and rag_system.llm.available else 'Not Available'}")
        print(f"🤖 OpenAI: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not Configured'}")
        print(f"🔍 Embeddings: {'Available' if rag_system and rag_system.embeddings.available else 'Fallback Mode'}")
        print(f"🌐 Access the dashboard at http://localhost:{port}")
        
        uvicorn.run(
            "real_estate_rag_system_openai_embeddings:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            workers=1
        )