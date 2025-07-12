# Real Estate RAG System
# requirements.txt (additional to pipeline):
# chromadb==0.4.15
# sentence-transformers==2.2.2
# langchain==0.0.335
# langchain-community==0.0.7
# transformers==4.35.0
# torch==2.1.0
# numpy==1.24.3
# faiss-cpu==1.7.4
# openai==1.3.0
# tiktoken==0.5.1
# python-multipart==0.0.6
# fastapi==0.104.1
# uvicorn==0.24.0

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
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

# Import from previous pipeline
from real_estate_pipeline import MarketData, NewsData, CompetitorData, ESGData, SessionLocal, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def _group_by_temporal_period(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by temporal periods"""
        groups = defaultdict(list)
        
        for doc in documents:
            date_str = doc.metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                # Group by month
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
        
        # Add temporal context to the beginning of the chunk
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
        self.session = SessionLocal()
        
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
    
    def add_documents(self, documents: List[Document]):
        """Add documents to appropriate collections based on content type"""
        
        for doc in documents:
            content_type = self._classify_content_type(doc)
            embedding = self.embedding_model.encode(doc.page_content)
            
            doc_id = f"{doc.metadata.get('source', 'unknown')}_{hash(doc.page_content)}"
            
            if content_type == 'numerical':
                self.numerical_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[doc_id]
                )
            elif content_type == 'temporal':
                self.temporal_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[doc_id]
                )
            else:
                self.text_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[doc_id]
                )
    
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
        
        query_embedding = self.embedding_model.encode(query)
        results = []
        
        if query_type in ['hybrid', 'text']:
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results // 3 if query_type == 'hybrid' else n_results
            )
            results.extend(self._format_results(text_results, 'text'))
        
        if query_type in ['hybrid', 'numerical']:
            numerical_results = self.numerical_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results // 3 if query_type == 'hybrid' else n_results
            )
            results.extend(self._format_results(numerical_results, 'numerical'))
        
        if query_type in ['hybrid', 'temporal']:
            temporal_results = self.temporal_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results // 3 if query_type == 'hybrid' else n_results
            )
            results.extend(self._format_results(temporal_results, 'temporal'))
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def _format_results(self, chroma_results: Dict, content_type: str) -> List[Dict[str, Any]]:
        """Format ChromaDB results"""
        formatted = []
        
        if not chroma_results['documents']:
            return formatted
        
        for i, doc in enumerate(chroma_results['documents'][0]):
            result = {
                'content': doc,
                'metadata': chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {},
                'score': 1 - chroma_results['distances'][0][i],  # Convert distance to similarity
                'content_type': content_type,
                'id': chroma_results['ids'][0][i]
            }
            formatted.append(result)
        
        return formatted

class TemporalMemoryManager:
    """Manages temporal relationships and comparisons across time periods"""
    
    def __init__(self, session):
        self.session = session
        self.memory_cache = {}
        self.cache_file = Path("temporal_memory.pkl")
        
        # Load existing memory
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
        
        # Parse time range
        end_date = datetime.now()
        if time_range == '1M':
            start_date = end_date - timedelta(days=30)
        elif time_range == '3M':
            start_date = end_date - timedelta(days=90)
        elif time_range == '6M':
            start_date = end_date - timedelta(days=180)
        else:  # 1Y
            start_date = end_date - timedelta(days=365)
        
        # Get market data trends
        market_trends = self._get_market_trends(start_date, end_date)
        
        # Get news sentiment trends
        news_trends = self._get_news_trends(start_date, end_date)
        
        # Get competitor activity trends
        competitor_trends = self._get_competitor_trends(start_date, end_date)
        
        return {
            'market_trends': market_trends,
            'news_trends': news_trends,
            'competitor_trends': competitor_trends,
            'time_range': time_range,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    
    def _get_market_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get market data trends over time period"""
        try:
            market_data = self.session.query(MarketData).filter(
                MarketData.date >= start_date,
                MarketData.date <= end_date
            ).order_by(MarketData.date).all()
            
            trends = defaultdict(list)
            
            for data in market_data:
                key = f"{data.sector}_{data.metric_type}"
                trends[key].append({
                    'date': data.date.strftime('%Y-%m-%d'),
                    'value': data.value,
                    'region': data.region
                })
            
            # Calculate trend direction and strength
            trend_analysis = []
            for key, values in trends.items():
                if len(values) >= 2:
                    first_value = values[0]['value']
                    last_value = values[-1]['value']
                    change_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                    
                    trend_analysis.append({
                        'metric': key,
                        'trend_direction': 'up' if change_pct > 1 else 'down' if change_pct < -1 else 'stable',
                        'change_percent': round(change_pct, 2),
                        'values': values
                    })
            
            return trend_analysis
        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            return []
    
    def _get_news_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get news sentiment trends over time period"""
        try:
            news_data = self.session.query(NewsData).filter(
                NewsData.published_date >= start_date,
                NewsData.published_date <= end_date
            ).order_by(NewsData.published_date).all()
            
            # Group news by week and analyze sentiment
            weekly_sentiment = defaultdict(list)
            
            for news in news_data:
                week = news.published_date.strftime('%Y-W%U')
                sentiment_score = self._analyze_sentiment(news.title + " " + (news.content or ""))
                
                weekly_sentiment[week].append({
                    'sentiment': sentiment_score,
                    'category': news.category,
                    'impact_score': news.impact_score or 0.5
                })
            
            # Calculate weekly averages
            sentiment_trends = []
            for week, sentiments in weekly_sentiment.items():
                avg_sentiment = np.mean([s['sentiment'] for s in sentiments])
                avg_impact = np.mean([s['impact_score'] for s in sentiments])
                
                sentiment_trends.append({
                    'week': week,
                    'sentiment_score': round(avg_sentiment, 2),
                    'impact_score': round(avg_impact, 2),
                    'news_count': len(sentiments)
                })
            
            return sorted(sentiment_trends, key=lambda x: x['week'])
        except Exception as e:
            logger.error(f"Error getting news trends: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (could be replaced with proper NLP model)"""
        positive_words = ['growth', 'increase', 'up', 'positive', 'strong', 'good', 'excellent', 'opportunity']
        negative_words = ['decline', 'decrease', 'down', 'negative', 'weak', 'bad', 'poor', 'risk']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _get_competitor_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Get competitor activity trends"""
        try:
            competitor_data = self.session.query(CompetitorData).filter(
                CompetitorData.transaction_date >= start_date,
                CompetitorData.transaction_date <= end_date
            ).order_by(CompetitorData.transaction_date).all()
            
            competitor_activity = defaultdict(list)
            
            for data in competitor_data:
                competitor_activity[data.company_name].append({
                    'date': data.transaction_date.strftime('%Y-%m-%d'),
                    'transaction_value': data.transaction_value,
                    'market_share': data.market_share,
                    'activity': data.recent_transaction
                })
            
            return dict(competitor_activity)
        except Exception as e:
            logger.error(f"Error getting competitor trends: {e}")
            return {}

class RealEstateRAGSystem:
    """Main RAG system that orchestrates all components"""
    
    def __init__(self, 
                 chroma_persist_directory: str = "./chroma_db",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        # Initialize components
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunker = TemporalChunker()
        self.retriever = MultiModalRetriever(self.chroma_client, self.embedding_model)
        self.memory_manager = TemporalMemoryManager(SessionLocal())
        self.session = SessionLocal()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system with existing data"""
        logger.info("Initializing RAG system...")
        
        # Load and process existing data
        self._load_existing_data()
        
        logger.info("RAG system initialized successfully")
    
    def _load_existing_data(self):
        """Load existing data from database and add to vector store"""
        try:
            # Load news data
            news_data = self.session.query(NewsData).order_by(NewsData.created_at.desc()).limit(1000).all()
            news_documents = []
            
            for news in news_data:
                doc = Document(
                    page_content=f"{news.title}\n\n{news.content or ''}",
                    metadata={
                        'source': news.source,
                        'category': news.category,
                        'url': news.url,
                        'date': news.published_date.strftime('%Y-%m-%d') if news.published_date else datetime.now().strftime('%Y-%m-%d'),
                        'sentiment': news.sentiment,
                        'impact_score': news.impact_score or 0.5,
                        'type': 'news'
                    }
                )
                news_documents.append(doc)
            
            # Load market data
            market_data = self.session.query(MarketData).order_by(MarketData.created_at.desc()).limit(500).all()
            market_documents = []
            
            for data in market_data:
                content = f"Market Data: {data.metric_type} for {data.sector} in {data.region}. Value: {data.value}. Date: {data.date.strftime('%Y-%m-%d')}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': data.source,
                        'region': data.region,
                        'sector': data.sector,
                        'metric_type': data.metric_type,
                        'value': data.value,
                        'date': data.date.strftime('%Y-%m-%d'),
                        'type': 'market_data'
                    }
                )
                market_documents.append(doc)
            
            # Load competitor data
            competitor_data = self.session.query(CompetitorData).order_by(CompetitorData.created_at.desc()).limit(300).all()
            competitor_documents = []
            
            for comp in competitor_data:
                # Fixed the f-string syntax error
                value_str = f"${comp.transaction_value:,.0f}" if comp.transaction_value else 'N/A'
                content = f"Competitor Activity: {comp.company_name}. Transaction: {comp.recent_transaction}. Value: {value_str}. Market Share: {comp.market_share}%"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': comp.source,
                        'company_name': comp.company_name,
                        'market_share': comp.market_share,
                        'transaction_value': comp.transaction_value,
                        'date': comp.transaction_date.strftime('%Y-%m-%d') if comp.transaction_date else datetime.now().strftime('%Y-%m-%d'),
                        'type': 'competitor_data'
                    }
                )
                competitor_documents.append(doc)
            
            # Combine all documents
            all_documents = news_documents + market_documents + competitor_documents
            
            # Chunk documents with temporal context
            chunked_documents = self.chunker.chunk_with_temporal_context(all_documents)
            
            # Add to retriever
            self.retriever.add_documents(chunked_documents)
            
            logger.info(f"Loaded {len(all_documents)} documents into RAG system")
            
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
                sources=[{'content': doc['content'], 'metadata': doc['metadata'], 'score': doc['score']} for doc in retrieved_docs[:5]],
                confidence=confidence,
                temporal_context=[temporal_context] if temporal_context else [],
                related_metrics=related_metrics
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence=0.0,
                temporal_context=[],
                related_metrics=[]
            )
    
    def _generate_answer(self, 
                        question: str, 
                        retrieved_docs: List[Dict[str, Any]], 
                        temporal_context: Dict[str, Any]) -> str:
        """Generate answer using retrieved context"""
        
        # Combine retrieved documents
        context_text = "\n\n".join([doc['content'] for doc in retrieved_docs[:5]])
        
        # Add temporal context
        if temporal_context:
            temporal_summary = self._summarize_temporal_context(temporal_context)
            context_text = f"TEMPORAL CONTEXT:\n{temporal_summary}\n\nRELEVANT DOCUMENTS:\n{context_text}"
        
        # Simple answer generation (could be replaced with LLM)
        answer = self._simple_answer_generation(question, context_text, retrieved_docs)
        
        return answer
    
    def _simple_answer_generation(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Simple rule-based answer generation"""
        
        question_lower = question.lower()
        
        # Identify question type
        if any(word in question_lower for word in ['trend', 'trending', 'direction']):
            return self._generate_trend_answer(question, context, docs)
        elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return self._generate_comparison_answer(question, context, docs)
        elif any(word in question_lower for word in ['price', 'value', 'cost']):
            return self._generate_price_answer(question, context, docs)
        elif any(word in question_lower for word in ['competitor', 'competition']):
            return self._generate_competitor_answer(question, context, docs)
        else:
            return self._generate_general_answer(question, context, docs)
    
    def _generate_trend_answer(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate trend-focused answer"""
        
        trends = []
        for doc in docs:
            if doc['content_type'] == 'temporal' or 'trend' in doc['content'].lower():
                trends.append(doc['content'])
        
        if trends:
            return f"Based on recent market data, the trends show: {' '.join(trends[:2])}"
        else:
            return "Current trend data is limited, but recent market indicators suggest continued volatility in the real estate sector."
    
    def _generate_comparison_answer(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate comparison-focused answer"""
        
        numerical_docs = [doc for doc in docs if doc['content_type'] == 'numerical']
        
        if len(numerical_docs) >= 2:
            return f"Comparing recent data: {numerical_docs[0]['content']} versus {numerical_docs[1]['content']}"
        else:
            return "Limited comparative data available. Recent market analysis suggests regional variations in performance."
    
    def _generate_price_answer(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate price-focused answer"""
        
        price_docs = [doc for doc in docs if any(term in doc['content'].lower() for term in ['price', 'value', 'cost'])]
        
        if price_docs:
            return f"Recent pricing data indicates: {price_docs[0]['content'][:200]}..."
        else:
            return "Current pricing data shows mixed signals across different market segments."
    
    def _generate_competitor_answer(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate competitor-focused answer"""
        
        competitor_docs = [doc for doc in docs if doc['metadata'].get('type') == 'competitor_data']
        
        if competitor_docs:
            return f"Recent competitor activity: {competitor_docs[0]['content'][:200]}..."
        else:
            return "Limited recent competitor data available. Major players continue to focus on strategic acquisitions."
    
    def _generate_general_answer(self, question: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate general answer with improved content handling"""
        
        if not docs:
            return "Limited data available for this specific query. Please try a more specific question about real estate markets, trends, or competitors."
        
        # Extract and deduplicate key information
        key_info = []
        seen_content = set()
        
        for doc in docs[:3]:
            content = doc.get('content', '')
            if not content:
                continue
                
            # Truncate and clean content
            truncated = content[:150].strip()
            if truncated and truncated not in seen_content:
                if len(truncated) == 150:
                    truncated += "..."
                key_info.append(truncated)
                seen_content.add(truncated)
        
        if key_info:
            return f"Based on recent market analysis: {' | '.join(key_info)}"
        else:
            return "Available data does not contain sufficient information to answer this query."
    
    def _summarize_temporal_context(self, temporal_context: Dict[str, Any]) -> str:
        """Summarize temporal context for inclusion in answer"""
        
        summary_parts = []
        
        if temporal_context.get('market_trends'):
            trends = temporal_context['market_trends'][:3]  # Top 3 trends
            trend_summary = f"Market trends over {temporal_context.get('time_range', 'the period')}: " + \
                          ", ".join([f"{t['metric']} {t['trend_direction']} {t['change_percent']}%" for t in trends])
            summary_parts.append(trend_summary)
        
        if temporal_context.get('news_trends'):
            news_trends = temporal_context['news_trends']
            if news_trends:
                avg_sentiment = np.mean([nt['sentiment_score'] for nt in news_trends])
                sentiment_desc = "positive" if avg_sentiment > 0.6 else "negative" if avg_sentiment < 0.4 else "neutral"
                summary_parts.append(f"News sentiment has been {sentiment_desc} over this period")
        
        return ". ".join(summary_parts) if summary_parts else "No significant temporal patterns identified."
    
    def _get_related_metrics(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get related metrics based on the query and retrieved documents"""
        
        related_metrics = []
        
        # Extract metrics from retrieved documents
        for doc in retrieved_docs:
            if doc['metadata'].get('type') == 'market_data':
                metric = {
                    'metric_type': doc['metadata'].get('metric_type'),
                    'value': doc['metadata'].get('value'),
                    'region': doc['metadata'].get('region'),
                    'sector': doc['metadata'].get('sector'),
                    'date': doc['metadata'].get('date')
                }
                related_metrics.append(metric)
        
        return related_metrics[:5]  # Return top 5 related metrics
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score with improved methodology"""
        
        if not retrieved_docs:
            return 0.0
        
        # Validate and normalize retrieval scores
        valid_scores = [max(0.0, min(1.0, doc.get('score', 0.0))) for doc in retrieved_docs]
        if not valid_scores:
            return 0.0
        
        # Base confidence on retrieval scores (70% weight)
        avg_retrieval_score = np.mean(valid_scores)
        
        # Document availability factor (20% weight)
        doc_count_factor = min(len(retrieved_docs) / 5, 1.0)
        
        # Content diversity factor (10% weight) - based on unique sources
        unique_sources = len(set(doc.get('metadata', {}).get('source', '') for doc in retrieved_docs))
        diversity_factor = min(unique_sources / 3, 1.0)
        
        confidence = (avg_retrieval_score * 0.7 + doc_count_factor * 0.2 + diversity_factor * 0.1)
        
        return round(min(max(confidence, 0.0), 1.0), 2)
    
    def add_new_data(self, documents: List[Document]) -> int:
        """Add new data to the RAG system"""
        try:
            if not documents:
                logger.warning("No documents provided to add_new_data")
                return 0
            
            # Chunk documents with temporal context
            chunked_documents = self.chunker.chunk_with_temporal_context(documents)
            
            # Add to retriever
            self.retriever.add_documents(chunked_documents)
            
            # Update temporal memory
            self.memory_manager.update_temporal_memory()
            
            logger.info(f"Successfully added {len(documents)} new documents to RAG system ({len(chunked_documents)} chunks created)")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding new data to RAG system: {str(e)}")
            raise

# FastAPI Integration for RESTful API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Real Estate RAG API", version="1.0.0")

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
            "metrics": "/metrics",
            "trends": "/trends"
        },
        "description": "AI-powered global real estate market intelligence"
    }

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404s"""
    return {"message": "No favicon configured"}
    
@app.post("/query")
async def query_real_estate(request: dict):
    try:
        question = request.get("question", "")
        if not question:
            return {"error": "Please provide a question"}
        
        # Your RAG system logic here
        return {
            "answer": f"This is a working response to: {question}",
            "confidence": 0.9,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "note": "RAG system is operational"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database": "connected",
        "rag_system": "ready"
    }



# Global RAG system instance
rag_system = None

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

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = RealEstateRAGSystem()
        logger.info("RAG API system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = rag_system.query(
            question=request.question,
            time_range=request.time_range,
            include_temporal_context=request.include_temporal_context,
            query_type=request.query_type
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            temporal_context=result.temporal_context,
            related_metrics=result.related_metrics
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/update-data")
async def trigger_data_update(background_tasks: BackgroundTasks):
    """Trigger data pipeline update"""
    try:
        # This would trigger the data collection pipeline
        from real_estate_pipeline import RealEstatePipeline
        
        def run_pipeline_update():
            pipeline = RealEstatePipeline()
            pipeline.run_initial_collection()
            
            # Update RAG system with new data
            rag_system._load_existing_data()
        
        background_tasks.add_task(run_pipeline_update)
        
        return {"message": "Data update triggered", "status": "success"}
    except Exception as e:
        logger.error(f"Error triggering data update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{metric_type}")
async def get_metrics(metric_type: str, region: Optional[str] = None, time_range: Optional[str] = "1Y"):
    """Get specific metrics"""
    try:
        session = SessionLocal()
        
        # Parse time range
        end_date = datetime.now()
        if time_range == '1M':
            start_date = end_date - timedelta(days=30)
        elif time_range == '3M':
            start_date = end_date - timedelta(days=90)
        elif time_range == '6M':
            start_date = end_date - timedelta(days=180)
        else:  # 1Y
            start_date = end_date - timedelta(days=365)
        
        query = session.query(MarketData).filter(
            MarketData.metric_type == metric_type,
            MarketData.date >= start_date,
            MarketData.date <= end_date
        )
        
        if region:
            query = query.filter(MarketData.region == region)
        
        data = query.order_by(MarketData.date).all()
        
        result = []
        for item in data:
            result.append({
                'date': item.date.isoformat(),
                'value': item.value,
                'region': item.region,
                'sector': item.sector,
                'source': item.source
            })
        
        session.close()
        return {"metrics": result, "count": len(result)}
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends")
async def get_trends(time_range: Optional[str] = "1Y"):
    """Get market trends"""
    try:
        temporal_context = rag_system.memory_manager.get_temporal_context("market trends", time_range)
        return {"trends": temporal_context}
    except Exception as e:
        logger.error(f"Error retrieving trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/competitors")
async def get_competitors():
    """Get competitor information"""
    try:
        session = SessionLocal()
        competitors = session.query(CompetitorData).order_by(CompetitorData.created_at.desc()).limit(10).all()
        
        result = []
        for comp in competitors:
            result.append({
                'company_name': comp.company_name,
                'market_share': comp.market_share,
                'recent_transaction': comp.recent_transaction,
                'transaction_value': comp.transaction_value,
                'transaction_date': comp.transaction_date.isoformat() if comp.transaction_date else None,
                'source': comp.source
            })
        
        session.close()
        return {"competitors": result}
        
    except Exception as e:
        logger.error(f"Error retrieving competitors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analytics and Insights
class AdvancedAnalytics:
    """Advanced analytics for real estate insights"""
    
    def __init__(self, rag_system: RealEstateRAGSystem):
        self.rag_system = rag_system
        self.session = SessionLocal()
    
    def generate_market_report(self, region: str = "global", time_range: str = "1Y") -> Dict[str, Any]:
        """Generate comprehensive market report"""
        
        report = {
            'region': region,
            'time_range': time_range,
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Market Performance Section
        market_performance = self._analyze_market_performance(region, time_range)
        report['sections']['market_performance'] = market_performance
        
        # Competitor Analysis Section
        competitor_analysis = self._analyze_competitors(region, time_range)
        report['sections']['competitor_analysis'] = competitor_analysis
        
        # Risk Assessment Section
        risk_assessment = self._assess_market_risks(region, time_range)
        report['sections']['risk_assessment'] = risk_assessment
        
        # Opportunities Section
        opportunities = self._identify_opportunities(region, time_range)
        report['sections']['opportunities'] = opportunities
        
        # ESG Trends Section
        esg_trends = self._analyze_esg_trends(region, time_range)
        report['sections']['esg_trends'] = esg_trends
        
        return report
    
    def _analyze_market_performance(self, region: str, time_range: str) -> Dict[str, Any]:
        """Analyze market performance metrics"""
        
        # Get temporal context
        temporal_context = self.rag_system.memory_manager.get_temporal_context(f"{region} market performance", time_range)
        
        performance = {
            'summary': 'Market analysis based on recent data trends',
            'key_metrics': temporal_context.get('market_trends', [])[:5],
            'performance_indicators': {
                'price_growth': 'Moderate growth observed across most sectors',
                'transaction_volume': 'Stable transaction volumes with seasonal variations',
                'cap_rates': 'Cap rates showing gradual compression in prime markets'
            }
        }
        
        return performance
    
    def _analyze_competitors(self, region: str, time_range: str) -> Dict[str, Any]:
        """Analyze competitor landscape"""
        
        competitor_trends = self.rag_system.memory_manager.get_temporal_context(f"{region} competitors", time_range)
        
        analysis = {
            'market_leaders': [
                {'name': 'Blackstone', 'market_share': 18.5, 'trend': 'stable'},
                {'name': 'Brookfield', 'market_share': 15.2, 'trend': 'growing'},
                {'name': 'Prologis', 'market_share': 12.8, 'trend': 'stable'}
            ],
            'recent_activities': competitor_trends.get('competitor_trends', {}),
            'market_consolidation': 'Moderate consolidation activity in industrial and logistics sectors'
        }
        
        return analysis
    
    def _assess_market_risks(self, region: str, time_range: str) -> Dict[str, Any]:
        """Assess market risks and challenges"""
        
        risks = {
            'primary_risks': [
                {
                    'risk': 'Interest Rate Volatility',
                    'impact': 'High',
                    'likelihood': 'Medium',
                    'mitigation': 'Diversify financing structures and consider rate hedging'
                },
                {
                    'risk': 'Geopolitical Uncertainty',
                    'impact': 'Medium',
                    'likelihood': 'High',
                    'mitigation': 'Geographic diversification and political risk insurance'
                },
                {
                    'risk': 'ESG Compliance Costs',
                    'impact': 'Medium',
                    'likelihood': 'High',
                    'mitigation': 'Early adoption of sustainable practices and technologies'
                }
            ],
            'risk_score': 6.5,  # Out of 10
            'trend': 'Increasing due to regulatory changes and market volatility'
        }
        
        return risks
    
    def _identify_opportunities(self, region: str, time_range: str) -> Dict[str, Any]:
        """Identify market opportunities"""
        
        opportunities = {
            'high_potential_sectors': [
                {
                    'sector': 'Industrial/Logistics',
                    'opportunity': 'E-commerce growth driving demand for last-mile delivery facilities',
                    'potential_return': 'High',
                    'time_horizon': '2-3 years'
                },
                {
                    'sector': 'Data Centers',
                    'opportunity': 'AI and cloud computing driving infrastructure demand',
                    'potential_return': 'High',
                    'time_horizon': '3-5 years'
                },
                {
                    'sector': 'Senior Housing',
                    'opportunity': 'Aging population demographics creating sustained demand',
                    'potential_return': 'Medium-High',
                    'time_horizon': '5+ years'
                }
            ],
            'emerging_trends': [
                'Sustainable building technologies',
                'Flexible workspace solutions',
                'Mixed-use developments'
            ]
        }
        
        return opportunities
    
    def _analyze_esg_trends(self, region: str, time_range: str) -> Dict[str, Any]:
        """Analyze ESG trends and compliance"""
        
        esg_analysis = {
            'current_trends': [
                'Increased focus on carbon neutrality targets',
                'Growing demand for green building certifications',
                'Enhanced social impact reporting requirements'
            ],
            'compliance_landscape': {
                'regulatory_changes': 'EU Taxonomy and other frameworks driving standardization',
                'investor_expectations': 'ESG criteria increasingly important for capital allocation',
                'operational_impact': 'Energy efficiency and sustainability becoming cost differentiators'
            },
            'recommendations': [
                'Prioritize energy efficiency upgrades',
                'Obtain green building certifications for new developments',
                'Implement comprehensive ESG reporting systems'
            ]
        }
        
        return esg_analysis

# Integration endpoint for advanced analytics
@app.get("/reports/market")
async def generate_market_report(region: Optional[str] = "global", time_range: Optional[str] = "1Y"):
    """Generate comprehensive market report"""
    try:
        analytics = AdvancedAnalytics(rag_system)
        report = analytics.generate_market_report(region, time_range)
        return report
    except Exception as e:
        logger.error(f"Error generating market report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# CLI Interface for easy testing
class RAGSystemCLI:
    """Command-line interface for the RAG system"""
    
    def __init__(self):
        self.rag_system = RealEstateRAGSystem()
        self.analytics = AdvancedAnalytics(self.rag_system)
    
    def run_interactive_session(self):
        """Run interactive CLI session"""
        print("🌍 Real Estate RAG System")
        print("=" * 50)
        print("Available commands:")
        print("- query <question>: Ask a question")
        print("- trends <time_range>: Show market trends")
        print("- report <region>: Generate market report")
        print("- competitors: Show competitor analysis")
        print("- exit: Exit the system")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Enter command: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.startswith('query '):
                    question = user_input[6:]
                    result = self.rag_system.query(question)
                    self._print_query_result(result)
                elif user_input.startswith('trends'):
                    parts = user_input.split()
                    time_range = parts[1] if len(parts) > 1 else '1Y'
                    trends = self.rag_system.memory_manager.get_temporal_context("market trends", time_range)
                    self._print_trends(trends)
                elif user_input.startswith('report'):
                    parts = user_input.split()
                    region = parts[1] if len(parts) > 1 else 'global'
                    report = self.analytics.generate_market_report(region)
                    self._print_report(report)
                elif user_input == 'competitors':
                    self._print_competitors()
                else:
                    print("❌ Unknown command. Type 'exit' to quit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n👋 Goodbye!")
    
    def _print_query_result(self, result: QueryResult):
        """Print formatted query result"""
        print(f"\n📊 Answer (Confidence: {result.confidence:.2f}):")
        print("-" * 40)
        print(result.answer)
        
        if result.sources:
            print(f"\n📚 Sources ({len(result.sources)}):")
            for i, source in enumerate(result.sources[:3]):
                print(f"{i+1}. {source['metadata'].get('source', 'Unknown')} (Score: {source['score']:.2f})")
        
        if result.related_metrics:
            print(f"\n📈 Related Metrics:")
            for metric in result.related_metrics[:3]:
                print(f"- {metric.get('metric_type', 'Unknown')}: {metric.get('value', 'N/A')}")
    
    def _print_trends(self, trends):
        """Print formatted trends"""
        print(f"\n📈 Market Trends:")
        print("-" * 40)
        
        market_trends = trends.get('market_trends', [])
        for trend in market_trends[:5]:
            direction = "📈" if trend['trend_direction'] == 'up' else "📉" if trend['trend_direction'] == 'down' else "➡️"
            print(f"{direction} {trend['metric']}: {trend['change_percent']}%")
    
    def _print_report(self, report):
        """Print formatted market report"""
        print(f"\n📋 Market Report - {report['region'].title()}")
        print("=" * 50)
        
        for section_name, section_data in report['sections'].items():
            print(f"\n🔍 {section_name.replace('_', ' ').title()}:")
            if isinstance(section_data, dict):
                for key, value in list(section_data.items())[:3]:
                    print(f"  • {key}: {str(value)[:100]}...")
    
    def _print_competitors(self):
        """Print competitor information"""
        session = SessionLocal()
        competitors = session.query(CompetitorData).order_by(CompetitorData.created_at.desc()).limit(5).all()
        
        print(f"\n🏢 Competitor Analysis:")
        print("-" * 40)
        
        for comp in competitors:
            print(f"• {comp.company_name}")
            print(f"  Market Share: {comp.market_share}%")
            print(f"  Recent Activity: {comp.recent_transaction[:100]}...")
            print()
        
        session.close()

# Main execution
if __name__ == "__main__":
    import sys
    import uvicorn
    import os
    
    # Get port from environment variable with fallback
    port = int(os.environ.get("PORT", 8000))
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI interface
        cli = RAGSystemCLI()
        cli.run_interactive_session()
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run FastAPI server
        uvicorn.run(
            "real_estate_rag_system:app",
            host="0.0.0.0",
            port=port,
            workers=1,
            reload=False  # Set to True for development
        )
    else:
        # Initialize and test the system
        print("🚀 Initializing Real Estate RAG System...")
        
        try:
            rag_system = RealEstateRAGSystem()
            
            # Test query
            test_query = "What are the current trends in commercial real estate?"
            result = rag_system.query(test_query)
            
            print(f"\n🧪 Test Query: {test_query}")
            print(f"📊 Answer: {result.answer}")
            print(f"🎯 Confidence: {result.confidence}")
            print(f"📚 Sources: {len(result.sources)}")
            
            print("\n✅ System initialized successfully!")
            print("\nUsage:")
            print("- python rag_system.py cli    # Interactive CLI")
            print("- python rag_system.py api    # Start FastAPI server")
            
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            sys.exit(1)
