# Real Estate Data Collection Pipeline
# requirements.txt:
# scrapy==2.11.0
# requests==2.31.0
# beautifulsoup4==4.12.2
# pandas==2.0.3
# sqlalchemy==2.0.19
# psycopg2-binary==2.9.7
# celery==5.3.1
# redis==4.6.0
# python-dotenv==1.0.0
# feedparser==6.0.10
# yfinance==0.2.21

import scrapy
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import feedparser
import yfinance as yf
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Database Models
Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    region = Column(String(50))
    sector = Column(String(50))  # residential, commercial, industrial, retail
    metric_type = Column(String(50))  # price_index, cap_rate, vacancy_rate, etc.
    value = Column(Float)
    date = Column(DateTime)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class CompetitorData(Base):
    __tablename__ = 'competitor_data'
    
    id = Column(Integer, primary_key=True)
    company_name = Column(String(100))
    market_share = Column(Float)
    recent_transaction = Column(Text)
    transaction_value = Column(Float)
    transaction_date = Column(DateTime)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class NewsData(Base):
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    content = Column(Text)
    url = Column(String(500))
    published_date = Column(DateTime)
    source = Column(String(100))
    category = Column(String(50))
    impact_score = Column(Float)  # ML-generated impact score
    sentiment = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class ESGData(Base):
    __tablename__ = 'esg_data'
    
    id = Column(Integer, primary_key=True)
    company_name = Column(String(100))
    esg_category = Column(String(50))
    score = Column(Float)
    metric_name = Column(String(100))
    date = Column(DateTime)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

# Configuration
@dataclass
class Config:
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/realestate')
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    NEWS_API_KEY: str = os.getenv('NEWS_API_KEY', '')
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
config = Config()

# Database setup
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Celery setup
celery_app = Celery('real_estate_pipeline', broker=config.REDIS_URL)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Collection Classes

class RealEstateSpider(scrapy.Spider):
    name = 'real_estate_spider'
    
    def __init__(self):
        self.session = SessionLocal()
        
    def start_requests(self):
        urls = [
            'https://www.reit.com/data-research/reit-market-data',
            'https://www.cbre.com/insights/reports',
            'https://www.jll.com/en/trends-and-insights/research',
            'https://www.cushmanwakefield.com/en/insights',
        ]
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        # Extract market data from various real estate websites
        if 'reit.com' in response.url:
            yield from self.parse_reit_data(response)
        elif 'cbre.com' in response.url:
            yield from self.parse_cbre_data(response)
        elif 'jll.com' in response.url:
            yield from self.parse_jll_data(response)
        elif 'cushmanwakefield.com' in response.url:
            yield from self.parse_cushman_data(response)
    
    def parse_reit_data(self, response):
        # Parse REIT market data
        tables = response.css('table.market-data')
        for table in tables:
            rows = table.css('tr')
            for row in rows[1:]:  # Skip header
                cells = row.css('td::text').getall()
                if len(cells) >= 4:
                    yield {
                        'type': 'market_data',
                        'sector': cells[0].strip(),
                        'metric_type': 'price_index',
                        'value': float(cells[1].replace('%', '').replace(',', '')),
                        'region': 'USA',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'reit.com'
                    }
    
    def parse_cbre_data(self, response):
        # Parse CBRE research reports
        reports = response.css('.report-item')
        for report in reports:
            title = report.css('.report-title::text').get()
            url = report.css('a::attr(href)').get()
            date = report.css('.report-date::text').get()
            
            if title and url:
                yield {
                    'type': 'news_data',
                    'title': title.strip(),
                    'url': response.urljoin(url),
                    'published_date': self.parse_date(date),
                    'source': 'CBRE',
                    'category': 'Research Report'
                }
    
    def parse_jll_data(self, response):
        # Parse JLL insights
        insights = response.css('.insight-card')
        for insight in insights:
            title = insight.css('.insight-title::text').get()
            summary = insight.css('.insight-summary::text').get()
            url = insight.css('a::attr(href)').get()
            
            if title and url:
                yield {
                    'type': 'news_data',
                    'title': title.strip(),
                    'content': summary.strip() if summary else '',
                    'url': response.urljoin(url),
                    'published_date': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'JLL',
                    'category': 'Market Insight'
                }
    
    def parse_cushman_data(self, response):
        # Parse Cushman & Wakefield insights
        articles = response.css('.article-card')
        for article in articles:
            title = article.css('.article-title::text').get()
            url = article.css('a::attr(href)').get()
            
            if title and url:
                yield {
                    'type': 'news_data',
                    'title': title.strip(),
                    'url': response.urljoin(url),
                    'published_date': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'Cushman & Wakefield',
                    'category': 'Market Analysis'
                }
    
    def parse_date(self, date_str):
        # Parse various date formats
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Try different date formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

class NewsCollector:
    def __init__(self):
        self.session = SessionLocal()
        self.news_api_key = config.NEWS_API_KEY
        
    async def collect_news(self):
        """Collect real estate news from multiple sources"""
        
        # News API
        if self.news_api_key:
            await self.collect_from_news_api()
        
        # RSS Feeds
        await self.collect_from_rss_feeds()
        
        # Financial news
        await self.collect_financial_news()
    
    async def collect_from_news_api(self):
        """Collect from News API"""
        url = f"https://newsapi.org/v2/everything?q=real%20estate&apiKey={self.news_api_key}&sortBy=publishedAt&pageSize=100"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                
                for article in data.get('articles', []):
                    news_item = NewsData(
                        title=article.get('title', ''),
                        content=article.get('description', ''),
                        url=article.get('url', ''),
                        published_date=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                        source=article.get('source', {}).get('name', 'NewsAPI'),
                        category='General News'
                    )
                    self.session.add(news_item)
                
                self.session.commit()
    
    async def collect_from_rss_feeds(self):
        """Collect from RSS feeds"""
        rss_feeds = [
            'https://www.bisnow.com/rss',
            'https://www.globest.com/rss.xml',
            'https://therealdeal.com/feed/',
            'https://www.cpexecutive.com/feed/',
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    news_item = NewsData(
                        title=entry.get('title', ''),
                        content=entry.get('summary', ''),
                        url=entry.get('link', ''),
                        published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                        source=feed.feed.get('title', 'RSS Feed'),
                        category='Industry News'
                    )
                    self.session.add(news_item)
                
                self.session.commit()
            except Exception as e:
                logger.error(f"Error collecting from RSS feed {feed_url}: {str(e)}")
    
    async def collect_financial_news(self):
        """Collect financial news affecting real estate"""
        keywords = ['interest rates', 'federal reserve', 'real estate investment', 'REIT', 'commercial real estate']
        
        for keyword in keywords:
            if self.news_api_key:
                url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={self.news_api_key}&sortBy=publishedAt&pageSize=20"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            news_item = NewsData(
                                title=article.get('title', ''),
                                content=article.get('description', ''),
                                url=article.get('url', ''),
                                published_date=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                                source=article.get('source', {}).get('name', 'NewsAPI'),
                                category='Financial News'
                            )
                            self.session.add(news_item)
                        
                        self.session.commit()

class MarketDataCollector:
    def __init__(self):
        self.session = SessionLocal()
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
    
    async def collect_reit_data(self):
        """Collect REIT market data"""
        reit_symbols = ['VNQ', 'IYR', 'SCHH', 'FREL', 'RWR', 'XLRE']
        
        for symbol in reit_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                
                for date, row in hist.iterrows():
                    market_data = MarketData(
                        region='USA',
                        sector='REIT',
                        metric_type='price_index',
                        value=float(row['Close']),
                        date=date,
                        source=f'Yahoo Finance - {symbol}'
                    )
                    self.session.add(market_data)
                
                self.session.commit()
            except Exception as e:
                logger.error(f"Error collecting REIT data for {symbol}: {str(e)}")
    
    async def collect_economic_indicators(self):
        """Collect economic indicators affecting real estate"""
        if not self.alpha_vantage_key:
            return
        
        indicators = [
            ('FEDFUNDS', 'interest_rate'),
            ('MORTGAGE15US', 'mortgage_rate_15y'),
            ('MORTGAGE30US', 'mortgage_rate_30y'),
            ('CSUSHPISA', 'housing_price_index'),
        ]
        
        for indicator, metric_type in indicators:
            url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={self.alpha_vantage_key}&datatype=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    for record in data.get('data', []):
                        market_data = MarketData(
                            region='USA',
                            sector='Economic Indicator',
                            metric_type=metric_type,
                            value=float(record.get('value', 0)),
                            date=datetime.strptime(record.get('date'), '%Y-%m-%d'),
                            source='Alpha Vantage'
                        )
                        self.session.add(market_data)
                    
                    self.session.commit()

class CompetitorTracker:
    def __init__(self):
        self.session = SessionLocal()
        self.competitors = [
            'Blackstone Inc.',
            'Brookfield Asset Management',
            'Prologis Inc.',
            'CBRE Group Inc.',
            'Jones Lang LaSalle',
            'Cushman & Wakefield'
        ]
    
    async def track_competitors(self):
        """Track competitor activities and market moves"""
        for competitor in self.competitors:
            await self.collect_competitor_news(competitor)
            await self.collect_competitor_financials(competitor)
    
    async def collect_competitor_news(self, competitor):
        """Collect news about specific competitors"""
        if not config.NEWS_API_KEY:
            return
        
        url = f"https://newsapi.org/v2/everything?q={competitor}&apiKey={config.NEWS_API_KEY}&sortBy=publishedAt&pageSize=10"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                
                for article in data.get('articles', []):
                    # Extract transaction information using basic NLP
                    content = article.get('description', '')
                    transaction_value = self.extract_transaction_value(content)
                    
                    competitor_data = CompetitorData(
                        company_name=competitor,
                        recent_transaction=content,
                        transaction_value=transaction_value,
                        transaction_date=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                        source=article.get('source', {}).get('name', 'NewsAPI')
                    )
                    self.session.add(competitor_data)
                
                self.session.commit()
    
    def extract_transaction_value(self, text):
        """Extract transaction values from text using regex"""
        import re
        
        # Look for patterns like $1.5B, $500M, etc.
        patterns = [
            r'\$(\d+\.?\d*)\s*[Bb]illion',
            r'\$(\d+\.?\d*)\s*[Mm]illion',
            r'\$(\d+\.?\d*)\s*B',
            r'\$(\d+\.?\d*)\s*M',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if 'B' in match.group(0) or 'billion' in match.group(0).lower():
                    return value * 1000000000
                elif 'M' in match.group(0) or 'million' in match.group(0).lower():
                    return value * 1000000
        
        return None
    
    async def collect_competitor_financials(self, competitor):
        """Collect financial data for competitors"""
        # Map competitor names to stock symbols
        symbol_map = {
            'Blackstone Inc.': 'BX',
            'Brookfield Asset Management': 'BAM',
            'Prologis Inc.': 'PLD',
            'CBRE Group Inc.': 'CBRE',
            'Jones Lang LaSalle': 'JLL',
            'Cushman & Wakefield': 'CWK'
        }
        
        symbol = symbol_map.get(competitor)
        if not symbol:
            return
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Store market cap as market share proxy
            market_cap = info.get('marketCap', 0)
            if market_cap:
                competitor_data = CompetitorData(
                    company_name=competitor,
                    market_share=market_cap / 1000000000,  # Convert to billions
                    recent_transaction=f"Market Cap: ${market_cap:,.0f}",
                    transaction_value=market_cap,
                    transaction_date=datetime.now(),
                    source=f'Yahoo Finance - {symbol}'
                )
                self.session.add(competitor_data)
                self.session.commit()
        except Exception as e:
            logger.error(f"Error collecting financial data for {competitor}: {str(e)}")

# Celery Tasks
@celery_app.task
def collect_news_task():
    """Celery task to collect news"""
    collector = NewsCollector()
    asyncio.run(collector.collect_news())

@celery_app.task
def collect_market_data_task():
    """Celery task to collect market data"""
    collector = MarketDataCollector()
    asyncio.run(collector.collect_reit_data())
    asyncio.run(collector.collect_economic_indicators())

@celery_app.task
def track_competitors_task():
    """Celery task to track competitors"""
    tracker = CompetitorTracker()
    asyncio.run(tracker.track_competitors())

@celery_app.task
def run_scrapy_spider():
    """Run Scrapy spider as Celery task"""
    from scrapy.crawler import CrawlerProcess
    
    process = CrawlerProcess()
    process.crawl(RealEstateSpider)
    process.start()

# Scheduler
class DataPipelineScheduler:
    def __init__(self):
        self.celery_app = celery_app
    
    def schedule_tasks(self):
        """Schedule recurring tasks"""
        
        # Schedule news collection every 4 hours
        self.celery_app.conf.beat_schedule = {
            'collect-news': {
                'task': 'collect_news_task',
                'schedule': 4 * 60 * 60,  # 4 hours
            },
            'collect-market-data': {
                'task': 'collect_market_data_task',
                'schedule': 6 * 60 * 60,  # 6 hours
            },
            'track-competitors': {
                'task': 'track_competitors_task',
                'schedule': 12 * 60 * 60,  # 12 hours
            },
            'run-scrapy-spider': {
                'task': 'run_scrapy_spider',
                'schedule': 24 * 60 * 60,  # 24 hours
            },
        }
        
        self.celery_app.conf.timezone = 'UTC'

# Data Quality and Validation
class DataValidator:
    def __init__(self):
        self.session = SessionLocal()
    
    def validate_market_data(self):
        """Validate market data for anomalies"""
        # Check for extreme values
        recent_data = self.session.query(MarketData).filter(
            MarketData.date >= datetime.now() - timedelta(days=30)
        ).all()
        
        anomalies = []
        for data in recent_data:
            if data.metric_type == 'price_index' and (data.value < 0 or data.value > 1000):
                anomalies.append(f"Extreme price index value: {data.value}")
            elif data.metric_type == 'cap_rate' and (data.value < 0 or data.value > 20):
                anomalies.append(f"Extreme cap rate: {data.value}")
        
        return anomalies
    
    def deduplicate_data(self):
        """Remove duplicate entries"""
        # Remove duplicate news articles
        duplicates = self.session.query(NewsData).filter(
            NewsData.url.in_(
                self.session.query(NewsData.url).group_by(NewsData.url).having(
                    func.count(NewsData.url) > 1
                )
            )
        ).all()
        
        for duplicate in duplicates[1:]:  # Keep first occurrence
            self.session.delete(duplicate)
        
        self.session.commit()

# Main Pipeline Controller
class RealEstatePipeline:
    def __init__(self):
        self.scheduler = DataPipelineScheduler()
        self.validator = DataValidator()
    
    def start_pipeline(self):
        """Start the complete data pipeline"""
        logger.info("Starting Real Estate Data Pipeline...")
        
        # Schedule recurring tasks
        self.scheduler.schedule_tasks()
        
        # Run initial data collection
        self.run_initial_collection()
        
        logger.info("Pipeline started successfully!")
    
    def run_initial_collection(self):
        """Run initial data collection"""
        # Collect news
        collector = NewsCollector()
        asyncio.run(collector.collect_news())
        
        # Collect market data
        market_collector = MarketDataCollector()
        asyncio.run(market_collector.collect_reit_data())
        
        # Track competitors
        competitor_tracker = CompetitorTracker()
        asyncio.run(competitor_tracker.track_competitors())
        
        # Validate data
        anomalies = self.validator.validate_market_data()
        if anomalies:
            logger.warning(f"Data anomalies detected: {anomalies}")
        
        self.validator.deduplicate_data()

if __name__ == "__main__":
    pipeline = RealEstatePipeline()
    pipeline.start_pipeline()
