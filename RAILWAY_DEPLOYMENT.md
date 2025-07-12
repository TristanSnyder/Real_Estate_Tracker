# 🚀 Railway Deployment Guide for Real Estate RAG System

This guide will help you deploy the Real Estate RAG System to Railway with full dashboard functionality and Hugging Face integration.

## 📋 Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
3. **Git Repository**: Your code should be in a Git repository

## 🔑 Getting Your Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "Railway RAG System")
4. Select scope: **Read** (sufficient for model downloads)
5. Copy the token (it looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

## 🚂 Railway Deployment Steps

### Step 1: Connect Your Repository

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository

### Step 2: Configure Environment Variables

In your Railway project dashboard, go to **Variables** tab and add:

```bash
# Required: Hugging Face Authentication
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Database (Railway will auto-provide if you add PostgreSQL service)
# DATABASE_URL=postgresql://user:password@host:port/database

# Port (Railway auto-configures this)
PORT=8000
```

### Step 3: Add Database (Optional but Recommended)

1. In your Railway project, click "Add Service"
2. Select "PostgreSQL"
3. Railway will automatically set the `DATABASE_URL` environment variable

### Step 4: Deploy

1. Railway will automatically detect your Python application
2. It will install dependencies from `requirements.txt`
3. The app will start automatically using the command in your Python file

## 🌐 Accessing Your Dashboard

After deployment:

1. Railway will provide a public URL (e.g., `https://your-app-name.up.railway.app`)
2. Visit this URL to see your beautiful dashboard
3. The system will show status indicators for all components

## 🔧 Environment Variables Explained

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `HUGGINGFACE_HUB_TOKEN` | Your HF token for model access | **Yes** | `hf_abcd1234...` |
| `HF_TOKEN` | Alternative name for HF token | No | `hf_abcd1234...` |
| `DATABASE_URL` | PostgreSQL connection string | No | Auto-set by Railway |
| `PORT` | Application port | No | Auto-set by Railway |

## 🎯 Features Available After Deployment

### ✅ Dashboard Features
- **Real-time Status**: System health monitoring
- **AI Chat Interface**: Ask questions about real estate
- **Market Metrics**: Live market data display
- **Competitor Analysis**: Top player insights
- **Beautiful UI**: Modern glassmorphism design

### ✅ API Endpoints
- `GET /` - Main dashboard
- `GET /health` - System status check
- `POST /query` - Submit queries to AI
- `GET /metrics` - Market metrics
- `GET /competitors` - Competitor data
- `POST /add_documents` - Add new data

### ✅ AI Capabilities
- **LLM Integration**: Flan-T5 models for intelligent responses
- **RAG System**: Retrieval-augmented generation
- **Vector Search**: Semantic similarity search
- **Temporal Analysis**: Time-aware document processing

## 🛠️ Troubleshooting

### Issue: HuggingFace Authentication Failed
**Solution**: Check your token in Railway Variables tab
```bash
# Verify token format
HUGGINGFACE_HUB_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890
```

### Issue: Dashboard Not Loading
**Solution**: Check Railway logs for startup errors
1. Go to Railway project → Deployments
2. Click on latest deployment
3. Check logs for error messages

### Issue: LLM Not Available
**Symptoms**: System shows "Limited Mode" status
**Solutions**:
1. Verify HuggingFace token is set
2. Check Railway logs for model loading errors
3. System will fallback to simple text extraction

### Issue: Database Connection Failed
**Solution**: Add PostgreSQL service in Railway
1. Project dashboard → Add Service → PostgreSQL
2. Railway auto-configures `DATABASE_URL`

## 📊 Monitoring Your Deployment

### Health Check
Visit `https://your-app.up.railway.app/health` to see:
```json
{
  "status": "healthy",
  "database_available": true,
  "rag_system": "ready",
  "llm_status": "ready",
  "hf_token_configured": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Dashboard Status Indicators
- 🟢 **Green**: All systems operational
- 🟡 **Yellow**: Limited mode (basic functionality)
- 🔴 **Red**: System issues

## 🔄 Updating Your Deployment

1. Push changes to your Git repository
2. Railway automatically detects changes
3. New deployment starts automatically
4. Zero-downtime deployment

## 💡 Performance Tips

1. **Model Selection**: The system uses Flan-T5-Base by default (faster)
2. **Scaling**: Railway auto-scales based on usage
3. **Caching**: Vector embeddings are cached automatically
4. **Database**: PostgreSQL recommended for production

## 📞 Support

If you encounter issues:

1. **Railway**: Check [Railway Docs](https://docs.railway.app)
2. **HuggingFace**: Check [HF Docs](https://huggingface.co/docs)
3. **Application Logs**: Available in Railway dashboard

## 🎉 Success!

Once deployed, you'll have a fully functional AI-powered real estate analysis system with:

- Beautiful web dashboard
- Intelligent question answering
- Real-time market data
- Professional UI/UX
- Scalable cloud infrastructure

**Your system is now live and ready for real estate market analysis! 🏢✨**