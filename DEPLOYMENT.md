# 🚀 Deployment Guide

## Quick Deploy

### Local (Development)
```bash
pip install -r requirements.txt
python setup_kaggle.py
streamlit run streamlit_app.py
```
Done! Open `http://localhost:8501`

---

## Cloud Deployment

### Streamlit Cloud (Easiest - Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo → Set main file: `streamlit_app.py`
5. Add secrets (API keys) in settings
6. Deploy!

Secrets format:
```toml
WAQI_API_KEY = "your_key_here"
```

### Heroku

```bash
# Install Heroku CLI, then:
heroku create your-app-name
git push heroku main
```

Add `Procfile`:
```
web: streamlit run streamlit_app.py --server.port $PORT
```

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run streamlit_app.py --server.port 8501
```

Run:
```bash
docker build -t aqi-app .
docker run -p 8501:8501 aqi-app
```

---

## Production Settings

### Environment Variables
Create `.env` file:
```bash
WAQI_API_KEY=your_key
LOG_LEVEL=INFO
CACHE_TTL=300
```

### Performance
- Enable caching: Already built-in ✅
- Database indexing: Already optimized ✅  
- API rate limiting: Handled automatically ✅

---

## Requirements

Minimum:
- Python 3.9+
- 2GB RAM
- 1GB disk space

Recommended:
- 4GB RAM (for ML models)
- SSD storage
- Stable internet

---

## Health Checks

The app auto-validates:
- Database connection
- API availability
- Model loading
- Sample data fallback

---

## Monitoring

Built-in logging tracks:
- API calls & response times
- Errors & warnings
- User queries
- Data source status

Check logs in terminal or configure external monitoring.

---

That's it! Simple deployment, powerful features. 🎉
- Automatic HTTPS
- Easy deployment
- Automatic updates from GitHub

Cons:
- Limited resources on free tier
- Cold start delays
- Public by default

---

### 2. Heroku

Setup:
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.9.18" > runtime.txt

# Login to Heroku
heroku login

# Create app
heroku create your-aqi-app

# Set config vars
heroku config:set WAQI_API_KEY=your_key
heroku config:set OPENWEATHER_API_KEY=your_key

# Deploy
git push heroku main

# Open app
heroku open
```

Pros:
- Easy scaling
- Add-ons ecosystem
- Good documentation

Cons:
- Free tier has sleep time
- Paid tiers can be expensive

---

### 3. AWS EC2

Launch Instance:
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3.9 python3.9-venv python3-pip -y

# Clone repository
git clone <your-repo-url>
cd aqi-health-system

# Setup virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WAQI_API_KEY=your_key
export OPENWEATHER_API_KEY=your_key

# Run with nohup
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```

Using Systemd Service:
```bash
# Create service file
sudo nano /etc/systemd/system/aqi-app.service
```

```ini
[Unit]
Description=AQI Health App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/aqi-health-system
Environment="PATH=/home/ubuntu/aqi-health-system/venv/bin"
Environment="WAQI_API_KEY=your_key"
Environment="OPENWEATHER_API_KEY=your_key"
ExecStart=/home/ubuntu/aqi-health-system/venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl start aqi-app
sudo systemctl enable aqi-app
sudo systemctl status aqi-app
```

---

### 4. Azure App Service

Using Azure CLI:
```bash
# Login
az login

# Create resource group
az group create --name aqi-rg --location eastus

# Create App Service plan
az appservice plan create --name aqi-plan --resource-group aqi-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group aqi-rg --plan aqi-plan --name your-aqi-app --runtime "PYTHON:3.9"

# Configure deployment
az webapp deployment source config --name your-aqi-app --resource-group aqi-rg --repo-url <your-github-url> --branch main

# Set environment variables
az webapp config appsettings set --name your-aqi-app --resource-group aqi-rg --settings WAQI_API_KEY=your_key

# Configure startup command
az webapp config set --name your-aqi-app --resource-group aqi-rg --startup-file "streamlit run streamlit_app.py --server.port 8000"
```

---

### 5. Google Cloud Run

Using Docker:
```bash
# Build and push image
gcloud builds submit --tag gcr.io/your-project/aqi-app

# Deploy to Cloud Run
gcloud run deploy aqi-app \
    --image gcr.io/your-project/aqi-app \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars WAQI_API_KEY=your_key,OPENWEATHER_API_KEY=your_key \
    --memory 2Gi \
    --cpu 2
```

---

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory
RUN mkdir -p data models/cache

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  aqi-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - WAQI_API_KEY=${WAQI_API_KEY}
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
      - ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Deploy:
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Performance Tuning

### 1. Model Optimization
```python
# In utils/config.py
ML_CONFIG = {
    'cache_models': True,  # Cache loaded models
    'predictor': {
        'n_estimators': 50,  # Reduce for faster predictions
    }
}
```

### 2. Database Optimization
```bash
# Regularly clean old data
# Add to crontab
0 0 * * * python -c "from utils.data_manager import get_data_manager; get_data_manager().cleanup_old_data(30)"
```

### 3. Caching Strategy
```python
# Increase cache TTL for stable locations
CACHE_TTL=600  # 10 minutes instead of 5
```

### 4. NLP Model Selection
```python
# Use lighter models for faster responses
NLP_CONFIG = {
    'generator_model': 'distilgpt2',  # Lightweight
    # Disable heavy models in production if not needed
}
```

### 5. Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## Monitoring & Maintenance

### 1. Logging Setup
```python
# Enhanced logging configuration
import logging.handlers

handler = logging.handlers.RotatingFileHandler(
    'logs/app.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler]
)
```

### 2. Health Checks
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check model status
curl http://localhost:8501/api/health
```

### 3. Database Maintenance
```bash
# Backup database
cp data/aqi_history.db data/aqi_history_backup_$(date +%Y%m%d).db

# Optimize database
sqlite3 data/aqi_history.db "VACUUM;"
```

### 4. Monitoring Tools
- Application Performance: New Relic, Datadog
- Uptime Monitoring: UptimeRobot, Pingdom
- Log Analysis: ELK Stack, Splunk
- Error Tracking: Sentry

### 5. Backup Strategy
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backups/aqi_backup_$DATE.tar.gz data/ models/cache/

# Keep last 7 days
find backups/ -name "aqi_backup_*.tar.gz" -mtime +7 -delete
```

---

## Security Best Practices

1. Environment Variables: Never commit `.env` files
2. API Keys: Use secrets management (AWS Secrets Manager, Azure Key Vault)
3. HTTPS: Always use SSL/TLS in production
4. Rate Limiting: Implement rate limiting for API endpoints
5. Input Validation: Sanitize all user inputs
6. Updates: Regularly update dependencies

```bash
# Check for security vulnerabilities
pip install safety
safety check
```

---

## Troubleshooting

### Common Issues

1. Models not loading
```bash
# Clear cache and reinstall
pip install --force-reinstall transformers torch
```

2. Database locked
```bash
# Check database connections
lsof data/aqi_history.db
```

3. Memory issues
```bash
# Monitor memory usage
htop

# Reduce model size or disable NLP
export FEATURES__ENABLE_NLP=false
```

4. API rate limits
```bash
# Increase cache TTL
export CACHE_TTL=900  # 15 minutes
```

---

## Support

For deployment issues:
- Check logs in `logs/` directory
- Review [GitHub Issues](https://github.com/your-repo/issues)
- Contact: your-email@example.com

---

Last Updated: January 2026
