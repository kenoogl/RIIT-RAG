# RAG System API

REST API for the Supercomputer Support RAG (Retrieval-Augmented Generation) System.

## Quick Start

### 1. Start the API Server

```bash
# Using the CLI script
python run_api.py

# Or with custom settings
python run_api.py --host 0.0.0.0 --port 8080 --debug

# Or directly
python -m src.api.main
```

### 2. Verify the Server

```bash
# Check health
curl http://localhost:8000/health

# Or run the demo
python demo_api.py
```

### 3. Interactive Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## API Endpoints

### Core Endpoints

#### `GET /` - API Information
Returns basic API information and version.

#### `GET /health` - Health Check
Returns the health status of all services.

```json
{
  "status": "healthy",
  "services": {
    "rag_service": "healthy",
    "document_engine": "healthy",
    "search_engine": "healthy",
    "generative_model": "healthy"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "0.1.0"
}
```

#### `POST /ask` - Ask Question
Submit a question to the RAG system.

**Request:**
```json
{
  "question": "九州大学のスパコンの使い方を教えてください",
  "max_results": 5,
  "min_confidence": 0.5
}
```

**Response:**
```json
{
  "answer": "九州大学のスーパーコンピュータを利用するには...",
  "sources": ["https://www.cc.kyushu-u.ac.jp/scp/guide1"],
  "confidence": 0.85,
  "processing_time": 1.2,
  "search_results_count": 3,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Document Management

#### `POST /documents/process` - Process Documents
Trigger document processing and indexing.

**Request:**
```json
{
  "url": "https://example.com/docs",
  "force_refresh": false
}
```

**Response:**
```json
{
  "status": "processing",
  "message": "Document processing started",
  "documents_processed": 0,
  "processing_time": 0.1,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### `GET /documents/status` - Document Status
Get the status of document processing.

## Configuration

The API uses the same `config.yaml` file as the rest of the system. Key settings:

```yaml
app:
  host: "0.0.0.0"
  port: 8000
  debug: false

search:
  max_results: 5
  min_similarity_score: 0.5
  enable_search_analytics: true
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `422` - Validation Error (invalid request parameters)
- `500` - Internal Server Error
- `503` - Service Unavailable (services not initialized)

Error responses include detailed information:

```json
{
  "detail": "Answer confidence (0.30) below minimum threshold (0.50)"
}
```

## Client Examples

### Python with requests

```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", json={
    "question": "スパコンの使い方は？",
    "max_results": 3
})

if response.status_code == 200:
    answer = response.json()
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'スパコンの使い方は？',
    max_results: 3
  })
});

const answer = await response.json();
console.log('Answer:', answer.answer);
```

### curl

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "スパコンの使い方は？",
    "max_results": 3
  }'
```

## Offline Operation

The API is designed to work completely offline:

- No external API dependencies
- Local models for embedding and generation
- Local vector database storage
- Self-contained document processing

Verify offline operation:

```bash
python verify_offline.py
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run API tests only
python -m pytest tests/test_api.py -v
```

### Development Mode

```bash
# Enable auto-reload for development
python run_api.py --reload --debug
```

### Adding New Endpoints

1. Add endpoint to `src/api/main.py`
2. Add request/response models using Pydantic
3. Add tests in `tests/test_api.py`
4. Update this documentation

## Deployment

### Production Settings

```yaml
app:
  host: "0.0.0.0"
  port: 8000
  debug: false

logging:
  level: "INFO"
  enable_file: true
```

### Docker (Future)

```dockerfile
# Dockerfile example for future deployment
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "run_api.py"]
```

## Monitoring

The API includes built-in monitoring:

- Health check endpoint
- Request/response logging
- Performance metrics
- Search analytics

Access metrics through the logging service or health endpoint.

## Security Considerations

For production deployment:

1. Configure CORS appropriately
2. Add authentication/authorization
3. Use HTTPS
4. Rate limiting
5. Input validation and sanitization

## Support

For issues and questions:

1. Check the logs in `./logs/`
2. Run the verification script: `python verify_offline.py`
3. Check the health endpoint: `GET /health`
4. Review the interactive docs: http://localhost:8000/docs