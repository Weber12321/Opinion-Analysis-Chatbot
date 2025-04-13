# Opinion Analysis Chatbot

A specialized chatbot that answers questions related to opinion analysis by scraping and analyzing web news articles. The system provides comprehensive analysis including sentiment analysis, named entity recognition, and summaries.

## Features

- **Focused on Opinion Analysis**: Only responds to queries related to public sentiment, market research, and news analysis
- **Web News Analysis**: Scrapes and analyzes 1-10 relevant news articles based on user queries
- **Comprehensive Analysis**: Provides article titles, publication dates, content, summaries, sentiment analysis, and named entity recognition
- **Dockerized Architecture**: opinion_analysis (FastAPI + LangGraph) and Frontend (Streamlit) as separate containers

## Project Structure

```
Opinion-Analysis-Chatbot/
├── opinion_analysis/                  # FastAPI opinion_analysis service
│   ├── Dockerfile
│   ├── main.py               # FastAPI entry point
│   ├── requirements.txt
│   └── app/
│       ├── agents/
│       │   └── opinion_analysis_agent.py  # LangGraph agent
│       └── services/
│           ├── llm_service.py            # Gemini/OpenAI integration
│           ├── news_scraper.py           # Web news scraper
│           ├── ner_service.py            # Named Entity Recognition
│           └── sentiment_analysis.py     # Sentiment analysis
├── frontend/                 # Streamlit frontend
│   ├── Dockerfile
│   ├── app.py                # Streamlit application
│   └── requirements.txt
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # This file
```

## Requirements

- Docker and Docker Compose
- OpenAI API Key or Google Gemini API Key

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Opinion-Analysis-Chatbot.git
cd Opinion-Analysis-Chatbot
```

2. **Configure environment variables**

Create a `.env` file in the project root directory:

```bash
# .env
# Choose one: "openai" or "gemini"
LLM_PROVIDER=openai  

# Only one of these is needed, based on your LLM_PROVIDER choice
OPENAI_API_KEY=your-api-key-here
GEMINI_API_KEY=your-api-key-here
```

3. **Build and start the services**

```bash
docker compose up
```

The first build may take several minutes as it installs all dependencies.

4. **Access the application**

Frontend: [http://localhost:8501](http://localhost:8501)
opinion_analysis API: [http://localhost:8000](http://localhost:8000)

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### Health Check

```
GET /
```

Returns a message indicating the API is running.

**Response**:
```json
{
  "message": "Opinion Analysis Chatbot API is running"
}
```

#### Process Chat Message

```
POST /chat
```

Processes a user message and returns a response based on opinion analysis of web news.

**Request Body**:
```json
{
  "message": "What is the public sentiment about AI adoption?",
  "conversation_id": "optional-conversation-id"
}
```

**Response**:
```json
{
  "response": "Based on analysis of recent news articles...",
  "conversation_id": "conversation-id"
}
```

### Response Format

For opinion-related queries, responses include:
- Overview of findings
- Key sentiment trends
- Important entities mentioned
- Detailed insights
- Conclusion

## Implementation Details

### opinion_analysis

- **FastAPI**: Web framework for the API
- **LangGraph**: Workflow orchestration for the agent
- **OpenAI/Gemini**: LLM providers for text processing
- **newspaper3k**: News article scraping
- **spaCy**: Named entity recognition
- **Transformers**: Sentiment analysis

### Frontend

- **Streamlit**: User interface for interacting with the chatbot

## Troubleshooting

### Common Issues

1. **Container fails to start**:
   - Check if ports 8000 and 8501 are already in use
   - Verify your API keys are correctly set in the .env file

2. **News scraping fails**:
   - Some websites block web scraping; try different queries

3. **"Not opinion-related" responses**:
   - The chatbot is designed to only respond to opinion analysis questions
   - Try rephrasing your query to focus on public opinion aspects

## License

[Include your license information here]

## Contact

[Your contact information]
