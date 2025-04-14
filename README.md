# Opinion Analysis Chatbot

[![IMAGE ALT TEXT HERE]([https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE](https://youtu.be/PIbQ0CPNqEE))

## 1. Introduction

Opinion Analysis Chatbot is a specialized AI-powered assistant designed to analyze public opinion, sentiment trends, and news coverage. The system combines natural language processing with web scraping capabilities to provide comprehensive analyses of online discourse and news content.

This chatbot is particularly useful for:

- Market researchers analyzing public sentiment
- PR professionals tracking brand reception
- Journalists researching public opinion on specific topics
- Analysts monitoring opinion trends over time

The system features a modular architecture with a Streamlit frontend and a workflow-based backend utilizing LangGraph for orchestration and LLM integration.

## 2. Requirements

### Software Requirements

- Docker and Docker Compose

### API Keys

- OpenAI API key (for GPT model access) see [Open API keys](https://platform.openai.com/api-keys)

### Python Dependencies

Key dependencies include:

- langchain==0.1.0
- langgraph==0.0.17
- langchain-openai==0.3.12
- langchain-community>=0.0.10
- langchain-core>=0.1.10
- newspaper3k>=0.2.8
- spacy>=3.7.0
- streamlit>=1.24.0
- pydantic==2.3.0

For a complete list of dependencies, see `opinion_analysis/requirements.txt`.

## 3. Setup

### Using Docker (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Opinion-Analysis-Chatbot.git
cd Opinion-Analysis-Chatbot
```

2. Copy the `.env.example` to `.env` file in the project root with pasting your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

3. Noted that you can restrict the workflow to search only a local region news, see [information](https://duckduckgo.com/duckduckgo-help-pages/settings/params/) for getting the region code and paste it to the `.env`

```
# For Taiwan
REGION=tw-tzh
```

4. Copy the environment variables with vectorstore information:

```bash
DOCUMENTS_PATH=fixtures
VECTORSTORE_PATH=fixtures/vector_db
```

5. Build the docker image:

```bash
docker build -f opinion_analysis/Dockerfile -t opinion-analysis:develop .
```

6. Please follow the fixtures [README.md](opinion_analysis/fixtures/README.md) of the information for placing the `.md` file under the `opinion_analysis/fixtures` which you want to store and search with vectorstore.

```bash
fixtures
├── README.md
├── <your markdown file>
├── ...
└── ...
```

7. Run the server and build up vectorstore

```bash
docker-compose up &
docker-compose exec server python scripts/build_vectorstore.py
```

8. Access the application:
   - Opinion Analysis Chatbot: [http://localhost:8501](http://localhost:8501)
   - RAG Chatbot: [http://localhost:8501/chatbot_rag](http://localhost:8501/chatbot_rag)

## 4. Test with Chat Web

Once the application is running, you can interact with the Opinion Analysis Chatbot through the Streamlit web interface:

1. Navigate to [http://localhost:8501](http://localhost:8501) of [http://localhost:8501/chatbot_rag](http://localhost:8501/chatbot_rag) in your web browser.
2. Follow the page instructions and enter your query in the chat input field at the bottom of the page.
3. The chatbot will process your query, search for relevant news articles, and provide an analysis.

### Example Queries

Try asking questions like:

- For opinion analysis:
  - 請協助我搜集去年總統大選的新聞。
  - 替我分析 2025 立法委員大罷免的新聞。
- For self RAG (Based on the default markdown expamle):
  - KEYPO 的「熱門關鍵字」是如何計算出來的？
  - 使用 KEYPO 的「警報信」功能時，使用者可以自訂哪些設定？
- For other non-relevant question:
  - 今天中午妳想吃什麼當作午餐？

### Features

- **Contextual Memory**: The chatbot maintains conversation history using LangGraph's thread system
- **Real-time Web Search**: Searches current news sources for relevant articles
- **Sentiment Analysis**: Analyzes the emotional tone of news articles
- **Named Entity Recognition**: Identifies key people, organizations, and concepts
- **Summary Generation**: Creates concise summaries of complex articles
- **Self Retrieval-Augmented Generation**~: For searching, analyzing relevance from documents and double checking reponse, to provide the high quality document-based knowledgement.

## 5. Contact

For questions, issues, or contributions please contact:

Weber Huang
Email: doudi853@gmail.com
GitHub: [Weber12321](https://github.com/Weber12321)

---

**Note**: This project is designed for educational and research purposes. The accuracy of sentiment analysis and opinion extraction depends on the quality of sources available and the capabilities of the underlying language models.
