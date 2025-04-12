# Main FastAPI application file
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import asyncio
from app.agents.opinion_analysis_agent import OpinionAnalysisAgent
from app.utils.logger import setup_logger

# Set up logger
logger = setup_logger("main_api")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Opinion Analysis Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent = OpinionAnalysisAgent()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@app.get("/")
async def root():
    return {"message": "Opinion Analysis Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Traditional non-streaming chat endpoint"""
    try:
        # Get the generator for streaming responses
        response_generator = agent.process_message(
            request.message, request.conversation_id
        )
        
        # Consume the generator and get the final response
        final_response = None
        conversation_id = None
        
        async for response, conv_id in response_generator:
            final_response = response
            conversation_id = conv_id
            
        return ChatResponse(response=final_response, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Streaming chat endpoint that yields real-time updates"""
    try:
        async def event_generator():
            """Generator for server-sent events"""
            response_generator = agent.process_message(
                request.message, request.conversation_id
            )
            
            async for response, conversation_id in response_generator:
                # Format as a server-sent event
                data = json.dumps({
                    "response": response,
                    "conversation_id": conversation_id,
                    "is_complete": False  # Flag to identify intermediate updates
                })
                yield f"data: {data}\n\n"
                
                # Brief pause to avoid overwhelming the client
                await asyncio.sleep(0.1)
            
            # Send a final event signaling completion
            final_data = json.dumps({
                "is_complete": True
            })
            yield f"data: {final_data}\n\n"
                
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
