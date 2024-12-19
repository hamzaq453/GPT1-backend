from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Get the GEMINI_API_KEY from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Replace with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2
)

# Conversation context (shared for now, could be scoped to users)
context_memory = []

# Request model
class QueryRequest(BaseModel):
    query: str
    context_enabled: bool = False
    detailed: bool = False  # Optional flag to request detailed responses

# Function to decompose queries
def decompose_query_with_context(query: str) -> List[str]:
    subtasks = [fragment.strip() for fragment in query.replace("'", "").split("and")]
    for i in range(1, len(subtasks)):
        if "which" in subtasks[i] or "when" in subtasks[i]:
            subtasks[i] = f"{subtasks[i - 1].strip()} {subtasks[i].strip()}"
    return subtasks

# Function to fetch data for a single subtask
def fetch_subtask_data_sync(subtask: str, context: str = None, detailed: bool = False) -> dict:
    try:
        # Add formatting and response length control to the prompt
        format_instruction = (
            "Provide the response with proper formatting including bullet points, headings, one line space and whitespace."
        )
        length_instruction = (
            "Keep the response concise and under 300 words unless explicitly asked for a detailed answer."
        )
        if detailed:
            length_instruction = "Provide a detailed and comprehensive response."

        prompt = f"{format_instruction}\n{length_instruction}\n\nQuery: {subtask}"
        if context:
            prompt = f"Context: {context}\n{prompt}"

        response = llm.invoke(prompt)
        return {"subtask": subtask, "result": response.content}
    except Exception as e:
        return {"subtask": subtask, "error": str(e)}

# Endpoint: Handle User Query
@app.post("/query")
async def handle_query(request: QueryRequest):
    global context_memory

    # Determine context usage
    context = context_memory[-1] if request.context_enabled and context_memory else None

    # Decompose query
    subtasks = decompose_query_with_context(request.query)
    results = []

    # Execute subtasks in parallel
    with ThreadPoolExecutor(max_workers=len(subtasks)) as executor:
        future_to_subtask = {
            executor.submit(
                fetch_subtask_data_sync,
                subtask,
                context,
                request.detailed,
            ): subtask
            for subtask in subtasks
        }
        for future in concurrent.futures.as_completed(future_to_subtask):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"subtask": future_to_subtask[future], "error": str(e)})

    # Extract final response
    final_response = "\n\n".join(
        [result["result"] for result in results if "result" in result]
    )

    # Update context memory
    context_memory.append(final_response)

    # Return final response
    return {"response": final_response, "subtasks": results}
